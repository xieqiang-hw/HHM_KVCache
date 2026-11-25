"""Per-rank prefix cache and a shared, pinned ASU LRU."""

from collections import Counter, OrderedDict
from dataclasses import dataclass
from itertools import islice
import math


class ASU:
    def __init__(self, cfg, block_bytes):
        self.cfg = cfg
        self.block_bytes = block_bytes
        self.capacity = cfg.asu_devices * cfg.asu_device_bytes // block_bytes
        self.objects = OrderedDict()
        self.locks = Counter()
        self.pending = {}
        self.deletions = set()
        self.free_at = [0.0] * cfg.asu_devices
        self.evictions = 0

    def lookup(self, key):
        return key in self.objects

    def lookup_lock(self, key):
        if not self.readable(key):
            return False
        self.locks[key] += 1
        self.objects.move_to_end(key)
        return True

    def readable(self, key):
        return key in self.objects and key not in self.deletions

    def unlock(self, key):
        self.locks[key] -= 1
        if not self.locks[key]:
            del self.locks[key]
            if key in self.deletions:
                self.deletions.remove(key)
                del self.objects[key]

    def delete(self, key):
        if key in self.locks:
            self.deletions.add(key)
        else:
            self.objects.pop(key, None)

    def reserve_writes(self, keys, transfer):
        """Reserve complete blocks, with visibility delayed to forward completion."""
        extra = len(self.objects) + len(self.pending) + len(keys) - self.capacity
        if extra > len(self.objects) - len(self.locks):
            return False
        if extra > 0:
            victims = list(islice((k for k in self.objects if k not in self.locks), extra))
            for key in victims:
                del self.objects[key]
            self.evictions += extra
        for key in keys:
            self.pending[key] = transfer
        return True

    def finish(self, transfer):
        # One event publishes the entire successful path before unpinning it.
        for key in transfer.writes:
            del self.pending[key]
            self.objects[key] = None
        for key in transfer.locks:
            self.unlock(key)


class Transfer:
    """One rank's I/O in this forward. ASU devices serialize shared traffic."""

    def __init__(self, cfg, asu, counters):
        self.cfg, self.asu, self.counters = cfg, asu, counters
        self.maintenance_left = cfg.maintenance_budget_s
        self.asu_reads = Counter()
        self.maintenance = []
        self.post_compute = []
        self.win_copy_s = 0.0
        self.locks, self.writes = [], []
        self.new_win = []

    def read_asu(self, key, size=None, offset=0):
        size = self.asu.block_bytes - offset if size is None else size
        if (key not in self.asu.locks or offset < 0 or size <= 0
                or offset + size > self.asu.block_bytes):
            raise ValueError("ASU read needs a pinned block and a valid byte range")
        self.asu_reads[key % self.cfg.asu_devices] += size
        self.counters["asu_read_bytes"] += size

    def copy_to_win(self, seconds):
        self.win_copy_s += seconds
        self.maintenance.append((None, seconds))

    def backup(self, node, cache, produced=None):
        """Lock ancestors and reserve the entire missing suffix before writing."""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        acquired, missing = [], []
        success = False
        try:
            for current in reversed(path):
                key = current.key
                if self.asu.pending.get(key) is self:
                    continue
                if self.asu.lookup_lock(key):
                    acquired.append(key)
                elif (key not in self.asu.pending and not self.asu.lookup(key)
                      and (current in cache.dram.blocks or current in cache.win.blocks
                           or produced is not None and key in produced)):
                    missing.append(key)
                else:
                    return False
            size = len(missing) * self.asu.block_bytes
            seconds = size / self.cfg.asu_device_bytes_per_s
            if seconds > self.maintenance_left or not self.asu.reserve_writes(missing, self):
                return False
            self.maintenance_left -= seconds
            self.locks.extend(acquired)
            self.writes.extend(missing)
            timeline = self.maintenance if produced is None else self.post_compute
            block_s = self.asu.block_bytes / self.cfg.asu_device_bytes_per_s
            for key in missing:
                timeline.append((key % self.cfg.asu_devices, block_s))
            self.counters["asu_write_bytes"] += size
            success = True
            return True
        finally:
            if not success:
                for key in acquired:
                    self.asu.unlock(key)

    def schedule(self, start, after_compute=False):
        timeline = self.post_compute if after_compute else self.maintenance
        end = start
        # A WIN write follows the evictions that freed its destination.
        # Writes along one ASU path retain their ancestor-before-child order.
        for device, seconds in timeline:
            if device is not None:
                end = max(end, self.asu.free_at[device])
            end += seconds
            if device is not None:
                self.asu.free_at[device] = end
        if after_compute:
            return end - start
        reads_start = end
        for device, size in self.asu_reads.items():
            done = max(reads_start, self.asu.free_at[device]) + size / self.cfg.asu_device_bytes_per_s
            self.asu.free_at[device] = done
            end = max(end, done)
        return end - start


@dataclass(eq=False, slots=True)
class Node:
    key: int
    parent: "Node | None"
    children: int = 0
    dram_branches: int = 0
    win_branches: int = 0
    pinned: int = 0
    score: float = 0.0
    updated: float = 0.0

    def heat(self, now, tau):
        return self.score * math.exp(-(now - self.updated) / tau)

    def hit(self, now, tau):
        self.score = self.heat(now, tau) + 1
        self.updated = now


class Tier:
    def __init__(self, name, capacity):
        self.branch_count = name + "_branches"
        self.capacity = capacity
        self.blocks = {}     # node -> time the copy became resident
        self.frontier = {}   # resident nodes without a resident descendant

    def _refresh(self, node):
        if node in self.blocks and not getattr(node, self.branch_count):
            self.frontier[node] = None
        else:
            self.frontier.pop(node, None)

    def _propagate(self, node, change):
        parent = node.parent
        while parent is not None:
            before = getattr(parent, self.branch_count)
            after = before + change
            setattr(parent, self.branch_count, after)
            self._refresh(parent)
            if parent in self.blocks or bool(before) == bool(after):
                break
            parent = parent.parent

    def add(self, node, now):
        if node in self.blocks:
            return
        self.blocks[node] = now
        self._refresh(node)
        if not getattr(node, self.branch_count):
            self._propagate(node, 1)

    def remove(self, node):
        del self.blocks[node]
        self._refresh(node)
        if not getattr(node, self.branch_count):
            self._propagate(node, -1)


class LocalCache:
    def __init__(self, cfg, asu):
        self.cfg, self.asu = cfg, asu
        self.dram = Tier("dram", cfg.dram_bytes // asu.block_bytes)
        self.win = Tier("win", cfg.win_bytes // asu.block_bytes)
        self.nodes = {}
        self.reserved = self.private = 0
        self.pinned_dram = 0

    def pin(self, node):
        if not node.pinned and node in self.dram.blocks:
            self.pinned_dram += 1
        node.pinned += 1

    def unpin(self, node):
        node.pinned -= 1
        if not node.pinned and node in self.dram.blocks:
            self.pinned_dram -= 1

    def node(self, key, parent):
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = Node(key, parent)
            if parent is not None:
                parent.children += 1
        elif node.parent is not parent:
            raise ValueError(f"hash {key} refers to different prefixes; use prefix-qualified trace hashes")
        return node

    def tier_at(self, key):
        node = self.nodes.get(key)
        if node in self.dram.blocks:
            return self.dram
        if node in self.win.blocks:
            return self.win
        return None

    def local_prefix(self, keys):
        for index, key in enumerate(keys):
            if self.tier_at(key) is None:
                return index
        return len(keys)

    def prune(self, node):
        while node is not None and not node.children and not node.pinned:
            if node in self.dram.blocks or node in self.win.blocks:
                break
            del self.nodes[node.key]
            node = node.parent
            if node is not None:
                node.children -= 1

    def release(self, tier, node):
        tier.remove(node)
        self.prune(node)

    def reclaim_win(self, now, transfer):
        cfg, win = self.cfg, self.win
        while len(win.blocks) >= win.capacity:
            node = next((n for n in win.frontier
                         if not n.pinned and now - win.blocks[n] >= cfg.min_win_residence_s
                         and n.heat(now, cfg.tau_s) <= cfg.theta_cold), None)
            if node is None:
                return False
            if not self.asu.lookup(node.key) and node.heat(now, cfg.tau_s) >= cfg.theta_asu:
                transfer.backup(node, self)
            self.release(win, node)
            transfer.counters["win_evictions"] += 1
        return True

    def make_room(self, needed, now, transfer):
        def enough():
            return len(self.dram.blocks) + self.private + self.reserved + needed <= self.dram.capacity

        if enough():
            return True
        for node in tuple(self.dram.blocks):
            if not node.pinned and (node in self.win.blocks or self.asu.lookup(node.key)):
                self.release(self.dram, node)
                transfer.counters["dram_duplicate_releases"] += 1
                if enough():
                    return True
        if self.win.capacity:
            copy_s = self.asu.block_bytes / min(self.cfg.win_bytes_per_s, self.cfg.dram_bytes_per_s)
            for node in tuple(self.dram.blocks):
                if transfer.maintenance_left < copy_s:
                    break
                if node.pinned or node.heat(now, self.cfg.tau_s) < self.cfg.theta_hot:
                    continue
                # Set aside the copy budget before WIN recovery can spend on ASU.
                transfer.maintenance_left -= copy_s
                if not self.reclaim_win(now, transfer):
                    transfer.maintenance_left += copy_s
                    break
                transfer.copy_to_win(copy_s)
                self.win.add(node, now)
                transfer.new_win.append(node)
                self.release(self.dram, node)
                transfer.counters["win_write_bytes"] += self.asu.block_bytes
                transfer.counters["win_promotions"] += 1
                if enough():
                    return True
        while not enough():
            node = next((n for n in self.dram.frontier if not n.pinned), None)
            if node is None:
                return False
            if node.heat(now, self.cfg.tau_s) >= self.cfg.theta_asu:
                transfer.backup(node, self)
            self.release(self.dram, node)
            transfer.counters["dram_frontier_evictions"] += 1
        return True
