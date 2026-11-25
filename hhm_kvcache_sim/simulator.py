"""Forward-granularity simulation: independent machines, synchronized local ranks."""

from bisect import bisect_left
from collections import Counter, deque
from dataclasses import dataclass, field
import heapq
import math
import random

from cache import ASU, LocalCache, Transfer
from workload import IdealPrefix, TraceRequest


DECODE_MS = (
    (1, 40.6), (4, 44.5), (8, 53.2), (12, 59.1), (16, 66.7),
    (20, 73.1), (24, 83.2), (28, 88.6), (32, 97.9), (64, 172.3),
)


def decode_seconds(batch_size):
    if batch_size == 0:
        return 0.0
    if not 1 <= batch_size <= DECODE_MS[-1][0]:
        raise ValueError("decode batch size must be 0–64")
    index = bisect_left(DECODE_MS, (batch_size,))
    right, ms_right = DECODE_MS[index]
    if right == batch_size:
        return ms_right / 1000
    left, ms_left = DECODE_MS[index - 1]
    return (ms_left + (ms_right - ms_left) * (batch_size - left) / (right - left)) / 1000


@dataclass(slots=True)
class Request:
    trace: TraceRequest
    context: int = 0
    output: int = 0
    prefilled: bool = False
    first_token: float | None = None
    reserved: int = 0
    tail: bool = False
    win_blocks: int = 0
    nodes: list = field(default_factory=list)


@dataclass(slots=True)
class Work:
    request: Request
    prefill_end: int | None = None
    output: int = 0
    prefix: list = field(default_factory=list)


@dataclass(slots=True)
class RankStep:
    io: Transfer
    work: list = field(default_factory=list)
    prefill_tokens: int = 0
    decode_tokens: int = 0
    decode_batch: int = 0

    def compute_seconds(self, cfg):
        if self.prefill_tokens:
            return (self.prefill_tokens + self.decode_tokens) / cfg.prefill_tokens_per_s
        return decode_seconds(self.decode_batch) if self.decode_batch else 0.0


class Rank:
    def __init__(self, cfg, asu, block_tokens, rng, metrics):
        self.cfg, self.asu, self.size = cfg, asu, block_tokens
        self.rng, self.metrics = rng, metrics
        self.cache = LocalCache(cfg, asu)
        self.queue = deque()
        self.queued_prefill = self.running_prefill = 0
        self.last_routed = -1
        self.mean_mtp = sum(tokens * probability for tokens, probability in enumerate(cfg.mtp_probabilities, 1))
        self.running = {}
        self.forward_count = self.decode_steps = self.decode_sum = 0
        self.peak_dram = self.peak_win = 0

    def enqueue(self, trace, prefill_tokens):
        self.queue.append((trace, prefill_tokens))
        self.queued_prefill += prefill_tokens

    def estimate_prefill(self, trace, keys):
        cache = self.cache
        dram = win = new_pins = 0
        asu_reads = Counter()
        matched = 0
        for key in keys:
            node = cache.nodes.get(key)
            if node in cache.dram.blocks:
                dram += 1
                new_pins += not node.pinned
            elif node in cache.win.blocks:
                win += 1
            elif self.asu.readable(key):
                asu_reads[key % self.cfg.asu_devices] += self.asu.block_bytes
            else:
                break
            matched += 1
        prefill = max(1, trace.input_tokens - matched * self.size)
        required = math.ceil((trace.input_tokens + max(0, trace.output_tokens - 1)) / self.size) - dram - win
        return prefill, required, dram, new_pins, asu_reads

    def estimate_release(self, inflight):
        """Estimate the first running request to release its sequence and KV."""
        issued = {work.request.trace.number: work for work in inflight.work} if inflight else {}
        step_seconds = decode_seconds(len(self.running))
        earliest = math.inf
        for request in self.running.values():
            work = issued.get(request.trace.number)
            context = work.prefill_end if work and work.prefill_end is not None else request.context
            output = request.output + (work.output if work else 0)
            if context >= request.trace.input_tokens and output >= request.trace.output_tokens:
                return 0.0
            prefill_seconds = max(0, request.trace.input_tokens - context) / self.cfg.prefill_tokens_per_s
            decode_left = max(0, request.trace.output_tokens - max(1, output))
            earliest = min(earliest, prefill_seconds + math.ceil(decode_left / self.mean_mtp) * step_seconds)
        return earliest

    def admit(self, trace, now, io):
        prefix = []
        for key in trace.prompt_keys(self.size):
            tier = self.cache.tier_at(key)
            if tier is not None:
                self.cache.pin(self.cache.nodes[key])
                prefix.append((key, "dram" if tier is self.cache.dram else "win"))
            elif self.asu.lookup_lock(key):
                prefix.append((key, "asu"))
            else:
                break
        counts = Counter(source for _, source in prefix)
        total = trace.input_tokens + max(0, trace.output_tokens - 1)
        required = math.ceil(total / self.size) - counts["dram"] - counts["win"]
        if required > self.cache.dram.capacity or not self.cache.make_room(required, now, io):
            for key, source in prefix:
                if source == "asu":
                    self.asu.unlock(key)
                else:
                    self.cache.unpin(self.cache.nodes[key])
            return None, []
        self.cache.reserved += required
        request = Request(trace, context=len(prefix) * self.size, reserved=required)
        self.running[trace.number] = request
        self.running_prefill += max(1, trace.input_tokens - request.context)
        for key, source in prefix:
            if source == "asu":
                io.locks.append(key)
                io.read_asu(key)
            else:
                self.cache.nodes[key].hit(now, self.cfg.tau_s)
        for name in ("dram", "win", "asu"):
            self.metrics.counts[name + "_hit_tokens"] += counts[name] * self.size
        if self.metrics.series is not None:
            self.metrics.series.admission(now, trace.at, trace.input_tokens,
                                          (counts["dram"] + counts["win"]) * self.size,
                                          counts["asu"] * self.size)
        return request, prefix

    def _prefill(self, step, request, prefix, budget):
        remaining = request.trace.input_tokens - request.context
        # Even a completely cached prompt needs the final logits.
        count = min(budget, max(1, remaining))
        end = request.context + min(count, remaining)
        self.running_prefill -= count
        step.prefill_tokens += count
        step.work.append(Work(request, prefill_end=end, prefix=prefix))
        return budget - count

    def start(self, now):
        step = RankStep(Transfer(self.cfg, self.asu, self.metrics.counts))
        budget = self.cfg.max_num_batched_tokens
        existing = tuple(self.running.values())
        for request in existing:
            if not request.prefilled and budget:
                budget = self._prefill(step, request, [], budget)
        blocked = []
        while budget and self.queue and len(self.running) < self.cfg.max_num_seqs:
            trace, estimate = self.queue.popleft()
            request, prefix = self.admit(trace, now, step.io)
            if request is None:
                blocked.append((trace, estimate))
            else:
                self.queued_prefill -= estimate
                budget = self._prefill(step, request, prefix, budget)
        self.queue.extendleft(reversed(blocked))
        for request in existing:
            if request.prefilled:
                left = request.trace.output_tokens - request.output
                cost = min(self.cfg.mtp_width, left)
                if cost > budget:
                    continue
                emitted = self.rng.choices(range(1, self.cfg.mtp_width + 1),
                                           self.cfg.mtp_probabilities)[0]
                step.work.append(Work(request, output=min(emitted, left)))
                step.decode_batch += 1
                step.decode_tokens += cost
                budget -= cost
        for work in step.work:
            request = work.request
            share = request.trace.share_blocks
            if share and work.prefill_end == request.trace.input_tokens:
                parent = None
                for key in request.trace.keys[:share]:
                    parent = self.cache.node(key, parent)
                if not step.io.backup(parent, self.cache, set(request.trace.keys[:share])):
                    self.metrics.counts["explicit_backup_failures"] += 1
            cached_win = request.win_blocks
            if work.prefix:
                cached_win += sum(source == "win" for _, source in work.prefix)
            self.metrics.counts["win_rolling_read_bytes"] += cached_win * self.asu.block_bytes
        return step

    def _consume_reservation(self, request):
        if request.reserved <= 0:
            raise RuntimeError("KV allocation exceeded admission reservation")
        self.cache.reserved -= 1
        request.reserved -= 1

    def _materialize(self, request, context, now, matched=()):
        cache = self.cache
        count = context // self.size
        for index in range(len(request.nodes), count):
            key = request.trace.keys[index]
            node = cache.node(key, request.nodes[-1] if request.nodes else None)
            if node not in cache.dram.blocks and node not in cache.win.blocks:
                if request.tail:
                    cache.private -= 1
                    request.tail = False
                else:
                    self._consume_reservation(request)
                cache.dram.add(node, now)
            if index < len(matched) and matched[index][1] == "asu":
                node.hit(now, self.cfg.tau_s)
            if index >= len(matched) or matched[index][1] == "asu":
                cache.pin(node)
            # A running request pins these nodes, so their WIN residency
            # cannot change until the request finishes.
            request.win_blocks += node in cache.win.blocks
            request.nodes.append(node)
        tail = context % self.size != 0
        if tail and not request.tail:
            self._consume_reservation(request)
            cache.private += 1
        elif request.tail and not tail:
            cache.private -= 1
        request.tail = tail
        request.context = context

    def finish(self, step, now):
        for node in step.io.new_win:
            if node in self.cache.win.blocks:
                self.cache.win.blocks[node] = now
        for work in step.work:
            request, trace = work.request, work.request.trace
            if work.prefill_end is not None:
                self._materialize(request, work.prefill_end, now, work.prefix)
                if work.prefill_end == trace.input_tokens:
                    request.prefilled = True
                    if trace.output_tokens:
                        request.first_token = now
                        request.output = 1
            else:
                request.output += work.output
                context = trace.input_tokens + request.output - 1
                self._materialize(request, context, now)
            self.peak_dram = max(self.peak_dram, len(self.cache.dram.blocks) + self.cache.private)
            if request.prefilled and request.output == trace.output_tokens:
                self.metrics.finished(request, now)
                self.cache.reserved -= request.reserved
                self.cache.private -= request.tail
                for node in reversed(request.nodes):
                    self.cache.unpin(node)
                    self.cache.prune(node)
                del self.running[trace.number]
        self.asu.finish(step.io)
        self.forward_count += 1
        self.decode_sum += step.decode_batch
        self.decode_steps += bool(step.decode_batch)
        self.peak_dram = max(self.peak_dram, len(self.cache.dram.blocks) + self.cache.private)
        self.peak_win = max(self.peak_win, len(self.cache.win.blocks))


class Metrics:
    def __init__(self, series=None):
        self.series = series
        self.counts = Counter()
        self.ttft, self.tpot, self.te2e = [], [], []
        self.ideal = IdealPrefix()

    def arrival(self, trace, block_tokens):
        self.counts["requests"] += 1
        self.counts["prompt_tokens"] += trace.input_tokens
        ideal_tokens = block_tokens * self.ideal.observe(trace.prompt_keys(block_tokens), trace.keys)
        self.counts["ideal_hit_tokens"] += ideal_tokens
        if self.series is not None:
            self.series.arrival(trace.at, trace.input_tokens, ideal_tokens)

    def finished(self, request, now):
        self.counts["completed"] += 1
        self.counts["output_tokens"] += request.output
        te2e = now - request.trace.at
        ttft = tpot = None
        self.te2e.append(te2e)
        if request.first_token is not None:
            ttft = request.first_token - request.trace.at
            self.ttft.append(ttft)
        if request.output > 1:
            tpot = (now - request.first_token) / (request.output - 1)
            self.tpot.append(tpot)
        if self.series is not None:
            self.series.finished(request.trace.at, ttft, tpot, te2e)

    @staticmethod
    def timing(values):
        if not values:
            return {"samples": 0, "mean_ms": None, "p90_ms": None}
        ordered = sorted(values)
        return {"samples": len(values), "mean_ms": 1000 * math.fsum(values) / len(values),
                "p90_ms": 1000 * ordered[math.ceil(0.9 * len(values)) - 1]}


class Simulation:
    def __init__(self, cfg, block_tokens, batch_writer=None, series=None):
        cfg.validate()
        self.cfg, self.size, self.batch_writer = cfg, block_tokens, batch_writer
        self.asu = ASU(cfg, block_tokens * cfg.kv_bytes_per_token)
        self.metrics = Metrics(series)
        rng = random.Random(cfg.seed)
        self.machines = [
            [Rank(cfg, self.asu, block_tokens, rng, self.metrics) for _ in range(cfg.ranks_per_machine)]
            for _ in range(cfg.machines)
        ]
        if not self.machines[0][0].cache.dram.capacity:
            raise ValueError("DRAM must fit at least one KV block")
        self.events, self.busy = [], {}
        self.arrivals_routed = 0
        self.now = 0.0

    def _cache_first_rank(self, trace):
        keys = trace.prompt_keys(self.size)
        best = None
        for machine_id, ranks in enumerate(self.machines):
            hits = [rank.cache.local_prefix(keys) for rank in ranks]
            idx = min(range(len(ranks)), key=lambda i: (-hits[i], len(ranks[i].queue),
                                                        len(ranks[i].running), i))
            score = (-hits[idx], sum(len(rank.queue) for rank in ranks),
                     len(ranks[idx].queue), sum(len(rank.running) for rank in ranks),
                     machine_id, idx)
            if best is None or score < best:
                best = score
        return best[-2:]

    def _ttft_rank(self, trace):
        keys = trace.prompt_keys(self.size)
        rate = self.cfg.prefill_tokens_per_s
        best = None
        for machine, ranks in enumerate(self.machines):
            ready_at, steps = self.busy.get(machine, (self.now, None))
            forward_wait = ready_at - self.now
            for index, rank in enumerate(ranks):
                backlog = (rank.queued_prefill + rank.running_prefill) / rate
                lower_bound = (forward_wait + backlog + 1 / rate, rank.last_routed)
                if best is not None and lower_bound > best[:2]:
                    continue
                prefill, required, dram, new_pins, reads = rank.estimate_prefill(trace, keys)
                cache = rank.cache
                if required + dram > cache.dram.capacity:
                    score = math.inf
                else:
                    admission = 0.0
                    protected = cache.private + cache.reserved + cache.pinned_dram + new_pins
                    if len(rank.running) >= self.cfg.max_num_seqs or required + protected > cache.dram.capacity:
                        admission = rank.estimate_release(steps[index] if steps else None)
                    wait = max(backlog, admission)
                    start = ready_at + wait
                    compute = prefill / rate
                    io = max((max(0.0, self.asu.free_at[device] - start)
                              + size / self.cfg.asu_device_bytes_per_s for device, size in reads.items()), default=0.0)
                    score = forward_wait + wait + compute + max(0.0, io - compute * self.cfg.transfer_overlap)
                candidate = score, rank.last_routed, machine, index, prefill
                if best is None or candidate < best:
                    best = candidate
        return best[2:]

    def route(self, trace):
        if self.cfg.routing_policy == "cache_first":
            machine, index = self._cache_first_rank(trace)
            prefill = max(1, trace.input_tokens)
        else:
            machine, index, prefill = self._ttft_rank(trace)
        rank = self.machines[machine][index]
        rank.enqueue(trace, prefill)
        rank.last_routed = self.arrivals_routed
        self.arrivals_routed += 1
        return machine

    def start(self, machine):
        steps = [rank.start(self.now) for rank in self.machines[machine]]
        if not any(step.work or step.io.writes or step.io.win_copy_s for step in steps):
            return
        compute = max(step.compute_seconds(self.cfg) for step in steps)
        io = max(step.io.schedule(self.now) for step in steps)
        exposed = max(0.0, io - compute * self.cfg.transfer_overlap)
        core_end = self.now + compute + exposed
        post = max(step.io.schedule(core_end, after_compute=True) for step in steps)
        done = core_end + post
        self.busy[machine] = (done, steps)
        heapq.heappush(self.events, (done, machine, self.now, steps))
        self.metrics.counts["machine_forwards"] += 1
        self.metrics.counts["exposed_transfer_seconds"] += exposed + post
        if self.metrics.series is not None:
            for step in steps:
                self.metrics.series.decode(self.now, step.decode_batch)

    def run(self, trace, progress=None):
        source = iter(trace)
        pending = next(source, None)
        ticks = 0
        counts = self.metrics.counts
        while pending is not None or self.events:
            self.now = min(pending.at if pending is not None else math.inf,
                           self.events[0][0] if self.events else math.inf)
            ready = set()
            while self.events and self.events[0][0] == self.now:
                _, machine, begin, steps = heapq.heappop(self.events)
                for rank_id, (rank, step) in enumerate(zip(self.machines[machine], steps)):
                    rank.finish(step, self.now)
                    if self.batch_writer is not None:
                        self.batch_writer.writerow((f"{begin:.6f}", f"{self.now:.6f}", machine, rank_id,
                                                    step.decode_batch, step.prefill_tokens, step.decode_tokens))
                del self.busy[machine]
                ready.add(machine)
            while pending is not None and pending.at == self.now:
                self.metrics.arrival(pending, self.size)
                ready.add(self.route(pending))
                pending = next(source, None)
            for machine in sorted(ready - self.busy.keys()):
                self.start(machine)
            if progress is not None:
                ticks += 1
                if ticks % 128 == 0:
                    progress(counts["completed"], counts["requests"])
        if counts["completed"] != counts["requests"]:
            for ranks in self.machines:
                for rank in ranks:
                    if rank.queue:
                        trace, _ = rank.queue[0]
                        raise ValueError(f"request {trace.number} cannot fit DRAM with the current KV residency/capacity")
            raise RuntimeError("unfinished forward work")
        return self.summary()

    def summary(self):
        metrics, counts = self.metrics, self.metrics.counts
        dram, win, asu = (counts[name + "_hit_tokens"] for name in ("dram", "win", "asu"))
        tokens = counts["prompt_tokens"]

        def rate(value, total):
            return value / total if total else None

        ranks = []
        for machine, group in enumerate(self.machines):
            for number, rank in enumerate(group):
                ranks.append({"machine": machine, "rank": number, "forwards": rank.forward_count,
                              "decode_steps": rank.decode_steps,
                              "mean_decode_batch": rate(rank.decode_sum, rank.decode_steps),
                              "peak_dram_blocks": rank.peak_dram, "peak_win_blocks": rank.peak_win})
        for name in ("win_write_bytes", "asu_write_bytes", "asu_read_bytes", "win_rolling_read_bytes",
                     "dram_duplicate_releases", "dram_frontier_evictions", "win_evictions",
                     "win_promotions", "explicit_backup_failures"):
            counts.setdefault(name, 0)
        return {
            "simulated_seconds": self.now, "block_tokens": self.size,
            "counts": dict(counts), "asu_evictions": self.asu.evictions,
            "hit_rates": {"hhm": rate(dram + win, tokens), "dram": rate(dram, tokens),
                          "win": rate(win, tokens), "asu": rate(asu, tokens),
                          "asu_given_hhm_miss": rate(asu, tokens - dram - win),
                          "ideal": rate(counts["ideal_hit_tokens"], tokens)},
            "ttft": metrics.timing(metrics.ttft), "tpot": metrics.timing(metrics.tpot),
            "te2e": metrics.timing(metrics.te2e), "ranks": ranks,
        }
