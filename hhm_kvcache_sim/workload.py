"""Configuration and streaming Mooncake trace input."""

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from array import array
from hashlib import blake2b
import json
import math
import sys

from compact_trace import FORMAT, read_shard


@dataclass(frozen=True)
class Config:
    machines: int = 8
    ranks_per_machine: int = 8
    dram_bytes: int = 24_000_000_000
    win_bytes: int = 211_000_000_000
    asu_devices: int = 12
    asu_device_bytes: int = 64_000_000_000_000
    asu_device_bytes_per_s: int = 40_000_000_000
    win_bytes_per_s: int = 800_000_000_000
    dram_bytes_per_s: int = 6_400_000_000_000
    kv_bytes_per_token: int = 50_000
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 32768
    prefill_tokens_per_s: float = 2000
    mtp_probabilities: tuple = (0.05, 0.10, 0.20, 0.25, 0.25, 0.15)
    mtp_width: int = 6
    tau_s: float = 60
    theta_asu: float = 0.5
    theta_cold: float = 1.0
    theta_hot: float = 1.8
    min_win_residence_s: float = 100
    maintenance_budget_s: float = 0.05
    transfer_overlap: float = 0.8
    routing_policy: str = "estimated_ttft"
    time_scale: float = 1.0
    seed: int = 0

    def validate(self):
        if self.routing_policy not in ("estimated_ttft", "cache_first"):
            raise ValueError("routing_policy must be estimated_ttft or cache_first")
        positive = (
            self.machines, self.ranks_per_machine, self.dram_bytes,
            self.asu_devices, self.asu_device_bytes_per_s, self.win_bytes_per_s,
            self.dram_bytes_per_s, self.kv_bytes_per_token, self.prefill_tokens_per_s,
            self.tau_s, self.time_scale, self.max_num_batched_tokens,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("cluster size, DRAM, bandwidths, rates and time scale must be positive")
        if not 1 <= self.max_num_seqs <= 64 or not 1 <= self.mtp_width <= 6:
            raise ValueError("max_num_seqs must be 1–64 and mtp_width must be 1–6")
        if self.max_num_batched_tokens < max(self.max_num_seqs, self.mtp_width):
            raise ValueError("the token budget must fit max_num_seqs and one MTP verification")
        if not 0 <= self.theta_asu <= self.theta_cold < self.theta_hot:
            raise ValueError("require 0 <= theta_asu <= theta_cold < theta_hot")
        if not 0 <= self.transfer_overlap <= 1:
            raise ValueError("transfer_overlap must be 0–1")
        if min(self.min_win_residence_s, self.maintenance_budget_s,
               self.win_bytes, self.asu_device_bytes) < 0:
            raise ValueError("capacities and time budgets cannot be negative")
        probabilities = self.mtp_probabilities
        if (len(probabilities) != self.mtp_width
                or any(value < 0 for value in probabilities)
                or not math.isclose(sum(probabilities), 1.0)):
            raise ValueError("provide one probability for each MTP output count, summing to one")


@dataclass(slots=True)
class TraceRequest:
    number: int
    at: float
    input_tokens: int
    output_tokens: int
    keys: tuple[int, ...]
    share_blocks: int = 0

    def prompt_keys(self, block_tokens):
        return self.keys[:self.input_tokens // block_tokens]


def prefix_keys(input_ids, output_ids, block_tokens):
    """Fingerprint each full block and its preceding context using exact tokens."""
    tokens = array("I", input_ids)
    # The last emitted token has no KV until it is fed back into a later forward.
    tokens.extend(output_ids[:-1])
    if sys.byteorder != "little":
        tokens.byteswap()
    raw = memoryview(tokens).cast("B")
    width = block_tokens * tokens.itemsize
    parent = bytes(16)
    keys = []
    for offset in range(0, len(raw) - width + 1, width):
        digest = blake2b(parent, digest_size=16)
        digest.update(raw[offset:offset + width])
        parent = digest.digest()
        keys.append(int.from_bytes(parent, "little"))
    return tuple(keys)


def get_trace(path, block_tokens=None):
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if path.is_dir() and manifest.get("format") != FORMAT:
        path = path / "requests.jsonl"
    if not path.exists():
        raise ValueError(f"trace not found: {path}")
    declared = manifest.get("trace_block_size")
    if block_tokens is not None and declared is not None and block_tokens != declared:
        raise ValueError(f"block size must match the trace manifest ({declared})")
    size = block_tokens or declared or 16
    if type(size) is not int or size < 1:
        raise ValueError("block size must be a positive integer")
    return path, size


def requests(path, block_tokens, time_scale, skip=0, limit=None):
    """Read exact Mooncake records or their compact precomputed form."""
    if path.is_dir():
        yield from compact_requests(path, block_tokens, time_scale, skip, limit)
        return
    last, origin = -1, None
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(islice(source, skip, None if limit is None else skip + limit), skip):
            try:
                row = json.loads(line)
                stamp, prompt, output = (row[k] for k in ("timestamp", "input_length", "output_length"))
                input_ids, output_ids = row["input_token_ids"], row["output_token_ids"]
                full = prompt // block_tokens
                if (type(stamp) is not int or stamp < 0 or stamp < last
                        or type(prompt) is not int or prompt < 1
                        or type(output) is not int or output < 0
                        or len(input_ids) != prompt or len(output_ids) != output):
                    raise ValueError("invalid token IDs, lengths or timestamp")
                keys = prefix_keys(input_ids, output_ids, block_tokens)
                share = row.get("share_prefix_blocks", 0)
                if type(share) is not int or not 0 <= share <= full:
                    raise ValueError("share_prefix_blocks is outside the complete-block range")
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                raise ValueError(f"{path}:{number + 1}: {error}") from error
            last = stamp
            if origin is None:
                origin = stamp
            yield TraceRequest(number, (stamp - origin) * time_scale / 1000, prompt, output, keys, share)


def compact_requests(path, block_tokens, time_scale, skip, limit):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"unsupported compact trace format: {path}")
    number, origin, last, emitted = 0, None, -1, 0
    for shard in manifest["shards"]:
        if number + shard["count"] <= skip:
            number += shard["count"]
            continue
        file = path / shard["file"]
        count = 0
        for stamp, prompt, output, keys, share in read_shard(file, block_tokens):
            if stamp < last or prompt < 1 or share > prompt // block_tokens:
                raise ValueError(f"{file}: invalid timestamp or record")
            last = stamp
            number += 1
            count += 1
            if number <= skip:
                continue
            if origin is None:
                origin = stamp
            emitted += 1
            yield TraceRequest(number - 1, (stamp - origin) * time_scale / 1000,
                               prompt, output, keys, share)
            if limit is not None and emitted >= limit:
                return
        if count != shard["count"]:
            raise ValueError(f"{file}: count differs from manifest")


class IdealPrefix:
    """Unlimited, instant reuse of full blocks from earlier arrivals."""

    def __init__(self):
        self.seen = set()

    def observe(self, prompt_keys, context_keys):
        shared = 0
        for key in prompt_keys:
            if key not in self.seen:
                break
            shared += 1
        self.seen.update(context_keys[shared:])
        return shared
