"""Session-preserving replay overlays on top of an existing compact trace."""

import argparse
from bisect import bisect_right
from collections import defaultdict
from hashlib import sha256
from heapq import merge
from itertools import islice
import json
import os
from pathlib import Path
from struct import Struct
from tempfile import NamedTemporaryFile

from compact_trace import FORMAT, read_shard
from workload import TraceRequest


INDEX_FILENAME = "sessions.idx"
INDEX_MAGIC = b"HHMSES1\0"
INDEX_ROW = Struct("<QI")  # original timestamp, integer session ID
ALIGNMENTS = ("first-request", "segment")


def compact_manifest(trace):
    if not trace.is_dir():
        raise ValueError("session overlay requires a compact trace directory")
    path = trace / "manifest.json"
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != FORMAT:
        raise ValueError("session overlay requires a compact trace directory")
    return manifest, INDEX_MAGIC + sha256(manifest_bytes).digest()


def index_sessions(trace, sidecar, output):
    """Save the original timestamps and anonymous session IDs, in trace order."""
    manifest, header = compact_manifest(trace)
    bounds = {}
    next_row = 0
    for shard in manifest["shards"]:
        bounds[next_row] = shard["first_timestamp_ms"]
        next_row += shard["count"]
        bounds[next_row - 1] = shard["last_timestamp_ms"]
    session_ids = {}
    count, previous = 0, -1
    if output.exists():
        raise ValueError(f"session index already exists: {output}")
    with sidecar.open("rb") as source, NamedTemporaryFile(
        dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(header)
            for number, line in enumerate(source):
                row = json.loads(line)
                stamp, session = row["timestamp"], row["session_id"]
                if (row["trace_row"] != number or type(stamp) is not int or stamp < previous
                        or stamp != bounds.get(number, stamp) or not isinstance(session, str) or not session):
                    raise ValueError(f"{sidecar}:{number + 1}: metadata does not match the compact trace")
                identifier = session_ids.setdefault(session, len(session_ids))
                handle.write(INDEX_ROW.pack(stamp, identifier))
                count += 1
                previous = stamp
            if count != manifest["request_count"]:
                raise ValueError("session index row count differs from the compact trace")
            handle.close()
            temporary.replace(output)
        finally:
            if not handle.closed:
                handle.close()
            temporary.unlink(missing_ok=True)
    print(f"Indexed {count:,} requests across {len(session_ids):,} sessions: {output}")


class Overlay:
    """Plan the full overlay, then stream only the selected arrival window."""

    def __init__(self, trace, index, factor, align, skip=0, limit=None):
        self.trace = trace
        self.manifest, header = compact_manifest(trace)
        if factor < 1 or align not in ALIGNMENTS:
            raise ValueError("invalid overlay factor or alignment")
        if not index.is_file():
            raise ValueError(f"session index not found: {index}. Generate it with overlap.py")
        total = self.manifest["request_count"]
        if skip < 0 or limit is not None and limit < 1:
            raise ValueError("skip must be nonnegative and limit must be positive")
        if skip >= total:
            raise ValueError("the selected trace range contains no requests")
        raw = index.read_bytes()
        if not raw.startswith(header) or len(raw) != len(header) + total * INDEX_ROW.size:
            raise ValueError("session index does not match the compact trace manifest")
        self.records = list(INDEX_ROW.iter_unpack(memoryview(raw)[len(header):]))
        first = self.records[0][0]
        span = self.records[-1][0] - first + 1
        width = (span + factor - 1) // factor
        sessions, self.origins = {}, {}
        groups = defaultdict(list)
        for number, (stamp, session) in enumerate(self.records):
            segment = (stamp - first) // width
            group = sessions.setdefault(session, segment)
            self.origins.setdefault(group, stamp if align == "first-request" else first + group * width)
            groups[group].append(number)
        replay_span = max(self.records[numbers[-1]][0] - self.origins[group]
                          for group, numbers in groups.items()) + 1

        # Merge timestamps and row numbers from the index, without opening KV shards.
        stop = total if limit is None else min(total, skip + limit)
        arrivals = merge(*(self._group_arrivals(group, numbers) for group, numbers in groups.items()))
        selected = islice(arrivals, skip, stop)
        selected_first, number, group = next(selected)
        self.groups = defaultdict(list, {group: [number]})
        last = selected_first
        selected_sessions = {self.records[number][1]}
        tail_requests = int(selected_first >= width)
        for last, number, group in selected:
            self.groups[group].append(number)
            selected_sessions.add(self.records[number][1])
            tail_requests += last >= width
        selected_count = stop - skip
        selected_span = last - selected_first + 1
        segment_ids = sorted(self.groups)

        self.ends = []
        next_row = 0
        for shard in self.manifest["shards"]:
            next_row += shard["count"]
            self.ends.append(next_row)
        self.summary = {
            "factor": factor,
            "alignment": align,
            "selection_order": "after-overlay",
            "segment_ms": width,
            "active_segments": len(self.groups),
            "session_count": len(selected_sessions),
            "source_request_count": total,
            "source_span_ms": span,
            "replay_span_ms": replay_span,
            "effective_density_multiplier": span / replay_span,
            "selected_request_count": selected_count,
            "selected_first_arrival_ms": selected_first,
            "selected_last_arrival_ms": last,
            "selected_span_ms": selected_span,
            "selected_density_multiplier": (selected_count / selected_span) / (total / span),
            "requests_per_segment": [len(self.groups[group]) for group in segment_ids],
            "segment_ids": segment_ids,
            "tail_requests": tail_requests,
        }

    def _group_arrivals(self, group, numbers):
        origin = self.origins[group]
        for number in numbers:
            yield self.records[number][0] - origin, number, group

    def _group_records(self, group, block_tokens, time_scale):
        numbers = self.groups[group]
        origin = self.origins[group]
        position = 0
        while position < len(numbers):
            wanted = numbers[position]
            shard_number = bisect_right(self.ends, wanted)
            shard_start = self.ends[shard_number - 1] if shard_number else 0
            shard_end = self.ends[shard_number]
            file = self.trace / self.manifest["shards"][shard_number]["file"]
            for number, (stamp, prompt, output, keys, share) in enumerate(read_shard(file, block_tokens), shard_start):
                if number != wanted:
                    continue
                if stamp != self.records[number][0]:
                    raise ValueError(f"{file}: timestamp in session index differs at trace row {number}")
                shifted = stamp - origin
                at = shifted * time_scale / 1000
                yield shifted, number, TraceRequest(number, at, prompt, output, keys, share)
                position += 1
                if position == len(numbers):
                    return
                wanted = numbers[position]
                if wanted >= shard_end:
                    break
            else:
                raise ValueError(f"{file}: cannot find requested trace row {wanted}")

    def requests(self, block_tokens, time_scale):
        streams = [self._group_records(group, block_tokens, time_scale) for group in self.groups]
        for _, _, request in merge(*streams):
            yield request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a lightweight session index for a compact trace.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("sidecar", type=Path, help="original exact source_requests.jsonl")
    parser.add_argument("--output", type=Path, help="defaults to sessions.idx inside the trace directory")
    args = parser.parse_args()
    try:
        index_sessions(args.trace, args.sidecar, args.output or args.trace / INDEX_FILENAME)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Index generation failed: {error}\n")
