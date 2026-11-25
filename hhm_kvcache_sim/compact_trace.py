"""Convert an exact Mooncake trace to gzip shards of precomputed KV block keys."""

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import gzip
from hashlib import blake2b
import json
from pathlib import Path
from struct import Struct
import sys


FORMAT = "hhm-prefix-keys-v1"
# timestamp_ms, input_length, output_length, shared keys with previous row, share request
RECORD = Struct("<QIIII")
KEY_BYTES = 16
DAY_MS = 86_400_000
EPOCH = date(1970, 1, 1)


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def reference_time(source, manifest):
    """Unix ms corresponding to timestamp 0, for grouping by UTC date."""
    sidecar = source / "source_requests.jsonl"
    if sidecar.exists():
        with sidecar.open("rb") as handle:
            first = json.loads(handle.readline())
        wall = first["first_response_event_at"]
        relative = first["timestamp"]
    else:
        wall = manifest["first_response_event_at"]
        relative = manifest["first_timestamp_ms"]
    stamp = datetime.fromisoformat(wall.replace("Z", "+00:00"))
    return round(stamp.timestamp() * 1000) - relative


def prepare_keys(row, width, old_tokens, old_keys):
    """Reuse previous hashes only where the actual token bytes match."""
    prompt, output = row["input_token_ids"], row["output_token_ids"]
    if len(prompt) != row["input_length"] or len(output) != row["output_length"]:
        raise ValueError("trace lengths disagree with the number of token IDs")
    tokens = array("I", prompt)
    tokens.extend(output[:-1])
    if sys.byteorder != "little":
        tokens.byteswap()
    raw = tokens.tobytes()
    complete = len(raw) // width
    # Prefix byte comparisons run in native code rather than a Python token loop.
    low, high = 0, min(complete, len(old_keys))
    while low < high:
        mid = (low + high + 1) // 2
        end = mid * width
        if raw.startswith(old_tokens[:end]):
            low = mid
        else:
            high = mid - 1
    shared = low
    keys = old_keys[:shared]
    parent = keys[-1] if keys else bytes(KEY_BYTES)
    for offset in range(shared * width, complete * width, width):
        digest = blake2b(parent, digest_size=KEY_BYTES)
        digest.update(raw[offset:offset + width])
        parent = digest.digest()
        keys.append(parent)
    return raw, keys, shared


def convert_part(source, target, index, begin, end, block_tokens, origin, level):
    target = Path(target)
    receipt = target / ".work" / f"{index:04d}.json"
    if receipt.exists():
        result = json.loads(receipt.read_text(encoding="utf-8"))
        if all((target / shard["file"]).stat().st_size == shard["bytes"] for shard in result["shards"]):
            return result

    shards = []
    day, archived = None, None
    old_tokens, old_keys = b"", []
    width = block_tokens * array("I").itemsize

    def finish_shard():
        archived.close()
        final = temporary.with_suffix("")
        temporary.replace(final)
        shard["bytes"] = final.stat().st_size
        shards.append(shard)

    try:
        with Path(source).open("rb", buffering=1024 * 1024) as handle:
            if begin:
                handle.seek(begin - 1)
                handle.readline()
            processed_at = handle.tell()
            while processed_at < end:
                line = handle.readline()
                if not line:
                    break
                row = json.loads(line)
                timestamp = row["timestamp"]
                current_day = (origin + timestamp) // DAY_MS
                if day != current_day:
                    if archived is not None:
                        finish_shard()
                    day = current_day
                    utc_day = (EPOCH + timedelta(days=day)).isoformat()
                    name = f"{utc_day}.part{index:04d}.hkv.gz"
                    temporary = target / (name + ".tmp")
                    archived = gzip.open(temporary, "wb", compresslevel=level)
                    shard = {"date_utc": utc_day, "file": name, "count": 0,
                             "first_timestamp_ms": timestamp, "last_timestamp_ms": timestamp}
                    old_tokens, old_keys = b"", []
                old_tokens, keys, shared = prepare_keys(row, width, old_tokens, old_keys)
                archived.write(RECORD.pack(timestamp, row["input_length"], row["output_length"],
                                           shared, row.get("share_prefix_blocks", 0)))
                archived.write(b"".join(keys[shared:]))
                old_keys = keys
                shard["count"] += 1
                shard["last_timestamp_ms"] = timestamp
                processed_at += len(line)
    finally:
        if archived is not None and not archived.closed:
            archived.close()
    if archived is not None:
        # The file was closed in finally, but still needs publishing and a receipt.
        finish_shard()
    result = {"index": index, "shards": shards}
    json_write(receipt, result)
    return result


def convert(source, target, workers, chunk_mib, level, timestamp_zero_utc=None):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents:
        raise ValueError("use a separate output directory")
    src_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not src_manifest.get("includes_input_token_ids") or not src_manifest.get("includes_output_token_ids"):
        raise ValueError("conversion requires input and output token IDs")
    path = source / "requests.jsonl"
    size = path.stat().st_size
    width = chunk_mib * 1024 * 1024
    if timestamp_zero_utc:
        stamp = datetime.fromisoformat(timestamp_zero_utc.replace("Z", "+00:00"))
        if stamp.utcoffset() is None:
            raise ValueError("timestamp origin must include a UTC offset")
        origin = round(stamp.timestamp() * 1000)
    else:
        origin = reference_time(source, src_manifest)
    identity = {
        "format": FORMAT, "source": str(source), "source_bytes": size,
        "source_mtime_ns": path.stat().st_mtime_ns, "trace_block_size": src_manifest["trace_block_size"],
        "chunk_bytes": width, "gzip_level": level, "timestamp_zero_unix_ms": origin,
    }
    journal = target / ".work"
    if (target / "manifest.json").exists():
        raise ValueError(f"already converted: {target}")
    if target.exists() and any(target.iterdir()) and not (journal / "source.json").exists():
        raise ValueError(f"output directory is not empty: {target}")
    journal.mkdir(parents=True, exist_ok=True)
    marker = journal / "source.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != identity:
        raise ValueError("resume requires the same source, chunk size and compression level")
    json_write(marker, identity)
    parts = [(index, offset, min(size, offset + width))
             for index, offset in enumerate(range(0, size, width))]
    results = {}
    rows = total_output = 0
    print(f"Converting {size:,} bytes in {len(parts)} parts using {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(convert_part, path, target, index, begin, end,
                             identity["trace_block_size"], origin, level) for index, begin, end in parts]
        for task in as_completed(tasks):
            converted = task.result()
            results[converted["index"]] = converted["shards"]
            rows += sum(shard["count"] for shard in converted["shards"])
            total_output += sum(shard["bytes"] for shard in converted["shards"])
            print(f"[{len(results)}/{len(parts)}] {rows:,} requests; {total_output / 1e9:.2f} GB written", flush=True)
    shards = [shard for index in range(len(parts)) for shard in results[index]]
    if rows != src_manifest["request_count"] or any(
        prev["last_timestamp_ms"] > current["first_timestamp_ms"]
        for prev, current in zip(shards, shards[1:])
    ):
        raise ValueError("converted count or timestamp order disagrees with source")
    if path.stat().st_mtime_ns != identity["source_mtime_ns"] or path.stat().st_size != size:
        raise ValueError("source was modified during conversion")
    manifest = {**identity, "request_count": rows, "bytes": total_output,
                "first_timestamp_ms": shards[0]["first_timestamp_ms"],
                "last_timestamp_ms": shards[-1]["last_timestamp_ms"], "shards": shards}
    json_write(target / "manifest.json", manifest)
    print(f"Complete: {rows:,} requests, {len(shards)} shards, {total_output:,} bytes", flush=True)


def read_shard(path, block_tokens):
    """Yield compact rows using the same integer keys as the exact JSONL reader."""
    previous = ()
    with gzip.open(path, "rb") as source:
        while header := source.read(RECORD.size):
            if len(header) != RECORD.size:
                raise ValueError(f"{path}: truncated record")
            timestamp, prompt, output, shared, consumer = RECORD.unpack(header)
            full_blocks = (prompt + max(0, output - 1)) // block_tokens
            if shared > min(full_blocks, len(previous)):
                raise ValueError(f"{path}: invalid shared prefix length")
            size = (full_blocks - shared) * KEY_BYTES
            suffix = source.read(size)
            if len(suffix) != size:
                raise ValueError(f"{path}: truncated keys")
            keys = previous[:shared] + tuple(
                int.from_bytes(suffix[index:index + KEY_BYTES], "little")
                for index in range(0, size, KEY_BYTES)
            )
            previous = keys
            yield timestamp, prompt, output, keys, consumer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=512)
    parser.add_argument("--level", type=int, choices=range(1, 10), default=3)
    parser.add_argument("--timestamp-zero-utc", help="absolute trace origin if the source metadata lacks it")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_mib < 1:
        parser.error("workers and chunk-mib must be positive")
    convert(args.source, args.target, args.workers, args.chunk_mib, args.level, args.timestamp_zero_utc)
