"""Copy a contiguous slice of an exact Mooncake trace with its token IDs."""

import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import tempfile


def sample(source, output, count, skip=0):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("source and output directories must differ")
    if count < 1 or skip < 0:
        raise ValueError("count must be positive and skip nonnegative")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("includes_input_token_ids") or not manifest.get("includes_output_token_ids"):
        raise ValueError("source must be an exact trace with both input and output IDs")
    if (output / "requests.jsonl").exists() or (output / "manifest.json").exists():
        raise ValueError(f"output sample already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as directory:
        staging = Path(directory)
        with (source / "requests.jsonl").open("rb") as inp, (staging / "requests.jsonl").open("wb") as out:
            copied = 0
            for line in islice(inp, skip, skip + count):
                out.write(line)
                copied += 1
        if copied != count:
            raise ValueError(f"only {copied} of {count} requested records are available")
        provenance = {key: manifest[key] for key in (
            "format", "trace_block_size", "includes_input_token_ids", "includes_output_token_ids",
            "context_limit", "model_mapping", "timestamp_policy",
        ) if key in manifest}
        provenance.update(source=str(source), first_trace_row=skip, request_count=copied)
        (staging / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        for name in ("requests.jsonl", "manifest.json"):
            shutil.move(staging / name, output / name)
    print(f"Copied {copied} requests to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    sample(args.source, args.output, args.count, args.skip)
