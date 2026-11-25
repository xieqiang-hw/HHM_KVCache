"""Run the HHM–ASU KV cache simulation on an exact Mooncake JSONL trace."""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

from overlap import ALIGNMENTS, INDEX_FILENAME, Overlay
from progress import Progress, request_count
from simulator import Simulation
from timeseries import TimeSeries
from workload import Config, get_trace, requests


SAMPLE = Path(__file__).resolve().parents[1] / "data/claude-code-main-glm52-128k-exact-sample"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=SAMPLE, help="exact/compact directory or requests.jsonl")
    parser.add_argument("--config", type=Path, help="JSON object overriding Config defaults")
    parser.add_argument("--output", type=Path, required=True, help="directory for results")
    parser.add_argument("--limit", type=int, help="maximum number of requests to replay after overlay")
    parser.add_argument("--skip", type=int, default=0, help="requests to skip in replay order after overlay")
    parser.add_argument("--time-scale", type=float, help="multiply trace interarrival intervals")
    parser.add_argument("--overlay", type=int, default=1, metavar="N",
                        help="overlay N session-preserving time segments")
    parser.add_argument("--overlay-align", choices=ALIGNMENTS, default="segment",
                        help="align segment origins (default), or align each segment's first request")
    parser.add_argument("--session-index", type=Path, help="defaults to sessions.idx in the trace directory")
    parser.add_argument("--block-tokens", type=int, help="block size for a trace with no manifest")
    parser.add_argument("--no-batch-csv", action="store_true", help="omit per-forward CSV; keep the chart and summary")
    parser.add_argument("--no-progress", action="store_true", help="hide real-time replay progress")
    parser.add_argument("--metrics-interval-s", type=int, default=60,
                        help="base time-series window in simulated seconds (default: 60)")
    args = parser.parse_args()
    if args.skip < 0 or args.limit is not None and args.limit < 1:
        parser.error("--skip must be nonnegative and --limit must be positive")
    if args.overlay < 1:
        parser.error("--overlay must be positive")
    try:
        overrides = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        cfg = Config(**overrides)
        if args.time_scale is not None:
            cfg = replace(cfg, time_scale=args.time_scale)
        cfg.validate()
        trace_path, block_tokens = get_trace(args.trace, args.block_tokens)
        overlay = None
        if args.overlay > 1:
            overlay = Overlay(trace_path, args.session_index or trace_path / INDEX_FILENAME,
                              args.overlay, args.overlay_align, args.skip, args.limit)
            arrivals = overlay.requests(block_tokens, cfg.time_scale)
            print(f"Overlay {args.overlay}x ({args.overlay_align}): "
                  f"{overlay.summary['selected_request_count']:,} requests selected after overlay, "
                  f"{overlay.summary['active_segments']} active segments, "
                  f"{overlay.summary['selected_density_multiplier']:.2f}x selected arrival density before time scaling",
                  flush=True)
            total = overlay.summary["selected_request_count"]
        else:
            arrivals = requests(trace_path, block_tokens, cfg.time_scale, args.skip, args.limit)
            total = None if args.no_progress else request_count(trace_path, args.skip, args.limit)
        args.output.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            progress = None if args.no_progress else stack.enter_context(Progress(total))
            writer = None
            if not args.no_batch_csv:
                batch_file = stack.enter_context((args.output / "batches.csv").open(
                    "w", newline="", encoding="utf-8"))
                writer = csv.writer(batch_file)
                writer.writerow(("start_s", "end_s", "machine", "rank", "decode_batch_size",
                                 "prefill_tokens", "decode_tokens"))
            series = TimeSeries(args.metrics_interval_s)
            sim = Simulation(cfg, block_tokens, writer, series)
            summary = sim.run(arrivals, progress=progress.update if progress else None)
            if not summary["counts"].get("requests"):
                raise ValueError("the selected trace range contains no requests")
            if progress:
                progress.update(summary["counts"]["completed"], summary["counts"]["requests"], final=True)
        summary.update(config=asdict(cfg), trace=str(trace_path), skip=args.skip, limit=args.limit,
                       timeseries=series.write(args.output, sim.now))
        if overlay:
            summary["overlay"] = overlay.summary
        result = args.output / "summary.json"
        result.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts = summary["counts"]
        print(f"Completed {counts['completed']:,}/{counts['requests']:,} requests")
        print(f"HHM hit rate: {summary['hit_rates']['hhm']:.2%}")
        print(f"WIN / ASU bytes written: {counts['win_write_bytes']:,} / {counts['asu_write_bytes']:,}")
        print(f"Summary: {result.resolve()}")
        print(f"Chart: {(args.output / 'timeseries.svg').resolve()}")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Simulation failed: {error}\n")


if __name__ == "__main__":
    main()
