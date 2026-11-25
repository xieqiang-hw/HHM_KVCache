"""Run the estimated-TTFT experiments with four workers."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import subprocess
import sys


OUTPUT = Path(__file__).resolve().parent
SIMULATOR = OUTPUT.parents[1]
PLAN = {
    10_000: (1, 16, 32, 64),
}
WORKERS = 4
TRACE = "../data/claude-code-main-glm52-128k-compact"


def check_result(output, factor, limit):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    overlay = summary.get("overlay")
    if (summary["trace"].replace("\\", "/") != TRACE or summary["skip"] != 0 or summary["limit"] != limit
            or summary["counts"]["requests"] != limit or summary["counts"]["completed"] != limit
            or summary["config"]["routing_policy"] != "estimated_ttft"):
        raise ValueError("result does not match the requested experiment")
    if factor == 1 and overlay is not None:
        raise ValueError("1x replay must follow the original trace order")
    if factor > 1 and (overlay is None or overlay["factor"] != factor or overlay["selected_request_count"] != limit
                       or overlay["alignment"] != "segment" or overlay["selection_order"] != "after-overlay"):
        raise ValueError("result does not match the requested overlay")
    for filename in ("timeseries.csv", "timeseries.svg"):
        if not (output / filename).is_file():
            raise FileNotFoundError(output / filename)


def run_experiment(factor, limit):
    name = f"{factor}x_{limit}"
    output = OUTPUT / name
    if output.is_dir():
        check_result(output, factor, limit)
        logging.info("Reused %s", name)
        return
    output.mkdir()
    command = [
        sys.executable, "-u", "run.py",
        "--trace", TRACE,
        "--overlay", str(factor), "--overlay-align", "segment",
        "--limit", str(limit), "--no-batch-csv", "--output", str(output),
    ]
    logging.info("Started %s", name)
    with (output / "run.log").open("xb") as log:
        subprocess.run(command, cwd=SIMULATOR, stdout=log, stderr=subprocess.STDOUT,
                       creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    check_result(output, factor, limit)
    logging.info("Finished %s", name)


def main():
    logging.basicConfig(filename=OUTPUT / "batch.log", filemode="x", encoding="utf-8",
                        level=logging.INFO, format="%(asctime)s %(message)s")
    jobs = [(factor, limit) for limit, factors in PLAN.items() for factor in factors]
    logging.info("Queued %s experiments, %s concurrent: %s", len(jobs), WORKERS,
                 ", ".join(f"{factor}x_{limit}" for factor, limit in jobs))
    failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_experiment, factor, limit): (factor, limit) for factor, limit in jobs}
        for future in as_completed(futures):
            factor, limit = futures[future]
            try:
                future.result()
            except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
                failed += 1
                logging.exception("Failed %sx_%s", factor, limit)
    logging.info("Batch finished: %s succeeded, %s failed", len(jobs) - failed, failed)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
