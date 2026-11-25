"""Terminal progress for a streaming replay, without pre-scanning the trace."""

import json
import sys
from time import monotonic


def request_count(path, skip, limit):
    directory = path if path.is_dir() else path.parent
    manifest = directory / "manifest.json"
    if not manifest.is_file() or (path.is_file() and path.name != "requests.jsonl"):
        return None
    count = json.loads(manifest.read_text(encoding="utf-8")).get("request_count")
    if type(count) is not int or count < 0:
        return None
    remaining = max(0, count - skip)
    return min(remaining, limit) if limit is not None else remaining


def elapsed(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Progress:
    def __init__(self, total, stream=None, clock=monotonic):
        self.total = total
        self.stream = sys.stderr if stream is None else stream
        self.clock = clock
        self.interactive = self.stream.isatty()
        self.interval = 1.0 if self.interactive else 5.0
        self.started = self.clock()
        self.next_update = self.started
        self.line_width = 0

    def __enter__(self):
        self.update(0, 0)
        return self

    def __exit__(self, *_):
        if self.interactive and self.line_width:
            self.stream.write("\n")
            self.stream.flush()

    def update(self, completed, arrived, *, final=False):
        now = self.clock()
        if not final and now < self.next_update:
            return
        self.next_update = now + self.interval
        total = arrived if final else self.total
        if total:
            fraction = min(1.0, completed / total)
            filled = int(24 * fraction)
            bar = "#" * filled + "-" * (24 - filled)
            status = f"[{bar}] {fraction:6.2%}  {completed:,}/{total:,} completed"
        else:
            status = f"{completed:,} completed"
        line = f"Replay {status}  |  {arrived:,} arrived  |  elapsed {elapsed(now - self.started)}"
        if self.interactive:
            self.stream.write("\r" + line.ljust(self.line_width))
            self.line_width = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
