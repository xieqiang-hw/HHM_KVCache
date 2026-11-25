"""Windowed metrics and one six-panel SVG report. Standard library only."""

from collections import defaultdict
import csv
from dataclasses import dataclass, field
from html import escape
import math


@dataclass
class Window:
    requests: int = 0
    input_tokens: int = 0
    hhm_tokens: int = 0
    asu_tokens: int = 0
    ideal_tokens: int = 0
    decode_sum: int = 0
    decode_steps: int = 0
    queued_area: float = 0.0
    queued_peak: int = 0
    latencies: dict = field(default_factory=lambda: {"ttft": [], "tpot": [], "te2e": []})

    def merge(self, other):
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.hhm_tokens += other.hhm_tokens
        self.asu_tokens += other.asu_tokens
        self.ideal_tokens += other.ideal_tokens
        self.decode_sum += other.decode_sum
        self.decode_steps += other.decode_steps
        self.queued_area += other.queued_area
        self.queued_peak = max(self.queued_peak, other.queued_peak)
        for name, values in other.latencies.items():
            self.latencies[name].extend(values)


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


class TimeSeries:
    def __init__(self, interval_s=60):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("time-series interval must be positive")
        self.interval = interval_s
        self.windows = defaultdict(Window)
        self.queued = 0
        self.queued_since = 0.0

    def at(self, time_s):
        return self.windows[math.floor(time_s / self.interval)]

    def arrival(self, time_s, input_tokens, ideal_tokens):
        window = self.at(time_s)
        window.requests += 1
        window.input_tokens += input_tokens
        window.ideal_tokens += ideal_tokens
        self.change_queue(time_s, input_tokens)

    def admission(self, time_s, arrival_s, input_tokens, hhm_tokens, asu_tokens):
        window = self.at(arrival_s)
        window.hhm_tokens += hhm_tokens
        window.asu_tokens += asu_tokens
        self.change_queue(time_s, -input_tokens)

    def decode(self, time_s, batch_size):
        if batch_size:
            window = self.at(time_s)
            window.decode_sum += batch_size
            window.decode_steps += 1

    def finished(self, arrival_s, ttft, tpot, te2e):
        latencies = self.at(arrival_s).latencies
        for name, value in (("ttft", ttft), ("tpot", tpot), ("te2e", te2e)):
            if value is not None:
                latencies[name].append(value)

    def change_queue(self, now, tokens):
        # Integrate the old queue depth over the time it was actually held.
        previous = self.queued_since
        while self.queued and previous < now:
            index = math.floor(previous / self.interval)
            boundary = (index + 1) * self.interval
            if boundary <= previous:  # float rounding exactly on a boundary
                index += 1
                boundary = (index + 1) * self.interval
            end = min(now, boundary)
            window = self.windows[index]
            window.queued_area += self.queued * (end - previous)
            window.queued_peak = max(window.queued_peak, self.queued)
            previous = end
        self.queued_since = now
        self.queued += tokens
        if self.queued < 0:
            raise ValueError("prefill queue token accounting is negative")

    def rows(self, end_time, max_windows=240):
        self.change_queue(end_time, 0)
        base_count = max(1, math.ceil(end_time / self.interval), 1 + max(self.windows, default=0))
        stride = max(1, math.ceil(base_count / max_windows))
        width = stride * self.interval
        merged = defaultdict(Window)
        for index, window in self.windows.items():
            merged[index // stride].merge(window)
        records = []
        for index in range(math.ceil(base_count / stride)):
            window = merged[index]
            start, end = index * width, min(end_time, (index + 1) * width)
            record = {
                "start_s": start, "end_s": end, "requests": window.requests,
                "input_tokens": window.input_tokens, "hhm_hit_tokens": window.hhm_tokens,
                "asu_hit_tokens": window.asu_tokens, "ideal_hit_tokens": window.ideal_tokens,
                "hhm_hit_rate": divide(window.hhm_tokens, window.input_tokens),
                "asu_hit_rate_on_hhm_miss": divide(window.asu_tokens, window.input_tokens - window.hhm_tokens),
                "ideal_hit_rate": divide(window.ideal_tokens, window.input_tokens),
                "decode_rank_steps": window.decode_steps,
                "decode_bs_mean": divide(window.decode_sum, window.decode_steps),
                "queued_prefill_tokens_mean": divide(window.queued_area, end - start),
                "queued_prefill_tokens_peak": window.queued_peak,
            }
            for name, values in window.latencies.items():
                record[name + "_samples"] = len(values)
                record[name + "_mean_ms"] = 1000 * math.fsum(values) / len(values) if values else None
                record[name + "_p90_ms"] = (
                    1000 * sorted(values)[math.ceil(0.9 * len(values)) - 1] if values else None
                )
            records.append(record)
        return records, width

    def write(self, directory, end_time):
        rows, width = self.rows(end_time)
        with (directory / "timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (directory / "timeseries.svg").write_text(render_chart(rows, end_time, width), encoding="utf-8")
        return {"interval_seconds": width, "file": "timeseries.csv", "chart": "timeseries.svg"}


def render_chart(rows, end_time, interval):
    """Lay out six readable plots, including empty windows as line breaks."""
    duration = max(1.0, end_time)
    tick_scale, time_unit = ((86400, "days") if duration >= 172800 else
                            (3600, "h") if duration >= 7200 else
                            (60, "min") if duration >= 120 else (1, "s"))
    blue, teal, purple, orange = "#175da2", "#087f72", "#7755a4", "#b2542a"
    panels = [
        ("Prefix cache hit rate", "%",
         [("hhm_hit_rate", "HHM", blue, ""), ("asu_hit_rate_on_hhm_miss", "ASU | HHM miss", teal, ""),
          ("ideal_hit_rate", "Ideal", purple, "6 3")], 100, False),
        ("Decode batch size", "seqs / rank",
         [("decode_bs_mean", "Mean on active decode steps", blue, "")], 1, False),
        ("Cluster queued prefill", "tokens",
         [("queued_prefill_tokens_mean", "Time-weighted mean", teal, ""),
          ("queued_prefill_tokens_peak", "Peak", orange, "6 3")], 1, False),
    ]
    for name, title in (("ttft", "TTFT"), ("tpot", "TPOT"), ("te2e", "TE2E")):
        panels.append((title, "ms",
                       [(f"{name}_mean_ms", "Mean", blue, ""), (f"{name}_p90_ms", "P90", orange, "6 3")],
                       1, True))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="820" viewBox="0 0 1440 820"'
           ' role="img" aria-label="HHM cache simulation time series, six panels">',
           '<rect width="1440" height="820" fill="#f4f7fb"/>',
           '<g font-family="Segoe UI, Arial, sans-serif" fill="#152a44">',
           '<text x="35" y="46" font-size="30" font-weight="700">HHM cache simulation</text>',
           f'<text x="35" y="74" font-size="16" fill="#526984">Elapsed simulation time · '
           f'{interval / tick_scale:.3g} {time_unit} per window</text>']
    for index, panel in enumerate(panels):
        title, unit, series, scale, latency_units = panel
        values = [row[key] for key, _, _, _ in series for row in rows if row[key] is not None]
        if latency_units and max(values, default=0) >= 5000:
            unit, scale = "s", 0.001
        top = 100 if unit == "%" else nice_top(max(values, default=0) * scale)
        left, y = 35 + index % 3 * 462, 94 + index // 3 * 345
        x0, x1, y0, y1 = left + 65, left + 420, y + 106, y + 275
        svg.extend([
            f'<rect x="{left}" y="{y}" width="444" height="322" rx="14" fill="#fff" stroke="#d7e1ee"/>',
            f'<text x="{left+21}" y="{y+32}" font-size="21" font-weight="700">{escape(title)}</text>',
            f'<text x="{left+21}" y="{y+60}" font-size="13" fill="#526984">{escape(unit)}</text>',
        ])
        legend_x = x0
        for _, label, color, dash in series:
            dashes = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(f'<path d="M{legend_x} {y+83}h20" stroke="{color}" stroke-width="2.6"{dashes}/>'
                       f'<text x="{legend_x+25}" y="{y+87}" font-size="12">{escape(label)}</text>')
            legend_x += 44 + 7 * len(label)
        for mark in range(5):
            yy = y1 - mark * (y1 - y0) / 4
            xx = x0 + mark * (x1 - x0) / 4
            svg.append(f'<path d="M{x0} {yy:.2f}H{x1} M{xx:.2f} {y0}V{y1}"'
                       ' fill="none" stroke="#e2e9f2"/>')
            svg.append(f'<text x="{x0-8}" y="{yy+4:.2f}" text-anchor="end" font-size="12"'
                       f' fill="#526984">{axis_value(top * mark / 4)}</text>')
            svg.append(f'<text x="{xx:.2f}" y="{y1+20}" text-anchor="middle" font-size="12"'
                       f' fill="#526984">{axis_value(duration / tick_scale * mark / 4)}</text>')
        if not values:
            svg.append(f'<text x="{(x0+x1)/2}" y="{(y0+y1)/2}" text-anchor="middle" font-size="14"'
                       ' fill="#526984">No samples</text>')
        for key, _, color, dash in series:
            svg.extend(draw_series(rows, key, color, dash, scale, top, duration, (x0, x1, y0, y1)))
    svg.append("</g></svg>")
    return "\n".join(svg) + "\n"


def nice_top(value):
    if value <= 0:
        return 1
    base = 10 ** math.floor(math.log10(value))
    return next(mark * base for mark in (1, 2, 4, 5, 8, 10) if value <= mark * base)


def axis_value(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1000:
        return f"{value / 1000:.3g}k"
    return f"{value:.3g}"


def draw_series(rows, key, color, dash, scale, top, duration, bounds):
    x0, x1, y0, y1 = bounds
    path, dots = [], []
    segment = []
    for row in [*rows, {key: None}]:
        value = row[key]
        if value is not None:
            at = (row["start_s"] + row["end_s"]) / 2
            x = x0 + at / duration * (x1 - x0)
            y = y1 - value * scale / top * (y1 - y0)
            segment.append((x, y))
        elif segment:
            if len(segment) == 1:
                x, y = segment[0]
                dots.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.6" fill="{color}"/>')
            else:
                path.append("M" + "L".join(f"{x:.2f} {y:.2f}" for x, y in segment))
            segment = []
    if path:
        dashes = f' stroke-dasharray="{dash}"' if dash else ""
        dots.insert(0, f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="2.2"'
                       f' stroke-linecap="round" stroke-linejoin="round"{dashes}/>')
    return dots
