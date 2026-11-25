"""Focused checks for the cache and replay contracts."""

from dataclasses import replace
import gzip
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import json
import unittest

from cache import ASU, LocalCache, Transfer
from collections import Counter
from compact_trace import FORMAT, RECORD, prepare_keys, read_shard
from overlap import INDEX_FILENAME, Overlay, index_sessions
from simulator import Simulation, decode_seconds
from timeseries import TimeSeries
from workload import Config, TraceRequest, prefix_keys


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(Config(), machines=1, ranks_per_machine=1, asu_devices=1,
                           dram_bytes=5, win_bytes=1, asu_device_bytes=8, kv_bytes_per_token=1)
        self.asu = ASU(self.cfg, 1)
        self.cache = LocalCache(self.cfg, self.asu)
        self.io = Transfer(self.cfg, self.asu, Counter())

    def test_frontier_looks_past_gap(self):
        a = self.cache.node(1, None)
        b = self.cache.node(2, a)
        c = self.cache.node(3, b)
        self.cache.dram.add(a, 0)
        self.cache.win.add(b, 0)
        self.cache.dram.add(c, 0)
        self.assertEqual(set(self.cache.dram.frontier), {c})
        self.cache.release(self.cache.dram, c)
        self.assertEqual(set(self.cache.dram.frontier), {a})

    def test_backup_does_not_publish_until_forward_finishes(self):
        a = self.cache.node(1, None)
        b = self.cache.node(2, a)
        self.cache.dram.add(b, 0)
        self.assertFalse(self.io.backup(b, self.cache))
        self.assertFalse(self.asu.pending)
        self.cache.win.add(a, 0)
        self.assertTrue(self.io.backup(b, self.cache))
        self.assertFalse(self.asu.lookup(1))
        self.assertFalse(self.asu.lookup(2))
        self.asu.finish(self.io)
        self.assertEqual(list(self.asu.objects), [1, 2])

    def test_locked_ancestor_survives_lru_and_delete(self):
        self.asu.capacity = 2
        self.asu.objects.update({1: None, 2: None})
        self.assertTrue(self.asu.lookup_lock(1))
        self.asu.delete(1)
        self.assertTrue(self.asu.reserve_writes([3], self.io))
        self.io.writes.append(3)
        self.assertTrue(self.asu.lookup(1))
        self.assertFalse(self.asu.lookup(2))
        self.asu.finish(self.io)
        self.asu.unlock(1)
        self.assertFalse(self.asu.lookup(1))
        self.assertTrue(self.asu.lookup(3))

    def test_dram_hot_migration_triggers_win_cold_recovery(self):
        self.cache.dram.capacity = 1
        cold, hot = self.cache.node(1, None), self.cache.node(2, None)
        hot.score, hot.updated = 3.0, 110
        self.cache.win.add(cold, 0)
        self.cache.dram.add(hot, 0)
        self.assertTrue(self.cache.make_room(1, 110, self.io))
        self.assertIn(hot, self.cache.win.blocks)
        self.assertEqual(self.io.counters["win_evictions"], 1)
        self.assertNotIn(hot, self.cache.dram.blocks)


class ReplayTests(unittest.TestCase):
    def test_overlay_aligns_first_requests_and_preserves_session_tails(self):
        timestamps = (0, 4, 15, 17, 19, 24)
        sessions = ("a", "b", "c", "a", "c", "b")
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            directory = Path(folder)
            shards = []
            for index, values in enumerate((timestamps[:3], timestamps[3:])):
                name = f"{index}.hkv.gz"
                with gzip.open(directory / name, "wb") as shard:
                    for stamp in values:
                        shard.write(RECORD.pack(stamp, 1, 1, 0, 0))
                shards.append({"file": name, "count": len(values),
                               "first_timestamp_ms": values[0], "last_timestamp_ms": values[-1]})
            (directory / "manifest.json").write_text(json.dumps({
                "format": FORMAT, "request_count": len(timestamps), "shards": shards,
            }), encoding="utf-8")
            sidecar = directory / "source_requests.jsonl"
            sidecar.write_text("\n".join(json.dumps({
                "trace_row": row, "timestamp": stamp, "session_id": session,
            }) for row, (stamp, session) in enumerate(zip(timestamps, sessions))), encoding="utf-8")
            index = directory / INDEX_FILENAME
            index_sessions(directory, sidecar, index)
            overlay = Overlay(directory, index, 2, "first-request")
            aligned = [(req.number, round(req.at * 1000)) for req in overlay.requests(2, 1)]
            self.assertEqual(aligned, [(0, 0), (2, 0), (1, 4), (4, 4), (3, 17), (5, 24)])
            self.assertEqual(overlay.summary["tail_requests"], 2)
            legacy = Overlay(directory, index, 2, "segment")
            self.assertEqual([(r.number, round(r.at * 1000)) for r in legacy.requests(2, 1)],
                             [(0, 0), (2, 2), (1, 4), (4, 6), (3, 17), (5, 24)])
            for full in (overlay, legacy):
                requests = list(full.requests(2, 0.7))
                for skip, limit in ((0, 2), (1, 4), (4, 10)):
                    with self.subTest(align=full.summary["alignment"], skip=skip, limit=limit):
                        subset = Overlay(directory, index, 2, full.summary["alignment"], skip, limit)
                        expected = requests[skip:skip + limit]
                        self.assertEqual(list(subset.requests(2, 0.7)), expected)
                        self.assertEqual(subset.summary["selected_request_count"], len(expected))
                        self.assertEqual(subset.summary["segment_ms"], full.summary["segment_ms"])
                        self.assertEqual(subset.summary["replay_span_ms"], full.summary["replay_span_ms"])
                        self.assertEqual(subset.summary["selected_last_arrival_ms"], round(expected[-1].at * 1000 / 0.7))
            with self.assertRaisesRegex(ValueError, "contains no requests"):
                Overlay(directory, index, 2, "segment", skip=len(timestamps))

    def test_win_reads_continue_after_prefill_admission(self):
        cfg = replace(Config(), machines=1, ranks_per_machine=1,
                      mtp_width=2, mtp_probabilities=(0.0, 1.0))
        run = Simulation(cfg, 2)
        keys = prefix_keys([1, 2], [3, 4, 5], 2)
        cache = run.machines[0][0].cache
        cache.win.add(cache.node(keys[0], None), 0)
        result = run.run([TraceRequest(0, 0.0, 2, 3, keys)])
        self.assertEqual(result["counts"]["win_hit_tokens"], 2)
        # One rolling read in prefill and one in the following decode step.
        self.assertEqual(result["counts"]["win_rolling_read_bytes"], 2 * 2 * cfg.kv_bytes_per_token)

    def test_timeseries_uses_arrival_windows_and_time_weighted_queue(self):
        series = TimeSeries(interval_s=5)
        series.arrival(0, 10, 8)
        series.arrival(2, 20, 12)
        series.admission(4, 0, 10, 6, 1)
        series.admission(7, 2, 20, 4, 4)
        series.decode(3, 2)
        series.decode(4, 4)
        series.decode(6, 0)
        series.finished(0, 5, 0.01, 8)
        series.finished(2, 6, 0.02, 8)
        (first, second), width = series.rows(10)
        self.assertEqual(width, 5)
        self.assertEqual(first["hhm_hit_rate"], 10 / 30)
        self.assertEqual(first["asu_hit_rate_on_hhm_miss"], 5 / 20)
        self.assertEqual(first["ideal_hit_rate"], 20 / 30)
        self.assertEqual(first["decode_bs_mean"], 3)
        self.assertEqual(first["queued_prefill_tokens_mean"], 20)
        self.assertEqual(first["queued_prefill_tokens_peak"], 30)
        self.assertEqual(first["ttft_p90_ms"], 6000)
        self.assertEqual(second["queued_prefill_tokens_mean"], 8)
        self.assertIsNone(second["decode_bs_mean"])

    def test_compact_prefix_delta_keeps_output_kv_and_handles_shorter_prompts(self):
        rows = [
            ([1, 2, 3], [4, 5, 6]),
            ([1, 2, 3, 4, 5], [7]),
            ([1, 2], [8]),
        ]
        raw, previous = b"", []
        expected, shared_counts = [], []
        with NamedTemporaryFile(dir=Path(__file__).resolve().parent, suffix=".hkv.gz", delete=False) as handle:
            path = Path(handle.name)
        try:
            with gzip.open(path, "wb") as shard:
                for number, (prompt, output) in enumerate(rows):
                    row = {"input_length": len(prompt), "output_length": len(output),
                           "input_token_ids": prompt, "output_token_ids": output}
                    raw, keys, shared = prepare_keys(row, 8, raw, previous)
                    previous = keys
                    shared_counts.append(shared)
                    shard.write(RECORD.pack(number, len(prompt), len(output), shared, 0))
                    shard.write(b"".join(keys[shared:]))
                    expected.append((number, len(prompt), len(output), prefix_keys(prompt, output, 2), 0))
            self.assertEqual(list(read_shard(path, 2)), expected)
        finally:
            path.unlink()
        self.assertEqual(shared_counts, [0, 2, 1])

    def test_exact_output_prefix_is_reusable_and_steps_complete(self):
        cfg = replace(Config(), machines=1, ranks_per_machine=2, max_num_seqs=2,
                      max_num_batched_tokens=6, mtp_width=2, mtp_probabilities=(0.0, 1.0))
        first = prefix_keys([1, 2], [3, 4, 5], 2)
        second = prefix_keys([1, 2, 3, 4], [6], 2)
        self.assertEqual(first, second)
        run = Simulation(cfg, 2)
        result = run.run([TraceRequest(0, 0.0, 2, 3, first),
                          TraceRequest(1, 1.0, 4, 1, second)])
        self.assertEqual(result["counts"]["completed"], 2)
        self.assertEqual(result["counts"]["output_tokens"], 4)
        self.assertEqual(result["counts"]["dram_hit_tokens"], 4)
        self.assertEqual(result["tpot"]["samples"], 1)
        self.assertAlmostEqual(decode_seconds(64), 0.1723)

    def test_other_machine_loads_published_asu_prefix(self):
        cfg = replace(Config(), machines=2, ranks_per_machine=1, max_num_seqs=2)
        run = Simulation(cfg, 2)
        keys = prefix_keys([1, 2], [], 2)
        # First request explicitly publishes its prefix. The second machine's
        # admission must promote that ASU block, rather than read the first card.
        run.route = lambda trace: self.route_by_id(run, trace)
        result = run.run([TraceRequest(0, 0.0, 2, 1, keys, share_blocks=1),
                          TraceRequest(1, 1.0, 2, 1, keys)])
        self.assertEqual(result["counts"]["asu_hit_tokens"], 2)
        self.assertEqual(result["counts"]["asu_read_bytes"], 100_000)
        self.assertEqual(result["counts"]["asu_write_bytes"], 100_000)

    @staticmethod
    def route_by_id(run, trace):
        run.machines[trace.number][0].enqueue(trace, trace.input_tokens)
        return trace.number

    def test_ttft_routing_trades_cached_compute_for_queue_wait(self):
        cfg = replace(Config(), machines=1, ranks_per_machine=2)
        trace = TraceRequest(0, 0, 8, 1, (1, 2, 3, 4))
        for policy, backlog, expected in (
            ("estimated_ttft", 2, 0),
            ("estimated_ttft", 10, 1),
            ("cache_first", 10, 0),
        ):
            with self.subTest(policy=policy, backlog=backlog):
                sim = Simulation(replace(cfg, routing_policy=policy), 2)
                warm, cold = sim.machines[0]
                warm.cache.dram.add(warm.cache.node(1, None), 0)
                warm.enqueue(TraceRequest(1, 0, backlog, 1, ()), backlog)
                sim.route(trace)
                self.assertEqual(sim.machines[0][expected].queue[-1][0], trace)

    def test_ttft_routing_rotates_ties_without_touching_asu(self):
        cfg = replace(Config(), machines=2, ranks_per_machine=1)
        sim = Simulation(cfg, 2)
        sim.asu.objects.update({1: None, 2: None})
        for number, machine in enumerate((0, 1, 0, 1)):
            trace = TraceRequest(number, 0, 2, 1, (1,))
            self.assertEqual(sim.route(trace), machine)
            rank = sim.machines[machine][0]
            _, estimate = rank.queue.pop()
            rank.queued_prefill -= estimate
        self.assertEqual(list(sim.asu.objects), [1, 2])
        self.assertFalse(sim.asu.locks)


if __name__ == "__main__":
    unittest.main()
