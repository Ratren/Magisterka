import unittest
import run_cache as rc


class TestParse(unittest.TestCase):
    def test_parses_values_and_unsupported(self):
        text = (
            "1000,,instructions:u,2000,100.00,,\n"
            "500,,cycles:u,2000,100.00,,\n"
            "<not supported>,,ls_any_fills_from_sys.dram_io_near:u,0,0.00,,\n"
            "0.001234,,seconds time elapsed\n"
        )
        counts = rc.parse_perf_stat(text)
        self.assertEqual(counts["instructions:u"], 1000.0)
        self.assertEqual(counts["cycles:u"], 500.0)
        self.assertIsNone(counts["ls_any_fills_from_sys.dram_io_near:u"])

    def test_get_count_modifier_variants(self):
        counts = {"instructions:u": 42.0}
        self.assertEqual(rc.get_count(counts, "instructions"), 42.0)
        self.assertEqual(rc.get_count(counts, "instructions:u"), 42.0)
        self.assertIsNone(rc.get_count(counts, "cycles"))

    def test_not_counted_maps_to_none(self):
        counts = rc.parse_perf_stat("<not counted>,,L1-dcache-loads:u,0,0.00,,\n")
        self.assertIsNone(counts["L1-dcache-loads:u"])

    def test_short_line_skipped_without_error(self):
        counts = rc.parse_perf_stat("foo,bar\n1000,,instructions:u,2000,100.00,,\n")
        self.assertEqual(counts["instructions:u"], 1000.0)

    def test_get_count_demodifies_requested_event(self):
        self.assertEqual(rc.get_count({"instructions": 5.0}, "instructions:k"), 5.0)


class TestMath(unittest.TestCase):
    def test_median(self):
        self.assertEqual(rc.median([3, 1, 2]), 2)
        self.assertEqual(rc.median([4, 1, 2, 3]), 2.5)
        self.assertIsNone(rc.median([None, None]))

    def test_per_iteration_subtracts_and_divides(self):
        self.assertEqual(rc.per_iteration(3000.0, 1000.0, 100), 20.0)

    def test_per_iteration_clamps_negative(self):
        self.assertEqual(rc.per_iteration(900.0, 1000.0, 100), 0.0)

    def test_per_iteration_none(self):
        self.assertIsNone(rc.per_iteration(None, 1000.0, 100))

    def test_per_iteration_zero_n(self):
        self.assertIsNone(rc.per_iteration(2000.0, 1000.0, 0))

    def test_median_empty(self):
        self.assertIsNone(rc.median([]))


class TestMetrics(unittest.TestCase):
    def test_compute_metrics_full(self):
        per_iter = {
            "l1_loads": 1000.0, "l1_load_misses": 50.0,
            "fill_l2": 30.0, "fill_l3": 15.0, "fill_dram": 5.0,
            "instructions": 2000.0, "cycles": 1000.0,
        }
        m = rc.compute_metrics(per_iter)
        self.assertAlmostEqual(m["l1_miss_rate"], 0.05)
        self.assertAlmostEqual(m["pct_l2"], 60.0)
        self.assertAlmostEqual(m["pct_l3"], 30.0)
        self.assertAlmostEqual(m["pct_dram"], 10.0)
        self.assertAlmostEqual(m["fills_per_iter"], 50.0)
        self.assertAlmostEqual(m["ipc"], 2.0)

    def test_compute_metrics_missing_events(self):
        m = rc.compute_metrics({"l1_loads": None, "fill_l2": None})
        self.assertIsNone(m["l1_miss_rate"])
        self.assertIsNone(m["pct_l2"])
        self.assertIsNone(m["ipc"])

    def test_zero_loads_gives_none_miss_rate(self):
        m = rc.compute_metrics({"l1_loads": 0.0, "l1_load_misses": 0.0})
        self.assertIsNone(m["l1_miss_rate"])

    def test_zero_instructions_gives_zero_ipc(self):
        m = rc.compute_metrics({"instructions": 0.0, "cycles": 100.0})
        self.assertEqual(m["ipc"], 0.0)

    def test_all_fills_none(self):
        m = rc.compute_metrics({"fill_l2": None, "fill_l3": None, "fill_dram": None})
        self.assertIsNone(m["pct_l2"])
        self.assertIsNone(m["fills_per_iter"])


if __name__ == "__main__":
    unittest.main()
