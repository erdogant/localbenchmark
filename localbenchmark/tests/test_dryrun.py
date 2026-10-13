"""Tests for --dry-run: GGUF reader, file sizes, KV cache, RAM offload, context, folders.
Run with: python -m unittest test_dryrun -v"""
import contextlib
import io
import math
import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import openai  # noqa: F401
except ImportError:
    _stub = types.ModuleType("openai")
    _stub.OpenAI = object
    sys.modules["openai"] = _stub
import localbenchmark as lb  # noqa: E402


def gguf_bytes(kv):
    """Minimal GGUF v3 header. kv: list of (key, type, value); type 4=uint32, 8=string, 9=array."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(kv))
    for key, vtype, value in kv:
        out += s(key) + struct.pack("<I", vtype)
        if vtype == 4:
            out += struct.pack("<I", value)
        elif vtype == 8:
            out += s(value)
        elif vtype == 9:                       # (element type, values)
            etype, items = value
            out += struct.pack("<I", etype) + struct.pack("<Q", len(items))
            for item in items:
                out += s(item) if etype == 8 else struct.pack("<I", item)
    return out


LLAMA_LIKE = [
    ("general.architecture", 8, "llama"),
    ("llama.block_count", 4, 32),
    ("llama.attention.head_count", 4, 32),
    ("llama.attention.head_count_kv", 4, 8),
    ("llama.embedding_length", 4, 4096),
    ("tokenizer.ggml.tokens", 9, (8, [f"tok{i}" for i in range(100)])),   # long array: skipped
]


class GgufReader(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def write(self, name, kv):
        p = self.dir / name
        p.write_bytes(gguf_bytes(kv))
        return p

    def test_kv_bytes_from_architecture(self):
        p = self.write("m.gguf", LLAMA_LIKE)
        profile = lb.kv_profile(lb.read_gguf_metadata(p))
        # 2 bytes x 32 layers x 8 KV heads x (128 key + 128 value)
        self.assertEqual(profile["kv_bytes_per_token"], 2 * 32 * 8 * 256)
        self.assertEqual(profile["kv_bytes_per_token"], 131072)

    def test_per_layer_kv_heads(self):
        kv = [("general.architecture", 8, "x"), ("x.block_count", 4, 3),
              ("x.attention.head_count", 4, 4), ("x.attention.head_count_kv", 9, (4, [2, 2, 1])),
              ("x.embedding_length", 4, 256)]
        profile = lb.kv_profile(lb.read_gguf_metadata(self.write("p.gguf", kv)))
        self.assertEqual(profile["kv_bytes_per_token"], 2 * 5 * (64 + 64))

    def test_not_a_gguf_file_is_rejected(self):
        p = self.dir / "bad.gguf"
        p.write_bytes(b"nope")
        with self.assertRaises(ValueError):
            lb.read_gguf_metadata(p)
        self.assertIsNone(lb.gguf_kv(p))


class HybridAndTags(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_hybrid_model_only_attention_layers_hold_kv(self):
        # 32 layers, 8 KV heads, 128-dim heads, attention in every fourth layer.
        # This reproduces the 33 KB per token measured for Ornith 9B (2 x 8 x 8 x 256 bytes).
        kv = LLAMA_LIKE[:5] + [("llama.full_attention_interval", 4, 4)]
        path = self.dir / "h.gguf"
        path.write_bytes(gguf_bytes(kv))
        profile = lb.kv_profile(lb.read_gguf_metadata(path))
        self.assertEqual(profile["attention_layers"], 8)
        self.assertEqual(profile["kv_bytes_per_token"], 32768)

    def test_quant_tag_in_folder_name_is_matched(self):
        folder = self.dir / "models" / "pub" / "ornith-1.0-9b" / "Q4_K_M"
        folder.mkdir(parents=True)
        (folder / "model-00001.gguf").write_bytes(gguf_bytes(LLAMA_LIKE))
        index = lb.scan_gguf_index([self.dir / "models"])
        self.assertIsNotNone(lb.find_gguf(index, "ornith-1.0-9b", "Q4_K_M"))
        self.assertIsNone(lb.find_gguf(index, "ornith-1.0-9b", "Q8_0"))

    def test_overhead_setting_changes_total(self):
        index = []
        base = lb.dryrun_rows(["ornith-1.0-9b"], [50000], 16.0, 600, 40, ["Q4_K_M"], index)
        more = lb.dryrun_rows(["ornith-1.0-9b"], [50000], 16.0, 600, 40, ["Q4_K_M"], index,
                              overhead_gb=2.0)
        self.assertAlmostEqual(more.iloc[0]["total_gb"] - base.iloc[0]["total_gb"], 1.5, places=6)


class Estimates(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        # a file the size of the real Ornith Q4_K_M (6.55 GB), with its architecture in the header
        name = "ornith-1.0-9b-q4_k_m.gguf"
        (self.dir / name).write_bytes(gguf_bytes(LLAMA_LIKE))
        with open(self.dir / name, "ab") as f:
            f.truncate(6_550_000_000)
        self.index = lb.scan_gguf_index([self.dir])

    def test_weights_come_from_the_file(self):
        df = lb.dryrun_rows(["ornith-1.0-9b"], [50000], 16.0, 600, 40, ["Q4_K_M"], self.index)
        row = df.iloc[0]
        self.assertEqual(row["weights_source"], "GGUF file")
        self.assertAlmostEqual(row["weights_gb"], 6.55, places=2)
        self.assertEqual(row["kv_source"], "GGUF file")

    def test_kv_uses_architecture_not_the_default(self):
        df = lb.dryrun_rows(["ornith-1.0-9b"], [50000], 16.0, 600, 40, ["Q4_K_M"], self.index)
        self.assertAlmostEqual(df.iloc[0]["kv_gb"], 131072 * 50000 / 1e9, places=3)

    def test_quant_without_file_falls_back_to_estimate(self):
        df = lb.dryrun_rows(["ornith-1.0-9b"], [50000], 16.0, 600, 40, ["Q8_0"], self.index)
        self.assertEqual(df.iloc[0]["weights_source"], "estimate")

    def test_unknown_size_is_nan_and_not_ranked(self):
        df = lb.dryrun_rows(["mystery-model-x", "ornith-1.0-9b"], [50000], 16.0, 600, 40,
                            ["Q4_K_M"], self.index)
        muse = df[df["label"] == "mystery-model-x"].iloc[0]
        self.assertTrue(math.isnan(muse["total_gb"]))
        self.assertTrue(math.isnan(muse["decode_ceiling_tps"]))
        board = lb.dryrun_leaderboard(df)
        self.assertTrue(board[board["label"] == "mystery-model-x"]["rank"].isna().all())
        self.assertEqual(board.iloc[0]["label"], "ornith-1.0-9b")

    def test_oversized_model_blends_in_ram_bandwidth(self):
        # 30B at Q8_0 is far above 16 GB: part of it is read from system RAM
        df = lb.dryrun_rows(["qwen3-coder-30b-a3b-instruct"], [50000], 16.0, 600, 40,
                            ["Q8_0"], [])
        row = df.iloc[0]
        self.assertTrue(row["exceeds_gpu"])
        self.assertEqual(row["bandwidth_used"], "GPU+RAM" if row["offload_share"] < 1 else "RAM")
        all_gpu = 1 / (row["read_gb_per_token"] / 600)
        self.assertLess(row["decode_ceiling_tps"], all_gpu)

    def test_model_within_vram_uses_gpu_bandwidth_only(self):
        df = lb.dryrun_rows(["ornith-1.0-9b"], [50000], 16.0, 600, 40, ["Q4_K_M"], self.index)
        row = df.iloc[0]
        self.assertFalse(row["exceeds_gpu"])
        self.assertEqual(row["bandwidth_used"], "GPU")
        self.assertAlmostEqual(row["decode_ceiling_tps"], 600 / row["read_gb_per_token"], places=3)

    def test_context_lowers_the_speed_ceiling(self):
        df = lb.dryrun_rows(["ornith-1.0-9b"], [50000, 120000, 180000], 16.0, 600, 40,
                            ["Q4_K_M"], self.index)
        speeds = list(df.sort_values("context")["decode_ceiling_tps"])
        self.assertEqual(speeds, sorted(speeds, reverse=True))
        self.assertGreater(speeds[0], speeds[-1])

    def test_every_context_is_ranked(self):
        df = lb.dryrun_rows(["ornith-1.0-9b"], [50000, 120000], 16.0, 600, 40,
                            ["Q4_K_M", "Q8_0"], self.index)
        board = lb.dryrun_leaderboard(df)
        self.assertEqual(len(board), 4)
        self.assertEqual(set(board["context"]), {50000, 120000})


class Lookup(unittest.TestCase):
    def test_gpt_oss_uses_active_3_6b(self):
        total, active, source = lb.model_params("openai/gpt-oss-20b")
        self.assertEqual((total, active, source), (21.0, 3.6, "lookup"))

    def test_phi4_is_dense_14b_with_16k_limit(self):
        self.assertEqual(lb.model_params("microsoft/phi-4")[:2], (14.0, 14.0))
        self.assertEqual(lb.lookup_model("phi-4")["max_context"], 16_000)

    def test_muse_glimmer_and_devstral(self):
        self.assertEqual(lb.model_params("muse-glimmer")[:2], (30.0, 30.0))
        self.assertEqual(lb.model_params("mistralai/devstral-small-2-2512")[:2], (24.0, 24.0))

    def test_exact_match_does_not_catch_phi4_mini(self):
        self.assertIsNone(lb.lookup_model("phi-4-mini"))

    def test_contexts_above_the_limit_are_skipped(self):
        df = lb.dryrun_rows(["phi-4"], [8000, 50000], 16.0, 600, 40, ["Q4_K_M"], [])
        self.assertEqual(list(df["context"]), [8000])

    def test_model_with_no_context_left_does_not_crash_the_report(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            df = lb.dryrun_rows(["phi-4"], [50000], 16.0, 600, 40, ["Q4_K_M"], [])
            lb.dryrun_console(df, ["phi-4"], [50000], ["Q4_K_M"], [])
        self.assertIn("no context to compute", out.getvalue())


class Run(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)
        self.saved = {n: getattr(lb, n) for n in ["check_lm_studio", "OpenAI", "LOCAL_MODEL_DIRS"]}
        def boom(*a, **k):
            raise AssertionError("network call during dry run")
        lb.check_lm_studio = boom
        lb.OpenAI = boom
        lb.LOCAL_MODEL_DIRS = [Path(self.tmp) / "no-models"]

    def tearDown(self):
        for n, f in self.saved.items():
            setattr(lb, n, f)
        os.chdir(self.cwd)

    def run_main(self, argv):
        sys.argv = ["localbenchmark"] + argv
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lb.main()
        return out.getvalue()

    def test_writes_dryrun_folder_and_plots(self):
        self.run_main(["--dryrun", "--models", "ornith-1.0-9b", "--contexts", "50000", "--vram-gb", "16"])
        self.assertTrue((Path(self.tmp) / "dryrun" / "dryrun_estimates.csv").exists())
        self.assertTrue((Path(self.tmp) / "dryrun" / "dryrun_leaderboard.csv").exists())
        for name in ["D1_total_gb_heatmap.png", "D2_memory_vs_context.png", "D3_speed_vs_memory.png",
                     "D4_ranking.png", "D5_speed_vs_context.png"]:
            self.assertTrue((Path(self.tmp) / "dryrun" / "plots" / name).exists(), name)

    def test_default_quants_exclude_q3_and_q5(self):
        self.run_main(["--dry-run", "--models", "ornith-1.0-9b", "--contexts", "50000", "--vram-gb", "16"])
        import pandas as pd
        df = pd.read_csv(Path(self.tmp) / "dryrun" / "dryrun_estimates.csv")
        self.assertEqual(sorted(df["quant"].unique()), sorted(lb.DRYRUN_DEFAULT_QUANTS))

    def test_unknown_model_prints_nan(self):
        out = self.run_main(["--dry-run", "--models", "mystery-model-x", "--contexts", "50000", "--vram-gb", "16"])
        self.assertIn("nan", out)

    def test_model_folder_is_used_when_given(self):
        models = Path(self.tmp) / "models" / "pub" / "ornith-1.0-9b"
        models.mkdir(parents=True)
        (models / "ornith-1.0-9b-Q4_K_M.gguf").write_bytes(gguf_bytes(LLAMA_LIKE))
        self.run_main(["--dry-run", "--models", "ornith-1.0-9b", "--contexts", "50000", "--vram-gb", "16",
                       "--quants", "Q4_K_M", "--model-dir", str(Path(self.tmp) / "models")])
        import pandas as pd
        df = pd.read_csv(Path(self.tmp) / "dryrun" / "dryrun_estimates.csv")
        self.assertEqual(df.iloc[0]["weights_source"], "GGUF file")
        self.assertEqual(df.iloc[0]["kv_source"], "GGUF file")

    def test_humaneval_folder_created_on_stop(self):
        out = self.run_main(["--benchmark", "humaneval"])
        self.assertTrue((Path(self.tmp) / "humaneval").is_dir())
        self.assertIn("STOP", out)

    def test_resume_from_current_folder_first(self):
        Path("coding_benchmark_results.csv").write_text("x\n")
        self.assertEqual(lb.resolve_output(None, "coding_benchmark_results.csv", Path("humaneval")),
                         Path("coding_benchmark_results.csv"))

    def test_resume_from_benchmark_folder_otherwise(self):
        Path("humaneval").mkdir()
        Path("humaneval", "coding_benchmark_results.csv").write_text("x\n")
        self.assertEqual(lb.resolve_output(None, "coding_benchmark_results.csv", Path("humaneval")),
                         Path("humaneval") / "coding_benchmark_results.csv")

    def test_new_file_goes_into_benchmark_folder(self):
        self.assertEqual(lb.resolve_output(None, "coding_benchmark_results.csv", Path("mbpp")),
                         Path("mbpp") / "coding_benchmark_results.csv")


if __name__ == "__main__":
    unittest.main()
