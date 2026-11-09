"""Allocation-free policy/CLI checks; no native/framework dependency required."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from klein_sliders.cli import main
from klein_sliders.contracts import WEIGHT_POLICY, validate_options, validate_weight_policy


class RawWeightPolicy(unittest.TestCase):
    def test_only_none_or_omitted_decay_is_allowed(self):
        for options in ({}, {"ema_decay": None}):
            validate_weight_policy(options)
        for value in (0., .9, .995, False, True, "none", [], {}, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "ema_decay=None"):
                validate_weight_policy({"ema_decay": value})

    def test_options_refuse_ema_before_filesystem_or_model_requirements(self):
        with self.assertRaisesRegex(ValueError, "RAW weights"):
            validate_options({"ema_decay": .995, "model_dir": "absent", "cache_dir": "absent"})

    def test_plan_and_train_refuse_historical_ema_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "historical.json"
            config.write_text(json.dumps(dict(backend="flux2_klein", cohort="historical-EMA",
                options=dict(ema_decay=.995, model_dir="absent", cache_dir="absent"))))
            with patch.dict(sys.modules, {"particle_sliders.lifecycle": None}):
                for command in (["plan", "--config", str(config)],
                                ["train", "--config", str(config), "--output", str(root / "run")]):
                    with self.subTest(command=command[0]), self.assertRaisesRegex(ValueError, "RAW weights"):
                        main(command)
            self.assertFalse((root / "run").exists())

    def test_raw_plan_declares_policy_without_allocating(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "raw.json"
            config.write_text(json.dumps(dict(backend="flux2_klein", cohort="fresh-RAW",
                options=dict(ema_decay=None, model_dir=str(root / "model"), cache_dir=str(root / "cache")))))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["plan", "--config", str(config)])
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["weight_policy"], WEIGHT_POLICY)
            self.assertEqual(plan["selection"], "raw")
            self.assertFalse(plan["staged_model"])
            self.assertFalse(plan["frozen_cache"])

    def test_cli_passes_raw_selection_to_export(self):
        # Explicit CLI stub: this does not train, construct a game or export a native artifact.
        exports = []
        class Config:
            @classmethod
            def from_dict(cls, value):
                item = cls(); item.value = value; return item
            def as_dict(self):
                return self.value
        class Session:
            def __enter__(self): return self
            def __exit__(self, *unused): pass
            def train(self, **unused): pass
            def checkpoint(self, path): pass
            def export_inference(self, path, *, selection): exports.append(selection)
        shared = ModuleType("particle_sliders.lifecycle")
        shared.SessionConfig = Config
        shared.read_packet = lambda path: None
        product = ModuleType("klein_sliders.lifecycle")
        product.BACKEND = SimpleNamespace(open=lambda config: Session())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = root / "raw.json"
            config.write_text(json.dumps(dict(backend="flux2_klein", cohort="fresh-RAW",
                                               options=dict(ema_decay=None))))
            with patch.dict(sys.modules, {"particle_sliders.lifecycle": shared,
                                          "klein_sliders.lifecycle": product}), contextlib.redirect_stdout(io.StringIO()):
                main(["train", "--config", str(config), "--output", str(root / "run")])
        self.assertEqual(exports, ["raw"])


if __name__ == "__main__":
    unittest.main()
