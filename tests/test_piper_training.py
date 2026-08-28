import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from piper_training import (
    PiperTrainPlan,
    export_command,
    latest_checkpoint,
    load_prompts,
    normalize_espeak_voice,
    safe_voice_name,
    training_command,
    write_metadata,
)


class PiperTrainingTests(unittest.TestCase):
    def test_safe_voice_name(self):
        self.assertEqual(safe_voice_name("en GB My Voice medium"), "en-GB-My-Voice-medium")
        with self.assertRaises(ValueError):
            safe_voice_name("!!!")

    def test_espeak_voice_aliases(self):
        self.assertEqual(normalize_espeak_voice("en-gb"), "en-GB-x-rp")
        self.assertEqual(normalize_espeak_voice("EN_GB"), "en-GB-x-rp")
        self.assertEqual(normalize_espeak_voice("en-us"), "en-US")
        self.assertEqual(normalize_espeak_voice("de"), "de")

    def test_prompt_loader_deduplicates_and_limits(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "prompts.txt"
            path.write_text("# comment\nHello world.\nHello world.\nSecond line.\n", encoding="utf-8")
            self.assertEqual(load_prompts(path, 1), ["Hello world."])

    def test_training_and_export_commands(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan = PiperTrainPlan(
                voice_name="en_GB-test-medium",
                trainer_python=root / "python.exe",
                trainer_source=root / "piper1-gpl",
                dataset_dir=root / "dataset",
                output_dir=root / "out",
                checkpoint=root / "base.ckpt",
            )
            command = training_command(plan)
            self.assertEqual(command[1:4], ["-m", "piper.train", "fit"])
            self.assertIn("--data.csv_path", command)
            self.assertIn("--ckpt_path", command)
            voice_index = command.index("--data.espeak_voice") + 1
            self.assertEqual(command[voice_index], "en-GB-x-rp")
            export = export_command(plan, root / "last.ckpt")
            self.assertEqual(export[1:3], ["-m", "piper.train.export_onnx"])
            self.assertIn(str(plan.onnx_path), export)

    def test_latest_checkpoint_prefers_newest(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            older = root / "a.ckpt"
            newer = root / "nested" / "b.ckpt"
            newer.parent.mkdir()
            older.write_text("a")
            newer.write_text("b")
            older.touch()
            import os, time
            old_time = time.time() - 20
            os.utime(older, (old_time, old_time))
            self.assertEqual(latest_checkpoint(root), newer)

    def test_metadata_writer_uses_pipe_format(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "metadata.csv"
            write_metadata(path, [("utt-00001.wav", "Hello there.")])
            self.assertEqual(path.read_text(encoding="utf-8"), "utt-00001.wav|Hello there.\n")


if __name__ == "__main__":
    unittest.main()
