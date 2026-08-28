import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import Settings, piper_command, rvc_command, validate


class CommandTests(unittest.TestCase):
    def test_piper_command_uses_explicit_config(self):
        s = Settings(piper_exe="piper.exe", piper_model="voice.onnx", piper_config="voice.json")
        self.assertEqual(piper_command(s, Path("out.wav")), ["piper.exe", "--model", "voice.onnx", "--config", "voice.json", "--output_file", "out.wav"])

    def test_rvc_optional_index(self):
        s = Settings(rvc_python="python.exe", rvc_root="rvc", rvc_model="voice.pth", rvc_index="voice.index")
        command = rvc_command(s, Path("in.wav"), Path("out.wav"))
        self.assertIn("--index", command)
        self.assertIn("voice.index", command)
        self.assertIn("--overwrite", command)
        self.assertEqual(command[1:3], ["-m", "infer.cli"])

    def test_rvc_pitch_is_forwarded(self):
        s = Settings(rvc_python="python.exe", rvc_root="rvc", rvc_model="voice.pth", pitch=-7)
        command = rvc_command(s, Path("in.wav"), Path("out.wav"))
        self.assertEqual(command[command.index("--pitch") + 1], "-7")

    def test_validation_accepts_a_complete_layout(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            files = [root / "piper.exe", root / "voice.onnx", root / "python.exe", root / "voice.pth", root / "infer" / "cli.py"]
            for item in files:
                item.parent.mkdir(parents=True, exist_ok=True)
                item.touch()
            s = Settings(piper_exe=str(files[0]), piper_model=str(files[1]), rvc_python=str(files[2]), rvc_model=str(files[3]), rvc_root=str(root))
            self.assertEqual(validate(s), [])


if __name__ == "__main__":
    unittest.main()
