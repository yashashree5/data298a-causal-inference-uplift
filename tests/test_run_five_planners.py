from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_five_planners as batch


class FivePlannerBatchTest(unittest.TestCase):
    def test_invalid_experiment_name_is_rejected_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            argv = [
                "batch",
                "--sample", str(Path("configs/scenarios/pittsburgh_65_sample.csv")),
                "--output-root", str(output),
                "--experiment", "../escape",
                "--smoke-experiment", "smoke",
                "--dataset-split", "train_pittsburgh",
            ]
            with patch("sys.argv", argv), self.assertRaises(ValueError):
                batch.main()
            self.assertFalse(output.exists())

    def test_smoke_and_full_experiments_must_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "batch",
                "--sample", "configs/scenarios/pittsburgh_65_sample.csv",
                "--output-root", directory,
                "--experiment", "same",
                "--smoke-experiment", "same",
                "--dataset-split", "train_pittsburgh",
            ]
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "different names"):
                batch.main()


if __name__ == "__main__":
    unittest.main()
