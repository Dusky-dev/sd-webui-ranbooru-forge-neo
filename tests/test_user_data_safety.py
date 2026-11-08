"""Prevent Git from tracking Ranbooru's local credentials and configuration."""
from pathlib import Path
import subprocess
import unittest


class LocalUserDataSafetyTests(unittest.TestCase):
    def test_git_does_not_track_user_data(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / ".git").exists():
            self.skipTest("Git metadata not available in this checkout")
        result = subprocess.run(
            ["git", "ls-files", "--", "user/"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            result.stdout.strip(), "",
            f"Personal user files must remain untracked: {result.stdout}",
        )

    def test_local_user_directory_is_ignored(self):
        root = Path(__file__).resolve().parents[1]
        self.assertIn("/user/", (root / ".gitignore").read_text())


if __name__ == "__main__":
    unittest.main()
