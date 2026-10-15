import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExampleImportPathTests(unittest.TestCase):
    def setUp(self):
        self.example_path = Path(__file__).resolve().parent / "D1" / "d1_1_base.py"

    def test_example_runs_from_repository_root(self):
        self._assert_example_runs(Path(__file__).resolve().parents[1])

    def test_example_runs_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._assert_example_runs(Path(temp_dir))

    def _assert_example_runs(self, cwd):
        with tempfile.TemporaryDirectory() as temp_dir:
            stub_dir = Path(temp_dir)
            langchain_dir = stub_dir / "langchain"
            langchain_dir.mkdir()
            (langchain_dir / "__init__.py").write_text("", encoding="utf-8")
            (langchain_dir / "chat_models.py").write_text(
                """
class FakeResponse:
    content = \"stub answer\"


class FakeModel:
    def invoke(self, messages):
        return FakeResponse()


def init_chat_model(*args, **kwargs):
    return FakeModel()
""".lstrip(),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(stub_dir)
            # 该用例只验证导入路径；使用离线假 Key 通过示例的 fail-fast，
            # 模型已由上面的本地 stub 替代，不会访问真实 API。
            environment["OPENAI_API_KEY"] = "sk-test"
            result = subprocess.run(
                [sys.executable, str(self.example_path)],
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("stub answer", result.stdout)
        self.assertNotIn("No module named 'config'", result.stderr)


if __name__ == "__main__":
    unittest.main()
