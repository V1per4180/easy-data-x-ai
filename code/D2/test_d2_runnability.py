"""D2 入口、配置加载与资源清理的离线测试。

大部分测试注入假的数据库客户端，只检查参数、返回值和清理顺序。
环境配置测试在临时目录启动子进程，并用本地替身模块阻止真实数据库连接。
这些测试说明程序流程是否符合约定；真实向量效果需要另外运行课程示例观察。
"""

import importlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


D2_MODULES = (
    "D2.d2_1_ingest",
    "D2.d2_2_vector_search",
    "D2.d2_3_hybrid_search",
    "D2.d2_4_compare",
)


class EmptyCollection:
    """统一返回空结果，用于检查程序是否错误地把“无命中”报告成成功。"""
    def count(self):
        return 0

    def query(self, **_kwargs):
        return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

    def hybrid_search(self, **_kwargs):
        return {"documents": [[]], "distances": [[]], "metadatas": [[]]}


class ExistingDatabase:
    """模拟集合已存在的数据库，便于单独验证查询入口。"""
    def __init__(self, collection=None):
        self.collection = collection or EmptyCollection()

    def has_collection(self, _name):
        return True

    def get_collection(self, _name):
        return self.collection


class ContextOnlyDatabase:
    """只实现 __exit__，验证清理函数兼容没有 close() 的客户端。"""
    def __init__(self):
        self.events = []

    def __exit__(self, *_args):
        self.events.append("exit")


class ComparisonCollection(EmptyCollection):
    """保留混合检索的参数记录，检查自然语言与关键词是否被正确分工。"""
    def __init__(self):
        self.hybrid_calls = []

    def hybrid_search(self, **kwargs):
        self.hybrid_calls.append(kwargs)
        return super().hybrid_search(**kwargs)


class D2ImportSafetyTests(unittest.TestCase):
    """导入应加载配置并定义函数，实际数据库操作应等 main 被调用后再发生。"""
    def test_fresh_d2_entry_loads_database_settings_from_code_env(self):
        """在全新子进程中验证 .env 生效，避免已导入的模块缓存掩盖问题。"""
        code_dir = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            copied_code = root / "code"
            copied_d2 = copied_code / "D2"
            copied_d2.mkdir(parents=True)
            (root / ".git").mkdir()
            for name in ("config.py", "seekdb_runtime.py"):
                shutil.copyfile(code_dir / name, copied_code / name)
            for name in ("d2_2_vector_search.py", "db_lifecycle.py"):
                shutil.copyfile(code_dir / "D2" / name, copied_d2 / name)
            (copied_code / ".env").write_text(
                "SEEKDB_MODE=server\nSEEKDB_HOST=env.example.test\n"
                "SEEKDB_PORT=3999\nSEEKDB_DATABASE=env_demo\n",
                encoding="utf-8",
            )
            (copied_code / "pyseekdb.py").write_text(
                "class Client:\n"
                "    def __init__(self, **kwargs):\n"
                "        assert kwargs['host'] == 'env.example.test'\n"
                "        assert kwargs['port'] == 3999\n"
                "        assert kwargs['database'] == 'env_demo'\n"
                "        print('env-settings-ok')\n"
                "    def has_collection(self, name): return False\n"
                "    def close(self): pass\n",
                encoding="utf-8",
            )
            # 去掉父进程的数据库配置，确保客户端参数确实来自临时 .env。
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("SEEKDB_")}
            environment["PYTHONPATH"] = str(copied_code)
            result = subprocess.run(
                [sys.executable, "-X", "utf8", str(copied_d2 / "d2_2_vector_search.py")],
                cwd=temp_dir, env=environment, capture_output=True,
                text=True, encoding="utf-8", timeout=15,
            )
        # 客户端使用临时 .env 成功初始化；随后按预期报告尚未写入知识库。
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("env-settings-ok", result.stdout)
        self.assertIn("请先运行 d2_1_ingest.py", result.stdout)

    def test_importing_entry_modules_does_not_open_or_mutate_database(self):
        """把客户端构造替换为立即失败，确保仅 import 不会打开数据库。"""
        for module_name in D2_MODULES:
            with self.subTest(module=module_name):
                sys.modules.pop(module_name, None)
                with patch(
                    "pyseekdb.Client",
                    side_effect=AssertionError("导入时不应打开数据库"),
                ):
                    module = importlib.import_module(module_name)

                self.assertTrue(callable(module.main))
                self.assertTrue(callable(module.run_demo))


class D2PathAndOutputTests(unittest.TestCase):
    """验证路径、连接释放与输出提示等入口约定。"""
    def test_entry_modules_release_context_only_clients(self):
        """清理逻辑应能使用上下文协议释放连接。"""
        for module_name in D2_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                database = ContextOnlyDatabase()

                module._close_database(database)

                self.assertEqual(database.events, ["exit"])

    def test_cli_override_is_passed_to_client_as_an_absolute_path(self):
        """显式传入的路径应转换成绝对路径，避免依赖启动时的工作目录。"""
        module = importlib.import_module("D2.d2_2_vector_search")
        received_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "custom-seekdb"

            def client_factory(**kwargs):
                received_paths.append(kwargs["path"])
                return ExistingDatabase()

            with redirect_stdout(io.StringIO()):
                status = module.main(
                    client_factory=client_factory,
                    db_path=db_path,
                )

        self.assertEqual(status, 0)
        self.assertEqual(received_paths, [str(db_path.resolve())])

    def test_empty_results_do_not_print_false_success_claims(self):
        """没有返回文档时，不应出现肯定命中的绿色成功提示。"""
        for module_name in (
            "D2.d2_2_vector_search",
            "D2.d2_3_hybrid_search",
            "D2.d2_4_compare",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                output = io.StringIO()

                with redirect_stdout(output):
                    module.run_demo(EmptyCollection())

                self.assertNotIn("✅", output.getvalue())


class D2ComparisonTests(unittest.TestCase):
    """检查并排对比实验是否正确构造两类检索输入。"""
    def test_natural_language_query_and_fulltext_keyword_are_separate(self):
        """knn 使用完整问题，正文条件只使用提炼出的错误码关键词。"""
        module = importlib.import_module("D2.d2_4_compare")
        collection = ComparisonCollection()

        with redirect_stdout(io.StringIO()):
            module.run_comparison(
                collection,
                query_text="错误码 E-4012 的解决方案",
                keyword="E-4012",
            )

        hybrid_call = collection.hybrid_calls[0]
        self.assertEqual(
            hybrid_call["query"]["where_document"],
            {"$contains": "E-4012"},
        )
        self.assertEqual(
            hybrid_call["knn"]["query_texts"],
            ["错误码 E-4012 的解决方案"],
        )


class D2ChunkingLifecycleTests(unittest.TestCase):
    """检查评测入口的路径选项，以及失败降级时的集合与连接清理。"""
    def test_chunking_compare_accepts_cli_database_path_override(self):
        """确认命令行允许指定独立的本地实验目录。"""
        module = importlib.import_module("D2.d2_5_chunking_compare")
        expected = Path("/tmp/d2-test-seekdb")

        args = module.parse_args(["--db-path", str(expected)])

        self.assertEqual(args.db_path, expected)

    def test_seekdb_failure_degrades_to_memory_and_cleans_created_collection(self):
        """模拟写入失败，要求先删除本次集合、关闭连接，再使用内存后端。"""
        module = importlib.import_module("D2.d2_5_chunking_compare")
        events = []

        class FailingCollection:
            def add(self, **_kwargs):
                events.append("add")
                raise RuntimeError("写入失败")

        class FailingDatabase:
            def has_collection(self, _name):
                return False

            def create_collection(self, name):
                events.append(("create", name))
                return FailingCollection()

            def delete_collection(self, name):
                events.append(("delete", name))

            def __exit__(self, *_args):
                events.append("exit")

        with redirect_stdout(io.StringIO()):
            row, backend = module.evaluate_with_fallback(
                "fixed",
                client_factory=lambda **_kwargs: FailingDatabase(),
            )

        self.assertEqual(backend, "memory")
        self.assertEqual(row["strategy"], "fixed")
        self.assertEqual(events[-2:], [("delete", "d2_chunking_fixed"), "exit"])


if __name__ == "__main__":
    unittest.main()
