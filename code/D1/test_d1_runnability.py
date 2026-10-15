import importlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from config import Config


D1_MODULES = (
    "D1.d1_1_base",
    "D1.d1_2_multi_turn",
    "D1.d1_3_streaming",
    "D1.d1_4_tool_use_mock",
    "D1.d1_5_tool_use_seekdb",
    "D1.d1_6_agent",
)


class FakeChatModel:
    def __init__(self, content="测试回答"):
        self.content = content
        self.invocations = []

    def invoke(self, messages, **_kwargs):
        self.invocations.append(messages)
        return AIMessage(content=self.content)

    def stream(self, messages):
        self.invocations.append(messages)
        return iter([AIMessageChunk(content=self.content)])

    def bind_tools(self, _tools):
        return self


class FakeCollection:
    def add(self, **_kwargs):
        return None

    def query(self, **_kwargs):
        return {
            "documents": [["测试知识"]],
            "distances": [[0.1]],
            "metadatas": [[{}]],
        }


class FakeDatabase:
    def __init__(self, cleanup_error=None):
        self.collection = FakeCollection()
        self.cleanup_error = cleanup_error
        self.events = []
        self.exists = False

    def has_collection(self, name):
        self.events.append(("has", name))
        return self.exists

    def create_collection(self, name):
        self.events.append(("create", name))
        self.exists = True
        return self.collection

    def delete_collection(self, name):
        self.events.append(("delete", name))
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.exists = False

    def close(self):
        self.events.append(("close",))


class ContextOnlyDatabase(FakeDatabase):
    close = None

    def __exit__(self, *_args):
        self.events.append(("exit",))


class FakeAgent:
    def stream(self, _payload, **_kwargs):
        return iter([{"model": {"messages": [AIMessage(content="Agent 测试回答")]}}])


class D1ImportSafetyTests(unittest.TestCase):
    def test_importing_all_examples_does_not_initialize_models_agents_or_databases(self):
        for module_name in D1_MODULES:
            with self.subTest(module=module_name):
                sys.modules.pop(module_name, None)
                with (
                    patch(
                        "langchain.chat_models.init_chat_model",
                        side_effect=AssertionError("导入时不应初始化模型"),
                    ),
                    patch(
                        "langchain.agents.create_agent",
                        side_effect=AssertionError("导入时不应创建 Agent"),
                    ),
                    patch(
                        "pyseekdb.Client",
                        side_effect=AssertionError("导入时不应打开数据库"),
                    ),
                ):
                    module = importlib.import_module(module_name)

                self.assertTrue(callable(module.main))
                self.assertTrue(callable(module.run_demo))


class D1ApiKeyBoundaryTests(unittest.TestCase):
    def test_all_entries_use_the_configured_openai_compatible_endpoint(self):
        for module_name in D1_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                model_calls = []

                def model_factory(model_name, **kwargs):
                    model_calls.append((model_name, kwargs))
                    return FakeChatModel()

                kwargs = {"model_factory": model_factory}
                if module_name.endswith(("d1_5_tool_use_seekdb", "d1_6_agent")):
                    kwargs["client_factory"] = lambda **_kwargs: FakeDatabase()
                if module_name.endswith("d1_6_agent"):
                    kwargs["agent_factory"] = lambda **_kwargs: FakeAgent()
                with (
                    patch.object(Config, "OPENAI_API_KEY", "sk-test"),
                    patch.object(Config, "OPENAI_BASE_URL", "https://api.deepseek.com"),
                    patch.object(Config, "SILICONFLOW_API_KEY", "YOUR_API_KEY"),
                    patch.object(Config, "DASHSCOPE_API_KEY", "YOUR_API_KEY"),
                    patch.dict("os.environ", {"SEEKDB_MODE": "embedded"}),
                    redirect_stdout(io.StringIO()),
                ):
                    status = module.main(**kwargs)

                self.assertEqual(status, 0)
                expected_options = {
                    "model_provider": "openai",
                    "api_key": "sk-test",
                    "base_url": "https://api.deepseek.com",
                    "extra_body": {"thinking": {"type": "disabled"}},
                }
                if module_name.endswith("d1_6_agent"):
                    expected_options.update(timeout=30.0, max_retries=0)
                self.assertEqual(
                    model_calls,
                    [("deepseek-flash", expected_options)],
                )

    def test_missing_key_stops_before_model_and_database_initialization(self):
        for module_name, key_name in (
            ("D1.d1_1_base", "OPENAI_API_KEY"),
            ("D1.d1_2_multi_turn", "OPENAI_API_KEY"),
            ("D1.d1_3_streaming", "OPENAI_API_KEY"),
            ("D1.d1_4_tool_use_mock", "OPENAI_API_KEY"),
            ("D1.d1_5_tool_use_seekdb", "OPENAI_API_KEY"),
            ("D1.d1_6_agent", "OPENAI_API_KEY"),
        ):
            module = importlib.import_module(module_name)
            output = io.StringIO()
            kwargs = {
                "model_factory": lambda *_args, **_kwargs: self.fail("不应初始化模型"),
            }
            if module_name.endswith(("d1_5_tool_use_seekdb", "d1_6_agent")):
                kwargs["client_factory"] = (
                    lambda **_kwargs: self.fail("缺少 Key 时不应打开数据库")
                )

            with (
                patch.object(Config, key_name, "YOUR_API_KEY"),
                redirect_stdout(output),
            ):
                status = module.main(**kwargs)

            self.assertEqual(status, 1)
            self.assertIn("未配置", output.getvalue())


class D1SeekdbLifecycleTests(unittest.TestCase):
    def test_seekdb_entries_release_context_only_clients(self):
        for module_name in (
            "D1.d1_5_tool_use_seekdb",
            "D1.d1_6_agent",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                database = ContextOnlyDatabase()

                module._close_database(database)

                self.assertEqual(database.events, [("exit",)])

    def test_seekdb_entries_accept_cli_database_path_override(self):
        expected = Path("/tmp/d1-test-seekdb")

        for module_name in (
            "D1.d1_5_tool_use_seekdb",
            "D1.d1_6_agent",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                args = module.parse_args(["--db-path", str(expected)])
                self.assertEqual(args.db_path, expected)

    def test_d1_5_uses_overridden_path_and_always_cleans_up(self):
        module = importlib.import_module("D1.d1_5_tool_use_seekdb")
        database = FakeDatabase()
        received_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "seekdb"

            def client_factory(**kwargs):
                received_paths.append(kwargs["path"])
                return database

            with (
                patch.object(Config, "OPENAI_API_KEY", "sk-test"),
                redirect_stdout(io.StringIO()),
            ):
                status = module.main(
                    model_factory=lambda *_args, **_kwargs: FakeChatModel(),
                    client_factory=client_factory,
                    db_path=db_path,
                )

        self.assertEqual(status, 0)
        self.assertEqual(received_paths, [str(db_path.resolve())])
        self.assertIn(("delete", "d1_knowledge_base"), database.events)
        self.assertEqual(database.events[-1], ("close",))

    def test_d1_5_does_not_swallow_cleanup_errors(self):
        module = importlib.import_module("D1.d1_5_tool_use_seekdb")
        database = FakeDatabase(cleanup_error=RuntimeError("清理失败"))

        with (
            patch.object(Config, "OPENAI_API_KEY", "sk-test"),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "清理失败"),
        ):
            module.main(
                model_factory=lambda *_args, **_kwargs: FakeChatModel(),
                client_factory=lambda **_kwargs: database,
            )

    def test_d1_6_probes_configured_model_before_agent_and_cleans_up(self):
        module = importlib.import_module("D1.d1_6_agent")
        database = FakeDatabase()
        model = FakeChatModel(content="ok")
        model_calls = []

        def model_factory(model_name, **_kwargs):
            model_calls.append(model_name)
            return model

        with (
            patch.object(Config, "OPENAI_API_KEY", "sk-test"),
            redirect_stdout(io.StringIO()),
        ):
            status = module.main(
                model_factory=model_factory,
                agent_factory=lambda **_kwargs: FakeAgent(),
                client_factory=lambda **_kwargs: database,
                questions=["测试问题"],
            )

        self.assertEqual(status, 0)
        self.assertEqual(model_calls, ["deepseek-flash"])
        self.assertTrue(model.invocations, "创建 Agent 前应先执行真实模型探针")
        self.assertIn(("delete", "d1_agent_kb"), database.events)
        self.assertEqual(database.events[-1], ("close",))


class D1ToolProtocolTests(unittest.TestCase):
    def test_tool_examples_send_assistant_and_tool_messages_back_to_model(self):
        for module_name in ("D1.d1_4_tool_use_mock", "D1.d1_5_tool_use_seekdb"):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)

                class ToolCallingModel:
                    def __init__(self):
                        self.calls = []

                    def bind_tools(self, tools):
                        self.tool_name = tools[0].name
                        return self

                    def stream(self, messages):
                        self.calls.append(list(messages))
                        if len(self.calls) == 1:
                            return iter([AIMessageChunk(
                                content="",
                                tool_call_chunks=[{
                                    "name": self.tool_name,
                                    "args": '{"query":"检索方式"}',
                                    "id": "call_search",
                                    "index": 0,
                                }],
                            )])
                        if len(self.calls) != 2:
                            raise AssertionError("模型不应重复请求工具")
                        return iter([AIMessageChunk(content="基于检索结果的回答")])

                model = ToolCallingModel()
                with redirect_stdout(io.StringIO()):
                    if module_name.endswith("d1_5_tool_use_seekdb"):
                        response = module.run_demo(model, FakeCollection())
                    else:
                        response = module.run_demo(model)

                self.assertEqual(response.content, "基于检索结果的回答")
                self.assertIsInstance(model.calls[1][-2], AIMessage)
                self.assertIsInstance(model.calls[1][-1], ToolMessage)
                self.assertEqual(model.calls[1][-1].tool_call_id, "call_search")


class D1AgentWaitingTests(unittest.TestCase):
    def test_slow_http_probe_times_out_without_retry_or_database_access(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Event, Thread
        from langchain.chat_models import init_chat_model

        module = importlib.import_module("D1.d1_6_agent")
        release = Event()
        requests = []
        output = io.StringIO()

        class SlowHandler(BaseHTTPRequestHandler):
            def do_POST(handler_self):
                requests.append(handler_self.path)
                release.wait(3)
                handler_self.close_connection = True

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with (
                patch.object(Config, "OPENAI_API_KEY", "sk-local-test"),
                patch.object(Config, "OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"),
                redirect_stdout(output),
            ):
                status = module.main(
                    model_factory=init_chat_model,
                    request_timeout=0.2,
                    client_factory=lambda **_kwargs: self.fail("探针超时不应打开数据库"),
                )
            self.assertEqual(status, 1)
            self.assertEqual(requests, ["/v1/chat/completions"])
            self.assertIn("模型请求超时", output.getvalue())
            self.assertIn("检查模型接口", output.getvalue())
            self.assertNotIn("sk-local-test", output.getvalue())
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_repeated_tool_calls_stop_and_release_database(self):
        from langchain.agents import create_agent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.outputs import ChatGeneration, ChatResult

        module = importlib.import_module("D1.d1_6_agent")
        database = FakeDatabase()
        output = io.StringIO()

        class LoopingChatModel(BaseChatModel):
            @property
            def _llm_type(self):
                return "looping-test-chat"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                if len(messages) == 1 and messages[0].content == "请只回复 ok":
                    message = AIMessage(content="ok")
                else:
                    message = AIMessage(content="", tool_calls=[{
                        "name": "search_knowledge_base",
                        "args": {"query": "检索方式"},
                        "id": "call_search",
                    }])
                return ChatResult(generations=[ChatGeneration(message=message)])

        with (
            patch.object(Config, "OPENAI_API_KEY", "sk-test"),
            patch.dict("os.environ", {"SEEKDB_MODE": "embedded"}),
            redirect_stdout(output),
        ):
            status = module.main(
                model_factory=lambda *_args, **_kwargs: LoopingChatModel(),
                agent_factory=create_agent,
                client_factory=lambda **_kwargs: database,
                questions=["测试问题"],
            )

        self.assertEqual(status, 1)
        self.assertIn("步上限", output.getvalue())
        self.assertIn(("delete", "d1_agent_kb"), database.events)
        self.assertEqual(database.events[-1], ("close",))

    def test_probe_announces_wait_before_model_request(self):
        module = importlib.import_module("D1.d1_6_agent")
        output = io.StringIO()

        class ObservedModel(FakeChatModel):
            def invoke(model_self, messages, **kwargs):
                self.assertIn("检查模型接口", output.getvalue())
                return super().invoke(messages, **kwargs)

        with (
            patch.object(Config, "OPENAI_API_KEY", "sk-test"),
            patch.dict("os.environ", {"SEEKDB_MODE": "embedded"}),
            redirect_stdout(output),
        ):
            status = module.main(
                model_factory=lambda *_args, **_kwargs: ObservedModel(),
                agent_factory=lambda **_kwargs: FakeAgent(),
                client_factory=lambda **_kwargs: FakeDatabase(),
                questions=["测试问题"],
            )
        self.assertEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
