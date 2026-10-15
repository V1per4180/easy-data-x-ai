"""分块算法与向量配置的离线测试，适合结合失败案例理解函数的输入约束。

算法测试使用手工向量；配置测试替换模型和 API 客户端，所以不会下载权重、
调用 DeepSeek 或访问真实向量服务。with patch(...) 中的替换只在当前测试内生效。
"""

import importlib
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from D2.d2_chunking_strategies import (
    _detect_boundary_positions,
    dynamic_overlap_chunk,
    fixed_overlap_chunk,
    parent_child_chunk,
    semantic_chunk,
)


class FixedOverlapValidationTests(unittest.TestCase):
    """检查固定窗口的合法范围，保证算法能向前推进。"""
    def test_rejects_non_positive_size_and_invalid_overlap(self):
        """非法参数即使配上空文本，也应立即报错。"""
        invalid_parameters = (
            {"chunk_size": 0, "overlap": 0},
            {"chunk_size": -1, "overlap": 0},
            {"chunk_size": 3, "overlap": -1},
            {"chunk_size": 3, "overlap": 3},
            {"chunk_size": 3, "overlap": 4},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    fixed_overlap_chunk("", **parameters)


class SemanticChunkingTests(unittest.TestCase):
    """用可控向量验证语义分块的边界处理，不依赖模型输出。"""
    def test_rejects_non_numeric_non_finite_or_out_of_range_percentile(self):
        """阈值只能是 0～100 的有限数值，bool 也不作为有效数值使用。"""
        text = "第一句。第二句。"
        invalid_values = (None, "95", math.nan, math.inf, -1, 101)

        for percentile in invalid_values:
            with self.subTest(percentile=percentile):
                with self.assertRaisesRegex(ValueError, "percentile"):
                    semantic_chunk(
                        text,
                        lambda _sentences: [[1.0, 0.0], [0.0, 1.0]],
                        percentile=percentile,
                        min_chunk_chars=1,
                        max_chunk_chars=100,
                    )

    def test_identical_embeddings_do_not_create_breakpoints(self):
        """所有句向量相同，差异度全为零时，不应人为制造语义断点。"""
        text = "连接池需要合理配置。连接池需要持续监控。连接池需要避免泄漏。"

        def embed_same(sentences):
            return [[1.0, 0.0] for _ in sentences]

        chunks = semantic_chunk(
            text,
            embed_same,
            min_chunk_chars=1,
            max_chunk_chars=1000,
        )

        self.assertEqual(chunks, [text])

    def test_rejects_invalid_minimum_and_maximum_lengths(self):
        """最小值与最大值都需为正，且最小长度不能超过最大长度。"""
        invalid_parameters = (
            {"min_chunk_chars": 0, "max_chunk_chars": 10},
            {"min_chunk_chars": 1, "max_chunk_chars": 0},
            {"min_chunk_chars": 11, "max_chunk_chars": 10},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    semantic_chunk("", lambda _sentences: [], **parameters)

    def test_rejects_embedding_count_mismatch(self):
        """每句必须有一个对应向量；数量不足或过多都不能继续计算相邻关系。"""
        text = "第一句。第二句。第三句。"

        for embeddings in (
            [[1.0, 0.0]],
            [[1.0, 0.0]] * 4,
        ):
            with self.subTest(embedding_count=len(embeddings)):
                with self.assertRaisesRegex(ValueError, "数量"):
                    semantic_chunk(text, lambda _sentences, result=embeddings: result)

    def test_rejects_empty_mismatched_and_non_finite_embeddings(self):
        """空向量、维度不齐、NaN 和无穷值应在计算相似度前被拦截。"""
        text = "第一句。第二句。"
        invalid_embeddings = (
            [[], []],
            [[1.0, 0.0], [1.0]],
            [[1.0, math.nan], [1.0, 0.0]],
            [[1.0, math.inf], [1.0, 0.0]],
        )

        for embeddings in invalid_embeddings:
            with self.subTest(embeddings=embeddings):
                with self.assertRaisesRegex(ValueError, "向量"):
                    semantic_chunk(text, lambda _sentences, result=embeddings: result)

    def test_never_returns_chunks_larger_than_maximum(self):
        """即使全文只有一句、无法做句间比较，也必须遵守最大字符数限制。"""
        text = "甲" * 250

        chunks = semantic_chunk(
            text,
            lambda _sentences: [],
            min_chunk_chars=1,
            max_chunk_chars=100,
        )

        self.assertEqual(chunks, ["甲" * 100, "甲" * 100, "甲" * 50])


class DynamicOverlapTests(unittest.TestCase):
    """检查结构边界识别和动态重叠的参数范围。"""
    def test_detects_markdown_structure_boundaries(self):
        """用短 Markdown 样本确认表格和代码块标记能被识别为边界。"""
        text = (
            "# 标题\n"
            "正文段落\n\n"
            "| 字段 | 含义 |\n"
            "| --- | --- |\n"
            "| code | 错误码 |\n\n"
            "```python\n"
            "print('hello')\n"
            "```\n"
        )

        boundaries = _detect_boundary_positions(text)

        self.assertIn(text.index("| 字段 | 含义 |"), boundaries)
        self.assertIn(text.index("```python"), boundaries)
        self.assertIn(text.rindex("```"), boundaries)

    def test_rejects_invalid_sizes_and_overlaps(self):
        """保持 0 <= base_overlap <= max_overlap < chunk_size。"""
        invalid_parameters = (
            {"chunk_size": 0, "base_overlap": 0, "max_overlap": 0},
            {"chunk_size": 10, "base_overlap": -1, "max_overlap": 2},
            {"chunk_size": 10, "base_overlap": 3, "max_overlap": 2},
            {"chunk_size": 10, "base_overlap": 2, "max_overlap": 10},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    dynamic_overlap_chunk("", **parameters)


class ParentChildValidationTests(unittest.TestCase):
    """父块和子块都基于固定窗口，因此都需要满足窗口参数约束。"""
    def test_rejects_invalid_sizes_and_child_overlap(self):
        """验证父块长度、子块长度及子块重叠值的非法情况。"""
        invalid_parameters = (
            {"parent_size": 0, "child_size": 10, "child_overlap": 0},
            {"parent_size": 10, "child_size": 0, "child_overlap": 0},
            {"parent_size": 10, "child_size": 5, "child_overlap": -1},
            {"parent_size": 10, "child_size": 5, "child_overlap": 5},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    parent_child_chunk("", **parameters)


class CompareScriptTests(unittest.TestCase):
    """验证评测脚本可被导入，以及本地/API 两种向量来源的配置边界。"""
    def test_compare_script_can_be_imported_as_module(self):
        """导入脚本应只定义函数，不直接开始评测。"""
        module = importlib.import_module("D2.d2_5_chunking_compare")

        self.assertTrue(hasattr(module, "main"))

    def test_blank_and_template_api_keys_are_not_available(self):
        """空白和示例占位 Key 应被视为未配置，防止发送无效请求。"""
        module = importlib.import_module("D2.d2_5_chunking_compare")

        for api_key in (
            "   ",
            "  YOUR_API_KEY  ",
            "  your_siliconflow_api_key_here  ",
        ):
            with self.subTest(api_key=api_key):
                with (
                    patch.object(module.Config, "D2_EMBEDDING_PROVIDER", "api"),
                    patch.object(module.Config, "D2_EMBEDDING_API_KEY", api_key),
                    patch.object(module.Config, "D2_EMBEDDING_BASE_URL", "https://example.test/v1"),
                    patch.object(module.Config, "D2_EMBEDDING_MODEL", "test-embedding"),
                ):
                    self.assertFalse(module.has_embedding_config())

    def test_local_embedding_does_not_require_or_call_a_chat_api(self):
        """本地模式只调用本地向量模型；API 工厂一旦被调用就立即使测试失败。"""
        module = importlib.import_module("D2.d2_5_chunking_compare")
        local_model = Mock(return_value=[[1.0, 0.0], [0.0, 1.0]])
        with (
            patch.object(module.Config, "D2_EMBEDDING_PROVIDER", "local"),
            patch("pyseekdb.DefaultEmbeddingFunction", return_value=local_model),
            patch.object(module, "OpenAI", side_effect=AssertionError("本地模式不应创建 API 客户端")),
        ):
            embed_fn = module.build_embed_fn()
            vectors = embed_fn(["第一句", "第二句"])
            embed_fn.close()

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        local_model.assert_called_once_with(["第一句", "第二句"])

    def test_api_embedding_uses_its_own_endpoint_model_and_returns_input_order(self):
        """确认向量服务与聊天配置分离，并能将乱序响应按 index 还原。"""
        module = importlib.import_module("D2.d2_5_chunking_compare")
        client = Mock()
        client.embeddings.create.return_value.data = [
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ]
        with (
            patch.multiple(module.Config,
                D2_EMBEDDING_PROVIDER="api",
                D2_EMBEDDING_API_KEY="  sk-embedding-test  ",
                D2_EMBEDDING_BASE_URL="https://embedding.example.test/v1",
                D2_EMBEDDING_MODEL="embedding-test",
                OPENAI_API_KEY="sk-chat-test",
                OPENAI_BASE_URL="https://api.deepseek.com",
            ),
            patch.object(module, "OpenAI", return_value=client) as factory,
        ):
            embed_fn = module.build_embed_fn()
            vectors = embed_fn(["第一句", "第二句"])
            embed_fn.close()

        factory.assert_called_once_with(
            api_key="sk-embedding-test", base_url="https://embedding.example.test/v1",
            timeout=30.0, max_retries=0,
        )
        client.embeddings.create.assert_called_once_with(
            model="embedding-test", input=["第一句", "第二句"],
        )
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        client.close.assert_called_once_with()

    def test_incomplete_api_configuration_fails_before_sending_credentials(self):
        """缺少向量 API 参数时立即失败，不借用聊天 Key，也不创建 API 客户端。"""
        module = importlib.import_module("D2.d2_5_chunking_compare")
        with (
            patch.multiple(module.Config,
                D2_EMBEDDING_PROVIDER="api", D2_EMBEDDING_API_KEY="",
                D2_EMBEDDING_BASE_URL="", D2_EMBEDDING_MODEL="",
                OPENAI_API_KEY="sk-chat-test",
            ),
            patch.object(module, "OpenAI") as factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "D2_EMBEDDING_API_KEY"):
                module.build_embed_fn()
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
