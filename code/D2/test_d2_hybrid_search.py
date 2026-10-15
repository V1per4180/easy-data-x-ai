"""混合检索的调用契约测试：确认整体版本要求被传给两个检索支路。"""

import io
import unittest
from contextlib import redirect_stdout

from D2.d2_3_hybrid_search import run_demo


class FakeCollection:
    """记录 hybrid_search 收到的参数，返回空结果，不执行真实检索。"""
    def __init__(self):
        self.hybrid_calls = []

    def count(self):
        return 0

    def hybrid_search(self, **kwargs):
        self.hybrid_calls.append(kwargs)
        return {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }


class HybridSearchExampleTests(unittest.TestCase):
    """检查查询条件的构造方式，不评价模型的实际检索质量。"""
    def test_filtered_hybrid_search_filters_both_search_branches(self):
        """若只限制一个支路，另一个支路可能带入不符合版本要求的候选。"""
        collection = FakeCollection()

        with redirect_stdout(io.StringIO()):
            run_demo(collection)

        filtered_call = collection.hybrid_calls[2]
        self.assertEqual(
            filtered_call,
            {
                "query": {
                    "where_document": {"$contains": "性能优化"},
                    "where": {"version": "4.2"},
                    "n_results": 5,
                },
                "knn": {
                    "query_texts": ["性能优化"],
                    "where": {"version": "4.2"},
                    "n_results": 5,
                },
                "rank": {"rrf": {}},
                "n_results": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
