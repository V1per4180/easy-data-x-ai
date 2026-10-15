"""写入示例的参数校验测试：检查分块参数，运行时无需连接数据库。"""

import unittest

from D2.d2_1_ingest import chunk_document

class FakeCollection:
    """集合替身，只提供测试可能用到的最小方法。"""
    def add(self, **_kwargs):
        return None

    def count(self):
        return 0


class FakeDatabase:
    """数据库替身，避免单元测试触碰真实知识库。"""
    def has_collection(self, _name):
        return False

    def create_collection(self, name):
        return FakeCollection()


class IngestChunkValidationTests(unittest.TestCase):
    """分块步长必须为正，错误参数应在进入循环前被拒绝。"""
    def test_rejects_non_positive_size_and_invalid_overlap(self):
        """覆盖零长度、负长度，以及会让窗口停滞或后退的重叠值。"""
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
                    chunk_document("", **parameters)


if __name__ == "__main__":
    unittest.main()
