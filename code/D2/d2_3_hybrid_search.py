"""D2 第三步：把文本条件与向量检索组合起来，并加入元数据过滤。

先运行 d2_1_ingest.py。阅读 run_demo 时重点区分 query、knn、rank 三部分。
混合检索先分别取候选，再融合排名；某一支路的条件不会自动成为整体硬过滤。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seekdb_runtime import create_seekdb_client

if __package__:
    from .db_lifecycle import close_database as _close_database
else:
    from db_lifecycle import close_database as _close_database


D2_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = D2_DIR / "seekdb"
COLLECTION_NAME = "d2_knowledge_base"


def _documents(results):
    """从批量查询格式中取出第一条查询命中的文档。"""
    return results.get("documents", [[]])[0]


def print_results(results: dict, title: str):
    """格式化混合检索结果；展示分数只用于观察当前返回顺序。"""
    print(f"{'=' * 50}\n  {title}\n{'=' * 50}")
    docs = _documents(results)
    # 字段名沿用 SDK 返回结构；混合检索的融合分数不能直接与纯向量距离比较。
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not docs:
        print("  （无结果）")
    for index, (doc, distance, metadata) in enumerate(
        zip(docs, distances, metadatas), 1
    ):
        print(f"  [{index}] 分数：{distance:.4f}  内容：{doc[:80]}")
        print(
            f"       分类：{metadata.get('category', '-')}  "
            f"版本：{metadata.get('version', '-')}"
        )


def run_demo(collection):
    """依次演示精确编号、语义理解和带版本过滤的混合检索。

    query：按正文中的词项条件获取候选，适合保留错误码、产品名等字面信息。
    knn：把 query_texts 转为向量，寻找表达方式不同但可能语义相关的文档。
    rank：这里使用 RRF 按各路候选的名次融合，而非直接相加不同量纲的分数。
    两个支路各取最多 5 条候选，融合后由外层 n_results=3 决定最终条数。

    返回三组原始结果。注意：只在 query 支路指定 E-4012，并不保证
    所有最终结果都包含该编号，knn 支路仍可能贡献其他文档。
    """
    exact = collection.hybrid_search(
        query={"where_document": {"$contains": "E-4012"}, "n_results": 5},
        knn={"query_texts": ["错误码 E-4012 的解决方案"], "n_results": 5},
        rank={"rrf": {}},
        n_results=3,
    )
    print_results(exact, "混合搜索：精确编号")
    exact_docs = _documents(exact)
    if exact_docs and "E-4012" in exact_docs[0]:
        print("✅ E-4012 位于混合搜索第一名。\n")
    else:
        print("⚠️  本次结果未将 E-4012 排在第一名。\n")

    semantic = collection.hybrid_search(
        query={"where_document": {"$contains": "权限"}, "n_results": 5},
        knn={"query_texts": ["怎么设计用户权限"], "n_results": 5},
        rank={"rrf": {}},
        n_results=3,
    )
    print_results(semantic, "混合搜索：语义理解")
    semantic_docs = _documents(semantic)
    if any("访问控制" in doc or "RBAC" in doc for doc in semantic_docs):
        print("✅ 结果中找到了访问控制/RBAC 相关文档。\n")
    else:
        print("⚠️  本次结果未命中访问控制/RBAC。\n")

    # version 是整次查询都必须满足的要求，所以要同时限制 query 和 knn 两个支路。
    # where 筛结构化 metadata；where_document 筛正文内容，两者针对的数据不同。
    filtered = collection.hybrid_search(
        query={
            "where_document": {"$contains": "性能优化"},
            "where": {"version": "4.2"},
            "n_results": 5,
        },
        knn={
            "query_texts": ["性能优化"],
            "where": {"version": "4.2"},
            "n_results": 5,
        },
        rank={"rrf": {}},
        n_results=3,
    )
    print_results(filtered, "混合搜索 + 版本过滤")
    docs = _documents(filtered)
    metas = filtered.get("metadatas", [[]])[0]
    if docs and metas and all(meta.get("version") == "4.2" for meta in metas):
        print("✅ 返回结果均满足 version=4.2。\n")
    else:
        print("⚠️  无结果或结果未全部满足 version=4.2。\n")
    return exact, semantic, filtered


def main(client_factory=create_seekdb_client, db_path=None) -> int:
    """读取现有知识库并运行混合检索，退出时关闭客户端。"""
    resolved_path = Path(db_path or DEFAULT_DB_PATH).expanduser().resolve()
    print(">>> 正在按 .env / 环境变量连接 seekdb...", flush=True)
    db = client_factory(path=str(resolved_path))
    try:
        if not db.has_collection(COLLECTION_NAME):
            print("❌ 未找到知识库，请先运行 d2_1_ingest.py 写入数据")
            return 1
        collection = db.get_collection(COLLECTION_NAME)
        print(f">>> 已连接知识库：{COLLECTION_NAME}，共 {collection.count()} 条文档\n")
        run_demo(collection)
        return 0
    finally:
        _close_database(db)


def _parse_args():
    """读取可选的本地数据库路径。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(db_path=_parse_args().db_path))
