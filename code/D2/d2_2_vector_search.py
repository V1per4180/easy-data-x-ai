"""D2 第二步：只用向量相似性检索，观察语义查询和精确编号查询的差异。

先运行 d2_1_ingest.py。本文件读取现有集合，不重新写入文档。
重点看 run_demo：query_texts 由 SDK 转为向量，再到集合中寻找相近片段。
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
    """取第一条查询的文档列表；SDK 按“查询批次 → 命中列表”组织返回值。"""
    return results.get("documents", [[]])[0]


def print_results(results: dict, title: str):
    """把命中文本、距离和元数据并排打印，方便观察结果，不参与检索。"""
    print(f"{'=' * 50}\n  {title}\n{'=' * 50}")
    docs = _documents(results)
    # 这里取的是 SDK 的 distances；它不是概率，不能理解为“答案正确率”。
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not docs:
        print("  （无结果）")
    for index, (doc, distance, metadata) in enumerate(
        zip(docs, distances, metadatas), 1
    ):
        print(f"  [{index}] 相关度分数：{distance:.4f}")
        print(f"       内容：{doc[:80]}")
        print(
            f"       分类：{metadata.get('category', '-')}  "
            f"版本：{metadata.get('version', '-')}"
        )


def run_demo(collection):
    """对同一知识库执行两种查询，观察纯向量检索的能力与局限。

    第一个问题用“用户权限”寻找正文中的“访问控制/RBAC”，观察语义关联。
    第二个问题带 E-4012，观察向量模型能否区分相近的错误编号。
    n_results=3 表示最多取前三条；检索排名受向量模型、数据和索引状态影响。

    返回两次查询的原始结果，便于在交互环境继续分析。
    本示例只展示检索结果，不让聊天模型根据文档生成回答。
    """
    semantic = collection.query(query_texts="怎么设计用户权限", n_results=3)
    print_results(semantic, "向量搜索：语义理解")
    # 关键词命中只是演示用检查，不能代替人工判断或完整的检索质量评测。
    semantic_docs = _documents(semantic)
    if any("访问控制" in doc or "RBAC" in doc for doc in semantic_docs):
        print("✅ 结果中找到了访问控制/RBAC 相关文档。\n")
    else:
        print("⚠️  本次结果未命中访问控制/RBAC，不能断言语义检索成功。\n")

    error_code = collection.query(
        query_texts="错误码 E-4012 的解决方案",
        n_results=3,
    )
    print_results(error_code, "向量搜索：精确编号")
    docs = _documents(error_code)
    if docs and "E-4012" in docs[0]:
        print("✅ 本次向量搜索第一名恰好是 E-4012；实际排名仍可能受模型影响。\n")
    elif docs:
        print("⚠️  第一名不是 E-4012，展示了纯向量检索处理精确编号的风险。\n")
    else:
        print("⚠️  本次没有返回结果。\n")
    return semantic, error_code


def main(client_factory=create_seekdb_client, db_path=None) -> int:
    """按 .env 连接数据库，确认知识库存在后查询，最后释放连接。"""
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
    """读取 Embedded 模式可选的数据库目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(db_path=_parse_args().db_path))
