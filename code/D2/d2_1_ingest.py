"""D2 第一步：把原始文档分块，连同元数据一起写入 seekdb。

建议阅读顺序：chunk_document → run_demo → main。
本文件负责准备知识库，后续 d2_2～d2_4 会读取同一个集合。
向量由 seekdb SDK 的默认本地模型生成，这里不调用 DeepSeek 聊天接口。
"""

import argparse
import sys
from pathlib import Path

# 将 code 目录加入导入路径，保证从项目根目录或其他目录启动都能找到公共模块。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seekdb_runtime import create_seekdb_client, require_destructive_seekdb_access

if __package__:
    from .db_lifecycle import close_database as _close_database
else:
    from db_lifecycle import close_database as _close_database


D2_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = D2_DIR / "seekdb"
COLLECTION_NAME = "d2_knowledge_base"
# 每条原始文档包含正文、分类、版本；后两项会成为可用于过滤的 metadata。
RAW_DOCS = [
    ("错误码 E-4012 表示数据库连接超时。解决方案：检查网络和端口，超时建议设为 30 秒。", "error_codes", "4.2"),
    ("错误码 E-4013 表示认证失败。请检查用户名、密码和账户锁定状态。", "error_codes", "4.2"),
    ("错误码 E-4011 表示连接被拒绝。请确认数据库服务和防火墙规则。", "error_codes", "4.2"),
    ("数据库查询性能优化指南：为高频 WHERE 条件列建立索引，避免在索引列上使用函数。", "best_practices", "4.2"),
    ("访问控制架构设计：基于 RBAC 实现用户权限管理，并按最小权限原则分配角色。", "architecture", "4.1"),
    ("OB-4.2.1 支持在线 DDL，并改进并行查询引擎。升级前请备份数据。", "release_notes", "4.2.1"),
    ("数据备份与恢复：每天全量备份，每小时增量备份，并使用独立存储介质。", "best_practices", "4.2"),
    ("连接池配置：最大连接数建议为 CPU 核心数的 2-4 倍，连接超时建议 30 秒。", "best_practices", "4.2"),
]


def chunk_document(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """用固定长度滑动窗口，把一篇文档切成若干字符串。

    chunk_size 和 overlap 的单位都是 Python 字符数，不是模型的 token 数。
    例如 chunk_size=200、overlap=30，每次窗口前进 170 个字符；
    前一块的末尾 30 个字符会出现在下一块开头，帮助保留跨块的信息。

    overlap 必须小于 chunk_size，否则窗口无法向前推进。
    返回非空片段列表；输入为空或全为空白时返回空列表。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size].strip())
        start += chunk_size - overlap  # 步长 = 窗口长度 - 与上一块重叠的长度。
    return [chunk for chunk in chunks if chunk]


def run_demo(db) -> int:
    """重建演示知识库，完成“文档 → 片段 → 向量与元数据”的写入流程。

    db 是已经连接的客户端；一个 collection 可以理解为这次实验的文档集合。
    ids、documents、metadatas 按相同下标一一对应，不能分别打乱顺序。
    add() 未显式传入 embeddings，所以 SDK 会先把 documents 转成向量再写入。

    返回数据库中的实际条数，用来与预期片段数比较。
    注意：重复运行会删除同名集合并重建；完成后保留集合供后续示例查询。
    """
    if db.has_collection(COLLECTION_NAME):
        db.delete_collection(COLLECTION_NAME)
        print(f">>> 已删除旧的集合：{COLLECTION_NAME}")
    print(">>> 创建集合并加载本地向量模型（首次可能需要下载）...", flush=True)
    collection = db.create_collection(name=COLLECTION_NAME)

    # ID 同时记录来源文档和片段序号，方便定位检索结果对应的原文。
    ids, texts, metadatas = [], [], []
    for doc_index, (text, category, version) in enumerate(RAW_DOCS):
        for chunk_index, chunk in enumerate(chunk_document(text)):
            ids.append(f"doc_{doc_index}_chunk_{chunk_index}")
            texts.append(chunk)
            # 分类和版本保留为结构化字段，后续可以按 version 等条件过滤。
            metadatas.append(
                {
                    "category": category,
                    "version": version,
                    "source_doc_idx": doc_index,
                }
            )
    print(f">>> 正在生成向量并写入 {len(texts)} 个片段...", flush=True)
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    stored_count = collection.count()
    print(f">>> 原始文档数：{len(RAW_DOCS)}")
    print(f">>> 切分后片段数：{len(ids)}")
    print(f">>> seekdb 中实际存储条数：{stored_count}")
    if stored_count == len(ids):
        print("✅ d2_1 写入完成，可继续运行 d2_2 / d2_3 / d2_4。")
    else:
        print("⚠️  实际存储条数与预期不一致，请检查写入日志。")
    return stored_count


def main(client_factory=create_seekdb_client, db_path=None) -> int:
    """连接数据库、运行写入演示，并在成功或失败后关闭连接。

    client_factory 可替换成测试客户端，因此测试无需真实数据库。
    公共连接模块会加载 .env；Server 模式使用 SEEKDB_*，Embedded 模式使用 db_path。
    """
    # 路径边界：默认锚定 D2 章节目录，不使用当前工作目录。
    resolved_path = Path(db_path or DEFAULT_DB_PATH).expanduser().resolve()
    require_destructive_seekdb_access("重建 D2 演示知识库")
    print(">>> 正在按 .env / 环境变量连接 seekdb...", flush=True)
    db = client_factory(path=str(resolved_path))
    try:
        run_demo(db)
        return 0
    finally:
        # 关闭客户端不会删除知识库，d2_2～d2_4 仍可读取已写入的数据。
        _close_database(db)


def _parse_args():
    """读取可选的本地数据库路径；Server 模式下连接参数仍来自 .env。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(db_path=_parse_args().db_path))
