import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from seekdb_runtime import create_seekdb_client, require_destructive_seekdb_access

if __package__:
    from .tool_call_loop import run_tool_call_loop
else:
    from tool_call_loop import run_tool_call_loop


D1_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = D1_DIR / "seekdb"
COLLECTION_NAME = "d1_knowledge_base"
DOCS = [
    "seekdb 支持嵌入模式和服务器模式。",
    "seekdb 支持关键词检索、语义向量检索和混合检索。",
    "pyseekdb 是 seekdb 的 Python SDK，支持 Schemaless API。",
    "seekdb 使用 RRF 融合关键词与向量检索结果。",
]


def _close_database(database):
    """兼容 close() 和上下文管理协议两种 pyseekdb 客户端。"""
    close = getattr(database, "close", None)
    if callable(close):
        close()
        return
    exit_method = getattr(database, "__exit__", None)
    if callable(exit_method):
        exit_method(None, None, None)


def build_search_tool(collection):
    @tool
    def search_seekdb(query: str) -> str:
        """从 seekdb 知识库中检索产品功能和技术文档。"""
        results = collection.query(query_texts=[query], n_results=3)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "未找到相关文档。"
        return "\n\n".join(f"[文档{i + 1}] {doc}" for i, doc in enumerate(docs))

    return search_seekdb


def print_tool_result(tool_call, result):
    print(f">>> 模型决定调用工具：{tool_call['name']}")
    print(f">>> 工具参数：{tool_call['args']}\n")
    print(f">>> seekdb 检索结果：\n{result}\n")


def run_demo(llm, collection):
    search_tool = build_search_tool(collection)
    response = run_tool_call_loop(
        llm.bind_tools([search_tool]),
        [
            ("system", "你是技术文档助手，seekdb 问题请先查询知识库。"),
            ("user", "seekdb 支持哪些检索方式？混合检索是怎么实现的？"),
        ],
        {search_tool.name: search_tool},
        max_rounds=5,
        on_tool_result=print_tool_result,
    )
    print(f">>> 模型最终回答：\n{response.content}")
    return response


def main(
    model_factory=init_chat_model,
    client_factory=create_seekdb_client,
    db_path=None,
) -> int:
    # Key 边界：缺少 Key 时不创建模型，也不触碰本地数据库。
    try:
        Config.require_api_key("OPENAI_API_KEY")
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1

    llm = model_factory(
        "deepseek-flash",
        model_provider="openai",
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_BASE_URL,
        extra_body={"thinking": {"type": "disabled"}},
    )
    # 路径边界：默认锚定 D1 目录，测试和 CLI 可显式覆盖。
    resolved_path = Path(db_path or DEFAULT_DB_PATH).expanduser().resolve()
    require_destructive_seekdb_access("重建 D1 演示知识库")
    db = client_factory(path=str(resolved_path))
    created = False
    try:
        if db.has_collection(COLLECTION_NAME):
            db.delete_collection(COLLECTION_NAME)
        collection = db.create_collection(name=COLLECTION_NAME)
        created = True
        collection.add(
            ids=[f"doc_{i}" for i in range(len(DOCS))],
            documents=DOCS,
        )
        run_demo(llm, collection)
        return 0
    finally:
        try:
            # 资源清理边界：仅清理本次创建的集合，其他异常不得静默吞掉。
            if created and db.has_collection(COLLECTION_NAME):
                db.delete_collection(COLLECTION_NAME)
                print("\n>>> 知识库已清理")
        finally:
            _close_database(db)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(db_path=parse_args().db_path))
