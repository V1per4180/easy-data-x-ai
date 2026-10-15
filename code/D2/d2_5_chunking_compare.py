"""D2 第五步：用同一篇文档、同一组问题比较四种分块策略。

主流程：main → evaluate_strategy → prepare_strategy_chunks → 检索器 → 命中统计。
算法细节在 d2_chunking_strategies.py；本文件负责把算法接到数据库和评测流程。
分块决定“存什么片段”，检索后端决定“怎样找片段”，两者要分别观察。
本实验不生成最终问答，因此报告不评价聊天模型的回答质量。
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI, OpenAIError

from config import Config

if __package__:
    from .db_lifecycle import close_database as _close_database
else:
    from db_lifecycle import close_database as _close_database

if __package__:
    from .d2_chunking_strategies import (
        dynamic_overlap_chunk,
        fixed_overlap_chunk,
        flatten_parent_child,
        parent_child_chunk,
        semantic_chunk,
    )
else:
    from d2_chunking_strategies import (
        dynamic_overlap_chunk,
        fixed_overlap_chunk,
        flatten_parent_child,
        parent_child_chunk,
        semantic_chunk,
    )

# ============================================================
# d2_5：高级 Chunking 策略对比实验
#
# 对比三种策略 + 固定 overlap 基线：
#   1. 固定 overlap（基线）
#   2. 语义分块
#   3. 动态 overlap
#   4. 父子 chunk（子块检索 → 父块上下文）
#
# 运行前：
#   - pip install pyseekdb openai
#   - 优先使用 seekdb 向量检索；不可用时自动降级为内存检索
#   - 语义分块默认使用本地向量模型；API 模式通过 D2_EMBEDDING_* 配置
#
# 给学生的背景说明：
#   - 本实验同时包含「分块阶段」和「检索阶段」两个维度。
#   - seekdb 嵌入式模式不可用时，只影响检索后端，会降级为内存词项匹配。
#   - D2_EMBEDDING_* 只控制语义分块阶段，不改变 seekdb 默认的检索向量模型。
#   - OPENAI_* 仍用于 DeepSeek 聊天，不用于 Embedding。
# ============================================================

D2_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = D2_DIR / "seekdb"

SAMPLE_LONG_DOC = """
# 数据库运维手册

## 错误码参考

错误码 E-4011 表示连接被拒绝。解决方案：确认数据库服务是否正在运行，检查防火墙规则是否允许对应端口的访问。若服务正常但仍被拒绝，请核对白名单配置。

错误码 E-4012 表示数据库连接超时。解决方案：检查网络配置，确认数据库服务端口是否开放，建议超时时间设置为 30 秒。同时检查中间网络设备是否存在丢包。

错误码 E-4013 表示认证失败。解决方案：检查用户名和密码是否正确，确认账户是否被锁定。如果账户被锁定，需要联系管理员解锁后重试。

## 性能优化

数据库查询性能优化指南：合理使用索引可以将查询速度提升 10 倍以上。建议对高频查询的 WHERE 条件列建立索引。避免在索引列上使用函数，否则会导致索引失效。

执行计划分析是定位慢查询的关键步骤。开启慢查询日志后，优先优化扫描行数最多的 SQL。对于大表分页，推荐使用延迟关联或游标分页替代深分页 OFFSET。

## 访问控制

访问控制架构设计：基于 RBAC 模型实现用户权限管理，支持角色继承和细粒度的资源级权限控制。建议按最小权限原则分配角色，定期审计权限配置。

多租户场景下，应在连接层注入租户标识，并在 SQL 层强制附加租户过滤条件，避免跨租户数据泄露。

## 版本发布

OB-4.2.1 版本新特性：支持在线 DDL 操作、改进了并行查询引擎、修复了分区表在特定条件下的数据倾斜问题。升级前请备份数据并阅读升级指南。

## 备份与恢复

数据备份与恢复：建议每天进行全量备份，每小时进行增量备份。恢复时优先使用最近的全量备份，再应用增量备份。备份文件应存储在独立的存储介质上。

## 连接池配置

连接池配置最佳实践：最大连接数建议设置为 CPU 核心数的 2-4 倍。连接超时时间建议设置为 30 秒，空闲连接超时建议设置为 600 秒。监控活跃连接数，避免连接泄漏。
""".strip()


# 每个问题搭配一个人工指定的预期关键词；评测时检查检索上下文中是否出现它。
# 这是一组固定的小样本，便于重复观察，不能代表真实业务的所有问题。
EVAL_QUERIES = [
    {"query": "错误码 E-4012 连接超时怎么解决", "keyword": "E-4012"},
    {"query": "RBAC 角色权限如何设计", "keyword": "RBAC"},
    {"query": "OB-4.2.1 在线 DDL 新特性", "keyword": "在线 DDL"},
    {"query": "连接池最大连接数设多少", "keyword": "连接池"},
    {"query": "慢查询执行计划怎么分析", "keyword": "执行计划"},
]


def _extract_query_terms(query: str) -> list[str]:
    """从查询中提取中英文关键词片段。"""
    terms = re.findall(r"[A-Za-z0-9][\w.-]*|[\u4e00-\u9fff]{2,}", query)
    return [t for t in terms if len(t) >= 2]


def _lexical_score(query: str, text: str) -> float:
    """按查询词项在文本中出现的数量打分；属于简化词项匹配，不计算向量。"""
    terms = _extract_query_terms(query)
    if not terms:
        return 0.0
    return sum(1.0 for term in terms if term in text)


class InMemoryRetriever:
    """把片段保存在内存中，按词项匹配分数返回结果。

    对外提供与 SeekdbRetriever 相同的 search/search_with_context 方法，
    因此评测代码不必关心底层实现。降级后检索依据已经改变，
    比较策略效果时应同时关注实际使用的后端。
    """

    def __init__(self, texts: list[str], metadatas: list[dict]):
        """保存按下标对应的文本和元数据，不在构造时执行搜索。"""
        self.texts = texts
        self.metadatas = metadatas

    def search(self, query: str, k: int = 3) -> list[str]:
        """为所有片段打分并取前 k 条；同分时保留原有顺序，零分候选也可能返回。"""
        scored = [
            (_lexical_score(query, text), idx)
            for idx, text in enumerate(self.texts)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        top_indices = [idx for _, idx in scored[:k]]
        return [self.texts[idx] for idx in top_indices]

    def search_with_context(self, query: str, k: int = 3) -> list[str]:
        """命中子块后附带 metadata 中的父文；本教学实现暂不去重父文。"""
        scored = [
            (_lexical_score(query, text), idx)
            for idx, text in enumerate(self.texts)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        contexts: list[str] = []
        for _, idx in scored[:k]:
            contexts.append(self.texts[idx])
            parent_text = self.metadatas[idx].get("parent_text", "")
            if parent_text:
                contexts.append(parent_text)
        return contexts


class SeekdbRetriever:
    """把 seekdb 集合包装成评测可使用的检索接口。

    collection 提供真实向量检索，cleanup 负责实验完成后的集合与连接清理。
    它和 InMemoryRetriever 使用相同的方法名，评测函数可复用同一套调用。
    """

    def __init__(self, collection, cleanup):
        """保存集合及清理回调，清理时机交给外层评测函数决定。"""
        self.collection = collection
        self.cleanup = cleanup

    def search(self, query: str, k: int = 3) -> list[str]:
        """把问题交给集合做向量检索，返回第一条查询命中的片段。"""
        results = self.collection.query(query_texts=[query], n_results=k)
        return results.get("documents", [[]])[0]

    def search_with_context(self, query: str, k: int = 3) -> list[str]:
        """先检索最多 k 个子块，再附加对应父文；返回文本数可能大于 k。"""
        results = self.collection.query(query_texts=[query], n_results=k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        contexts = list(docs)
        for meta in metas:
            parent_text = meta.get("parent_text", "")
            if parent_text:
                contexts.append(parent_text)
        return contexts


def build_embed_fn():
    """根据 .env 构造语义分块专用的批量向量化函数。

    local 使用 seekdb 默认的本地模型，首次加载可能下载权重；api 使用
    D2_EMBEDDING_* 指定的独立服务，网络超时设为 30 秒且不自动重试。
    返回的函数接收字符串列表，输出顺序对应的向量列表，并附带 close()。

    这个函数只负责判断句间语义差异。把分好的片段写入 seekdb 时，
    集合仍使用自己的默认向量模型；两阶段可以采用不同的模型。
    OPENAI_* 保留给 DeepSeek 聊天，此处不会读取它作为向量接口凭据。
    """
    provider = Config.D2_EMBEDDING_PROVIDER.strip().lower()
    if provider == "local":
        from pyseekdb import DefaultEmbeddingFunction

        print(">>> 加载本地语义分块模型（首次可能需要下载，无需 API Key）...", flush=True)
        embedding_function = DefaultEmbeddingFunction()

        def embed_batch(texts: list[str]) -> list[list[float]]:
            """本地批量向量化，保持输出与输入文本一一对应。"""
            return embedding_function(texts)

        # 统一调用接口：本地模型没有这里需要关闭的 HTTP 客户端，所以 close 为空操作。
        embed_batch.close = lambda: None
        return embed_batch

    if provider != "api":
        raise RuntimeError("D2_EMBEDDING_PROVIDER 只能为 local 或 api")
    if not has_embedding_config():
        raise RuntimeError(
            "API 模式需要在 code/.env 中配置 D2_EMBEDDING_API_KEY、"
            "D2_EMBEDDING_BASE_URL 和 D2_EMBEDDING_MODEL；请使用支持 Embedding 的服务。"
        )

    client = OpenAI(
        api_key=Config.D2_EMBEDDING_API_KEY.strip(),
        base_url=Config.D2_EMBEDDING_BASE_URL.strip(),
        timeout=30.0,
        max_retries=0,
    )

    def embed_batch(texts: list[str]) -> list[list[float]]:
        """调用 Embedding API，并按服务端 index 恢复输入顺序。"""
        response = client.embeddings.create(model=Config.D2_EMBEDDING_MODEL.strip(), input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    embed_batch.close = client.close
    return embed_batch


def has_embedding_config() -> bool:
    """检查配置是否齐全；本地模式无需 Key，此函数不会验证网络或下载模型。"""
    provider = Config.D2_EMBEDDING_PROVIDER.strip().lower()
    if provider == "local":
        return True
    return provider == "api" and all(
        Config._is_configured_value(value)
        for value in (
            Config.D2_EMBEDDING_API_KEY,
            Config.D2_EMBEDDING_BASE_URL,
            Config.D2_EMBEDDING_MODEL,
        )
    )


def prepare_strategy_chunks(strategy: str) -> tuple[list[str], list[dict]]:
    """把指定策略的输出统一为 texts、metadatas，供两种检索后端共用。

    fixed：固定窗口与重叠，作为基线。
    semantic：生成句向量，根据语义差异分块，完成后释放向量 API 客户端。
    dynamic：根据结构边界调整重叠。
    parent_child：只返回子块文本，父块正文放入对应的 metadata。

    两个返回列表的长度和顺序必须对应。这样更换检索后端时，
    同一策略仍采用相同的分块算法和参数。
    """
    if strategy == "fixed":
        texts = fixed_overlap_chunk(SAMPLE_LONG_DOC, chunk_size=200, overlap=30)
        return texts, [{"strategy": strategy} for _ in texts]

    if strategy == "semantic":
        embed_fn = build_embed_fn()
        try:
            texts = semantic_chunk(SAMPLE_LONG_DOC, embed_fn)
        finally:
            embed_fn.close()
        return texts, [{"strategy": strategy} for _ in texts]

    if strategy == "dynamic":
        texts = dynamic_overlap_chunk(SAMPLE_LONG_DOC, chunk_size=200)
        return texts, [{"strategy": strategy} for _ in texts]

    if strategy == "parent_child":
        parents = parent_child_chunk(SAMPLE_LONG_DOC, parent_size=500, child_size=120)
        return flatten_parent_child(parents)

    raise ValueError(f"未知策略：{strategy}")


def create_seekdb_retriever(
    strategy: str,
    use_parent_context: bool,
    client_factory=None,
    db_path=None,
) -> SeekdbRetriever:
    """为一种分块策略创建实验集合，写入后返回可查询的包装对象。

    不同策略使用不同集合名，避免片段混在一起；同名旧集合会被重建。
    写入成功后，返回对象的 cleanup 由 evaluate_strategy 在 finally 中调用。
    初始化或写入失败时，本函数负责清理已创建的集合，并继续抛出异常，
    让上层决定是否降级到内存检索。

    client_factory 可注入测试客户端；Server 连接参数由公共模块从 .env 加载。
    """
    if client_factory is None:
        from seekdb_runtime import create_seekdb_client

        client_factory = create_seekdb_client

    # 路径边界：默认锚定 D2 目录，避免嵌入式数据库随 cwd 漂移。
    resolved_path = Path(db_path or DEFAULT_DB_PATH).expanduser().resolve()
    db = client_factory(path=str(resolved_path))
    created = False
    try:
        coll_name = f"d2_chunking_{strategy}"
        if db.has_collection(coll_name):
            db.delete_collection(coll_name)
        collection = db.create_collection(name=coll_name)
        created = True  # 只有本次创建成功，失败分支才有需要清理的集合。
        texts, metadatas = prepare_strategy_chunks(strategy)
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(texts))],
            documents=texts,
            metadatas=metadatas,
        )

        released = False

        def cleanup():
            """释放本次实验的集合和连接，重复调用时不再次操作。"""
            nonlocal released
            if released:
                return
            # 资源清理边界：只删除本轮实验拥有的集合。
            try:
                db.delete_collection(coll_name)
            finally:
                _close_database(db)
                released = True

        retriever = SeekdbRetriever(collection, cleanup)
        retriever.use_parent_context = use_parent_context
        retriever.chunk_count = len(texts)
        retriever.avg_chunk_size = sum(len(t) for t in texts) / max(len(texts), 1)
        return retriever
    except Exception:
        try:
            if created:
                db.delete_collection(coll_name)
        finally:
            _close_database(db)
        raise


def create_memory_retriever(strategy: str) -> InMemoryRetriever:
    """用相同分块数据构造内存检索器，附上报告需要的块数和平均长度。"""
    texts, metadatas = prepare_strategy_chunks(strategy)
    retriever = InMemoryRetriever(texts, metadatas)
    retriever.chunk_count = len(texts)
    retriever.avg_chunk_size = sum(len(t) for t in texts) / max(len(texts), 1)
    return retriever


def recall_at_k(retriever, query: str, keyword: str, use_parent_context: bool, k: int = 3) -> bool:
    """检查单个问题的前 k 个命中上下文是否包含预期关键词。

    返回 bool：本次命中为 True，否则为 False。父子策略会把父块正文也纳入检查。
    这里用关键词命中近似衡量检索效果，没有标注全部相关文档，
    因而不是严格的“找到的相关文档数 / 全部相关文档数”意义上的召回率。
    """
    if use_parent_context:
        docs = retriever.search_with_context(query, k=k)
    else:
        docs = retriever.search(query, k=k)
    return keyword in " ".join(docs)


def evaluate_strategy(
    strategy: str,
    backend: str,
    use_parent_context: bool = False,
    client_factory=None,
    db_path=None,
) -> dict:
    """执行一次完整评测：准备片段 → 建检索器 → 逐题查询 → 汇总 → 清理。

    strategy 决定分块方法，backend 决定向量检索还是内存词项检索，
    use_parent_context 决定命中子块后是否附带父文。
    返回策略名、块数、平均字符数及报告中的 recall_at_3。

    recall_at_3 实际是固定题集中“前三个命中上下文含预期关键词”的问题占比；
    例如 5 题中有 4 题命中，结果为 0.8。父块上下文更长，也更容易包含关键词，
    因此要结合平均片段长度和具体结果分析，不能只看这一项数字。
    """
    cleanup = None
    if backend == "seekdb":
        retriever = create_seekdb_retriever(
            strategy,
            use_parent_context,
            client_factory=client_factory,
            db_path=db_path,
        )
        cleanup = retriever.cleanup
    else:
        retriever = create_memory_retriever(strategy)

    try:
        # bool 转为 0/1 后求和，得到命中问题的数量。
        hits = sum(
            int(
                recall_at_k(
                    retriever,
                    item["query"],
                    item["keyword"],
                    use_parent_context,
                )
            )
            for item in EVAL_QUERIES
        )
        return {
            "strategy": strategy,
            "chunk_count": retriever.chunk_count,
            "avg_chunk_size": retriever.avg_chunk_size,
            "recall_at_3": hits / len(EVAL_QUERIES),
        }
    finally:
        if cleanup is not None:
            cleanup()


def evaluate_with_fallback(
    strategy: str,
    use_parent_context: bool = False,
    client_factory=None,
    db_path=None,
) -> tuple[dict, str]:
    """优先评测 seekdb，失败后用同一分块策略重新评测内存后端。

    返回 (指标字典, 实际后端名称)，后端名称用于提醒结果来自哪种检索方法。
    降级会改变检索依据，日志需要与报告一起阅读；若分块本身也失败，
    内存重试仍会抛出异常，由 main 决定是否跳过语义分块。
    """
    try:
        row = evaluate_strategy(
            strategy,
            "seekdb",
            use_parent_context,
            client_factory=client_factory,
            db_path=db_path,
        )
        return row, "seekdb"
    except Exception as exc:
        print(f">>> seekdb 策略 {strategy} 失败，降级为内存检索：{exc}")
        return (
            evaluate_strategy(strategy, "memory", use_parent_context),
            "memory",
        )


def detect_backend() -> str:
    """只检查 pyseekdb 能否导入；真实连接、写入和查询要到评测时才能验证。"""
    try:
        import pyseekdb
        del pyseekdb
        return "seekdb"
    except ImportError:
        print(">>> seekdb 嵌入式模式不可用，降级为内存词项匹配检索")
        return "memory"


def print_runtime_notes(backend: str):
    """打印实际配置的学习提示，不执行向量化或数据库查询。"""
    print(">>> 实验说明：分块策略与检索后端是两件事")
    if backend == "memory":
        print(">>> 当前 seekdb 不可用，仅检索阶段降级；语义分块使用独立的向量模型")
    if Config.D2_EMBEDDING_PROVIDER.strip().lower() == "local":
        print(">>> 语义分块使用本地向量模型，无需 DeepSeek 或硅基流动 API Key")
    elif has_embedding_config():
        print(">>> 已检测到独立的 D2_EMBEDDING_* API 配置，将尝试运行语义分块")
    else:
        print(">>> D2_EMBEDDING_* 配置不完整或 provider 无效，稍后会跳过语义分块")


def print_report(rows: list[dict], backend: str):
    """展示各策略的块数、平均字符数和关键词命中占比。"""
    print("\n" + "=" * 72)
    print("  D2 高级 Chunking 策略对比实验")
    print(f"  检索后端：{'seekdb 向量检索' if backend == 'seekdb' else '内存词项匹配（降级）'}")
    print("=" * 72)
    print(f"  {'策略':<16} {'块数':>6} {'均长':>8} {'Recall@3':>10}")
    print("  " + "-" * 44)

    labels = {
        "fixed": "固定 overlap",
        "semantic": "语义分块",
        "dynamic": "动态 overlap",
        "parent_child": "父子 chunk",
    }
    for row in rows:
        label = labels.get(row["strategy"], row["strategy"])
        print(
            f"  {label:<16} {row['chunk_count']:>6} "
            f"{row['avg_chunk_size']:>7.0f}字 {row['recall_at_3']:>9.0%}"
        )

    print("\n[分析要点]")
    print("   · 固定 overlap：实现简单，但容易在章节边界处切断语义")
    print("   · 语义分块：按主题跳变切分，适合话题切换频繁的长文")
    print("   · 动态 overlap：在标题/段落边界加大重叠，低成本改善边界召回")
    print("   · 父子 chunk：子块精准检索 + 父块完整上下文，适合多跳推理")
    print("=" * 72 + "\n")


def main(client_factory=None, db_path=None):
    """组织四种策略的对比实验，并在可用范围内继续完成报告。

    先评测不依赖句向量的 fixed、dynamic、parent_child，再评测 semantic。
    语义分块可能因模型下载或 API 配置失败而被跳过，其余策略仍保留结果。
    Server 模式需明确允许重建演示集合，避免把实验误用于其他数据库。
    """
    backend = detect_backend()
    if backend == "seekdb":
        from seekdb_runtime import require_destructive_seekdb_access

        require_destructive_seekdb_access("重建 D2 分块评测集合")
    print_runtime_notes(backend)
    results: list[dict] = []
    used_backends: list[str] = []

    for strategy, use_parent_context in (
        ("fixed", False),
        ("dynamic", False),
        ("parent_child", True),
    ):
        if backend == "seekdb":
            row, used_backend = evaluate_with_fallback(
                strategy,
                use_parent_context,
                client_factory=client_factory,
                db_path=db_path,
            )
        else:
            row = evaluate_strategy(strategy, "memory", use_parent_context)
            used_backend = "memory"
        results.append(row)
        used_backends.append(used_backend)

    try:
        if backend == "seekdb":
            row, used_backend = evaluate_with_fallback(
                "semantic",
                client_factory=client_factory,
                db_path=db_path,
            )
        else:
            row = evaluate_strategy("semantic", "memory")
            used_backend = "memory"
        results.append(row)
        used_backends.append(used_backend)
    except (RuntimeError, OpenAIError, ImportError, OSError, ValueError) as exc:
        print(">>> 跳过语义分块：", exc)

    # 只有每一行都来自 seekdb 才标为 seekdb；有任何降级时统一显示 memory。
    # 这个汇总标签不能描述逐行差异，具体哪种策略降级还需查看前面的日志。
    report_backend = "seekdb" if used_backends and all(
        item == "seekdb" for item in used_backends
    ) else "memory"
    print_report(results, report_backend)


def parse_args(argv=None):
    """解析可选的本地数据库目录；argv 便于测试命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(db_path=parse_args().db_path)
