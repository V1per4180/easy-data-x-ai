"""
D2 高级 Chunking 策略实现

包含三种策略：
  1. semantic_chunk       — 语义分块（基于句子 Embedding 的断点检测）
  2. dynamic_overlap_chunk — 动态 overlap（边界处加大重叠）
  3. parent_child_chunk   — 父子分块（小 chunk 检索，大 chunk 返回上下文）

另提供 fixed_overlap_chunk 作为对比基线。

本模块只处理文本和向量，不负责连接数据库。所有 size、overlap、chars
参数均按字符数计算；Embedding 由调用方传入，便于切换模型和离线测试。
建议先看 fixed_overlap_chunk，再看另外三种策略，最后看辅助函数。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from numbers import Real


@dataclass
class ChildChunk:
    """子块记录：正文用于检索，parent_id 用于追踪所属父块。"""

    text: str
    parent_id: str
    child_index: int


@dataclass
class ParentChunk:
    """父块记录：保存较长上下文，以及在本父块内切出的子块列表。"""

    parent_id: str
    text: str
    # 每个父块获得独立列表，避免多个实例意外共用同一个可变默认值。
    children: list[ChildChunk] = field(default_factory=list)


def split_sentences(text: str) -> list[str]:
    """按中英文结束标点切句并保留标点；这是简化规则，缩写等场景可能误切。"""
    parts = re.split(r"(?<=[。！？.!?])\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def fixed_overlap_chunk(
    text: str,
    chunk_size: int = 200,
    overlap: int = 30,
) -> list[str]:
    """固定长度、固定重叠的分块基线，返回非空文本片段列表。

    用长度为 chunk_size 的窗口依次截取文本，每次前进 chunk_size-overlap。
    重叠能保留切口附近的信息，但也会增加存储量和重复检索的内容。
    本策略只看字符位置，不识别句子、标题或主题边界，可能把一句话切开。
    其他策略可与它比较，以判断额外的复杂度是否带来收益。
    """
    _validate_fixed_chunk_parameters(chunk_size, overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks


def _validate_fixed_chunk_parameters(chunk_size: int, overlap: int) -> None:
    """校验固定分块参数，确保循环步长始终为正数。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算向量夹角的余弦值；越大表示方向越相近，零向量按 0 处理。"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_semantic_breakpoints(
    similarities: list[float],
    percentile: float = 95.0,
) -> set[int]:
    """
    从相邻句相似度中选出差异较大的句间位置，作为分块断点。

    similarities[i] 对应第 i 句与第 i+1 句；1-similarity 是两句的差异度，
    并不是“本次相似度减去上次相似度”。高于或等于分位数阈值时，
    在后一条句子的下标 i+1 处切开。例如断点 2 表示从第 3 句开始新块。

    percentile 越高，通常越倾向于只选择差异较大的边界；相等值可能同时入选。
    差异度全相同时不强行切分，避免内容相近的句子被任意拆开。
    本实现借用分位数阈值思路，是教学简化版本。
    """
    if len(similarities) < 2:
        return set()

    drops = [1.0 - sim for sim in similarities]
    if max(drops) == min(drops):
        return set()

    threshold = _percentile(drops, percentile)

    breakpoints: set[int] = set()
    for idx, drop in enumerate(drops):
        if drop >= threshold:
            breakpoints.add(idx + 1)  # 集合保存“下一块从哪一句开始”，而不是相似度下标。
    return breakpoints


def _percentile(values: list[float], percentile: float) -> float:
    """用线性插值计算 percentile，避免小样本时阈值过度偏向最大值。"""
    if not values:
        raise ValueError("values must not be empty")

    sorted_values = sorted(values)
    bounded = min(max(percentile, 0.0), 100.0)
    position = (len(sorted_values) - 1) * bounded / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]

    ratio = position - lower
    return sorted_values[lower] * (1 - ratio) + sorted_values[upper] * ratio


def semantic_chunk(
    text: str,
    embed_fn,
    percentile: float = 95.0,
    min_chunk_chars: int = 80,
    max_chunk_chars: int = 600,
) -> list[str]:
    """
    语义分块：切句 → 生成句向量 → 比较相邻句 → 选断点 → 调整块长度。

    embed_fn 接收字符串列表，返回数量相同、顺序对应的向量列表；
    本函数不关心向量来自本地模型还是 API，也不调用聊天模型。
    percentile 控制差异度阈值，min_chunk_chars 尽量避免过小片段，
    max_chunk_chars 限制最大长度。返回值仍是文本列表，而不是向量。

    单句无需比较相邻语义，因此直接进入长度处理；多句才会调用 embed_fn。
    “语义”来自所选模型的表示能力，断点不保证总是符合人的主题判断。
    最后的长度处理可能合并语义块或硬切长块；最小长度是软目标，
    不能为了满足最小长度而突破最大长度限制。
    """
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, Real)
        or not math.isfinite(percentile)
        or not 0 <= percentile <= 100
    ):
        raise ValueError("percentile 必须是 0 到 100 之间的有限数值")
    if min_chunk_chars <= 0:
        raise ValueError("min_chunk_chars 必须大于 0")
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars 必须大于 0")
    if min_chunk_chars > max_chunk_chars:
        raise ValueError("min_chunk_chars 不能大于 max_chunk_chars")

    # 第一步：确定参与比较的句子单位。
    sentences = split_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        raw_chunks = sentences
    else:
        # 第二步：批量向量化，随后确认每一句都有且只有一个有效向量。
        embeddings = list(embed_fn(sentences))
        _validate_embeddings(embeddings, len(sentences))
        # 只比较相邻句，不需要构造所有句子两两之间的相似度矩阵。
        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
        breakpoints = _find_semantic_breakpoints(similarities, percentile)

        # 第三步：按断点把连续句子拼回文本；末尾也要收集，防止遗漏最后一段。
        raw_chunks = []
        start = 0
        for idx in range(1, len(sentences) + 1):
            if idx in breakpoints or idx == len(sentences):
                raw_chunks.append("".join(sentences[start:idx]))
                start = idx

    # 第四步：把语义边界与长度约束结合起来，得到便于检索的片段。
    return _merge_small_chunks(raw_chunks, min_chunk_chars, max_chunk_chars)


def _validate_embeddings(embeddings: list[list[float]], expected_count: int) -> None:
    """校验 Embedding 数量、维度及数值有效性。"""
    if len(embeddings) != expected_count:
        raise ValueError(
            f"Embedding 数量应为 {expected_count}，实际为 {len(embeddings)}"
        )

    dimension = len(embeddings[0]) if embeddings else 0
    if dimension == 0:
        raise ValueError("Embedding 向量不能为空")

    for vector in embeddings:
        if len(vector) != dimension:
            raise ValueError("Embedding 向量维度必须一致")
        try:
            finite = all(math.isfinite(value) for value in vector)
        except TypeError as exc:
            raise ValueError("Embedding 向量必须只包含数值") from exc
        if not finite:
            raise ValueError("Embedding 向量必须只包含有限数值")


def _merge_small_chunks(
    chunks: list[str],
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """先硬切超长块，再缓冲和合并小块；尾块可能仍短于 min_chars。"""
    bounded_chunks = [
        chunk[start:start + max_chars]
        for chunk in chunks
        for start in range(0, len(chunk), max_chars)
    ]
    merged: list[str] = []
    buffer = ""  # 暂存还不适合单独输出的小块，等待与后续文本合并。
    for chunk in bounded_chunks:
        candidate = buffer + chunk
        if len(candidate) <= max_chars and len(candidate) < min_chars:
            buffer = candidate
            continue
        if len(candidate) > max_chars:
            merged.append(buffer)
            buffer = chunk
            continue
        merged.append(candidate)
        buffer = ""
    if buffer:
        if merged and len(merged[-1]) + len(buffer) <= max_chars:
            merged[-1] += buffer
        else:
            merged.append(buffer)
    return [c.strip() for c in merged if c.strip()]


def _detect_boundary_positions(text: str) -> set[int]:
    """按 Markdown 规则收集边界的字符偏移；只检测位置，不修改原文。"""
    boundaries: set[int] = set()
    for match in re.finditer(r"\n\n+", text):
        boundaries.add(match.start())
    for match in re.finditer(r"(?m)^#{1,3}\s", text):
        boundaries.add(match.start())
    for match in re.finditer(r"(?m)^[ \t]*\|.+\|[ \t]*$", text):
        boundaries.add(match.start())
    for match in re.finditer(r"(?m)^```", text):
        boundaries.add(match.start())
    return boundaries


def _overlap_near_boundary(
    position: int,
    boundaries: set[int],
    base_overlap: int,
    max_overlap: int,
    window: int = 40,
) -> int:
    """边界附近使用更大 overlap，同质段落内使用较小 overlap。"""
    for boundary in boundaries:
        if abs(position - boundary) <= window:
            return max_overlap
    return base_overlap


def dynamic_overlap_chunk(
    text: str,
    chunk_size: int = 200,
    base_overlap: int = 20,
    max_overlap: int = 60,
) -> list[str]:
    """
    按结构边界调整重叠长度，返回文本片段列表。

    窗口长度 chunk_size 保持不变；变化的是每次窗口前进的步长。
    如果窗口末尾靠近段落、标题、表格或代码块标记，就使用 max_overlap，
    否则使用 base_overlap。重叠越大，下一块起点离当前起点越近。

    这种方法只依赖文本结构规则，无需向量模型。
    它通过重复边界附近的文字降低信息被切散的影响，并不会把切口
    自动移动到标题或完整句子的末尾；也不保证整个表格或代码块不会被切开。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if base_overlap < 0 or max_overlap < 0:
        raise ValueError("overlap 必须大于等于 0")
    if base_overlap > max_overlap:
        raise ValueError("base_overlap 不能大于 max_overlap")
    if max_overlap >= chunk_size:
        raise ValueError("max_overlap 必须小于 chunk_size")

    if not text.strip():
        return []

    boundaries = _detect_boundary_positions(text)
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        # 观察当前窗口的末尾，而不是起点；下一块会重复这个切口附近的文字。
        overlap = _overlap_near_boundary(end, boundaries, base_overlap, max_overlap)
        stride = max(chunk_size - overlap, 1)
        start += stride

    return chunks


def parent_child_chunk(
    text: str,
    parent_size: int = 500,
    child_size: int = 120,
    child_overlap: int = 20,
) -> list[ParentChunk]:
    """
    先切较大的父块，再在每个父块内部切较小的子块。

    parent_size 控制父块长度；child_size 和 child_overlap 控制子块的窗口。
    父块之间在本实现中不重叠，子块之间允许重叠，子块不会跨越所属父块。
    返回带父子关系的 ParentChunk 列表，暂时不写数据库。

    设计目的：用短文本向量提高定位粒度，命中后附带更长上下文。
    d2_5 会只索引子块，并把父块正文放在子块 metadata 中供返回。
    这里的“父块上下文”只保证范围更长，仍是按字符硬切，不保证语义完整。
    """
    _validate_fixed_chunk_parameters(parent_size, 0)
    _validate_fixed_chunk_parameters(child_size, child_overlap)

    parent_texts = fixed_overlap_chunk(text, parent_size, overlap=0)
    parents: list[ParentChunk] = []

    for parent_idx, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_idx}"  # 所有同父子块共享该 ID，用于追踪来源。
        child_texts = fixed_overlap_chunk(parent_text, child_size, child_overlap)
        children = [
            ChildChunk(
                text=child_text,
                parent_id=parent_id,
                child_index=child_idx,
            )
            for child_idx, child_text in enumerate(child_texts)
        ]
        parents.append(
            ParentChunk(
                parent_id=parent_id,
                text=parent_text,
                children=children,
            )
        )
    return parents


def flatten_parent_child(parents: list[ParentChunk]) -> tuple[list[str], list[dict]]:
    """把父子对象展平为等长的 texts、metadatas 两个列表。

    texts 只包含子块正文，作为向量化与检索的内容；相同下标的 metadata
    记录父块 ID、父块全文和子块序号。d2_5 在写入时另外生成文档 ID。
    检索命中后可直接从 metadata 取父文，无需再次查询父块集合。
    这种写法便于教学，但每个子块重复保存父文，会增加存储开销。
    """
    texts: list[str] = []
    metadatas: list[dict] = []
    for parent in parents:
        for child in parent.children:
            texts.append(child.text)
            metadatas.append(
                {
                    "parent_id": parent.parent_id,
                    "parent_text": parent.text,
                    "child_index": child.child_index,
                    "chunk_role": "child",
                }
            )
    return texts, metadatas
