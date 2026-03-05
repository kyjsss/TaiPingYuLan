"""
Processor 模块 - 智能长文重组

本模块负责对 Retriever 返回的候选列表进行“瘦身”和重组。
严格遵循 project.md 2.3 节（Processor：智能长文重组）的技术规格和算法要求。

核心功能：
1. 长度判定：根据阈值决定是否进行切片选择
2. Chunk Selection：从长文中选出最能回答 User Query 的最佳切片
3. Reassembly：重组拼接，保留头尾完整性
4. 重复检测：检测 head/tail 与 text_chunks 的重复

输入：List[RetrievalResult]（包含完整 TextUnit 信息）
输出：List[ProcessedResult]（包含截断后的文本、元数据）
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from schemas import (
    RetrievalResult, ProcessedResult, TextUnit, TextUnitType,
    SourceType, Relationship
)
from config import LONG_TEXT_THRESHOLD, VECTOR_DIM

logger = logging.getLogger(__name__)


class Processor:
    """
    智能长文重组处理器
    
    设计原则：
    1. 纯 CPU 同步计算，无需 Async
    2. 严格遵循 project.md 2.3 节的技术规格
    3. 使用 DataManager 提供的长文切片管理功能
    4. 使用 NumPy 进行向量相似度计算
    5. 复用 schemas.py 中定义的数据模型
    """
    
    def __init__(self, data_manager=None):
        """
        初始化 Processor。
        
        Args:
            data_manager: DataManager 实例，用于访问长文切片管理功能
        """
        self.data_manager = data_manager
        self.long_text_threshold = LONG_TEXT_THRESHOLD
        
        logger.info(f"Processor 初始化完成，长文阈值: {self.long_text_threshold} 字符")
    
    def process(self, retrieval_results: List[RetrievalResult], 
                query_vector: Optional[List[float]] = None,
                query_keywords: Optional[List[str]] = None) -> List[ProcessedResult]:
        """
        处理检索结果，生成处理后的结果列表。
        
        Args:
            retrieval_results: 检索结果列表
            query_vector: 查询向量（4096维），用于 Chunk Selection
            query_keywords: 查询关键词列表（保留参数，当前版本未使用）
            
        Returns:
            List[ProcessedResult]: 处理后的结果列表
        """
        if not retrieval_results:
            logger.warning("检索结果为空，跳过处理")
            return []
        
        processed_results = []
        
        for i, retrieval_result in enumerate(retrieval_results):
            logger.debug(f"处理第 {i+1}/{len(retrieval_results)} 个检索结果")
            
            try:
                processed_result = self._process_single(
                    retrieval_result, query_vector, query_keywords
                )
                processed_results.append(processed_result)
            except Exception as e:
                logger.error(f"处理检索结果 {i} 时出错: {e}")
                # 出错时返回原始文本作为兜底
                processed_result = self._create_fallback_result(retrieval_result)
                processed_results.append(processed_result)
        
        logger.info(f"处理完成: {len(processed_results)} 个结果")
        return processed_results
    
    def _process_single(self, retrieval_result: RetrievalResult,
                       query_vector: Optional[List[float]],
                       query_keywords: Optional[List[str]]) -> ProcessedResult:
        """
        处理单个检索结果。
        
        执行步骤：
        1. 长度判定
        2. Chunk Selection（如果需要）
        3. Reassembly 重组拼接
        4. 重复检测与 head/tail 替换
        """
        text_unit = retrieval_result.text_unit
        
        # 1. 长度判定
        is_long_text = self._is_long_text(text_unit)
        
        # 2. Chunk Selection
        if is_long_text and query_vector is not None:
            best_chunk, chunk_vector, chunk_score = self._select_best_chunk(
                text_unit, query_vector
            )
            text_unit_type = TextUnitType.CHUNK
        else:
            # 短文或没有查询向量，使用完整文本
            best_chunk = text_unit.text
            chunk_score = 1.0
            text_unit_type = TextUnitType.FULL_TEXT
        
        # 3. Reassembly 重组拼接
        processed_text = self._reassemble_text(
            text_unit, best_chunk, is_long_text
        )
        
        # 4. 构建元数据
        metadata = self._build_metadata(
            retrieval_result, is_long_text, chunk_score
        )
        
        # 5. 创建 ProcessedResult
        processed_result = ProcessedResult(
            text_unit_id=text_unit.id,
            processed_text=processed_text,
            original_length=len(text_unit.text),
            processed_length=len(processed_text),
            text_unit_type=text_unit_type,
            highlight_spans=[],  # 高光标注功能已移除
            metadata=metadata
        )
        
        return processed_result
    
    def _is_long_text(self, text_unit: TextUnit) -> bool:
        """
        长度判定逻辑。
        
        根据 project.md 2.3 节：
        - 读取 text 字段长度
        - 若 len(text) <= 1000（阈值可配）：标记为 Full Text，不做剪裁
        - 若 len(text) > 1000：进入 Chunk Selection 流程
        
        Args:
            text_unit: 文本单元
            
        Returns:
            bool: 是否为长文
        """
        text_length = len(text_unit.text)
        is_long = text_length > self.long_text_threshold
        
        logger.debug(f"文本长度: {text_length}, 阈值: {self.long_text_threshold}, 是否为长文: {is_long}")
        return is_long
    
    def _select_best_chunk(self, text_unit: TextUnit, 
                          query_vector: List[float]) -> Tuple[str, List[float], float]:
        """
        Chunk Selection（长文最佳切片选择）。
        
        根据 project.md 2.3 节：
        - 从 text_chunks 列表中选出最能回答 User Query 的一段
        - 利用预计算的 chunk_vectors，计算与 Query 的 Cosine 相似度取 Top 1
        - 如果所有 chunk 的 score 差异极小，不要随机选，直接选第一个 chunk
        
        Args:
            text_unit: 文本单元
            query_vector: 查询向量（4096维）
            
        Returns:
            Tuple[str, List[float], float]: (最佳切片文本, 最佳切片向量, 相似度分数)
        """
        # 检查是否有预计算的切片
        if not text_unit.text_chunks or not text_unit.chunk_vectors:
            logger.warning(f"文本单元 {text_unit.id} 没有预计算的切片，使用完整文本")
            return text_unit.text, text_unit.vector, 1.0
        
        # 确保切片数量与向量数量一致
        n_chunks = len(text_unit.text_chunks)
        n_vectors = len(text_unit.chunk_vectors)
        if n_chunks != n_vectors:
            logger.warning(f"文本单元 {text_unit.id} 切片数量 ({n_chunks}) 与向量数量 ({n_vectors}) 不匹配，使用完整文本")
            return text_unit.text, text_unit.vector, 1.0
        
        # 如果有 DataManager，使用其提供的功能
        if self.data_manager:
            try:
                best_chunk, best_vector, best_score = self.data_manager.get_best_chunk_for_query(
                    text_unit, query_vector
                )
                return best_chunk, best_vector, best_score
            except Exception as e:
                logger.warning(f"使用 DataManager 选择最佳切片失败: {e}")
                # 降级到本地计算
        
        # 本地计算相似度
        return self._select_best_chunk_local(text_unit, query_vector)
    
    def _select_best_chunk_local(self, text_unit: TextUnit,
                                query_vector: List[float]) -> Tuple[str, List[float], float]:
        """
        本地计算最佳切片（当 DataManager 不可用时）。
        
        Args:
            text_unit: 文本单元
            query_vector: 查询向量
            
        Returns:
            Tuple[str, List[float], float]: (最佳切片文本, 最佳切片向量, 相似度分数)
        """
        chunks = text_unit.text_chunks
        chunk_vectors = text_unit.chunk_vectors
        
        # 转换为 numpy 数组
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        best_score = -float('inf')
        best_index = 0
        
        scores = []
        for i, chunk_vector in enumerate(chunk_vectors):
            chunk_np = np.array(chunk_vector, dtype=np.float32).reshape(1, -1)
            # L2 归一化（假设向量已归一化）
            score = np.dot(query_np, chunk_np.T).item()
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_index = i
        
        # 检查所有 chunk 的 score 差异是否极小
        # 定义"差异极小"为最高分与次高分差异小于 0.01
        if len(scores) > 1:
            sorted_scores = sorted(scores, reverse=True)
            if sorted_scores[0] - sorted_scores[1] < 0.01:
                logger.debug(f"所有 chunk 的 score 差异极小，选择第一个 chunk")
                best_index = 0
                best_score = scores[0]
        
        best_chunk = chunks[best_index]
        best_vector = chunk_vectors[best_index]
        
        logger.debug(f"选择切片 {best_index+1}/{len(chunks)}，相似度: {best_score:.4f}")
        return best_chunk, best_vector, best_score
    
    def _reassemble_text(self, text_unit: TextUnit, best_chunk: str, 
                        is_long_text: bool) -> str:
        """
        Reassembly（重组拼接）。
        
        根据 project.md 2.3 节：
        - 构造新的文本展示格式：[Title/Path] {head} ...[检索定位] {best_chunk} ... {tail}
        - 保留 head（前200字）和 tail（后100字）是为了保持史料的"头尾完整性"
        - 如果检测到第一个或最后一个 text_chunks 与 head/tail 有重复，请直接丢弃 head 或 tail，
          用这个 chunk 代替其位置
        
        Args:
            text_unit: 文本单元
            best_chunk: 最佳切片文本
            is_long_text: 是否为长文
            
        Returns:
            str: 重组后的文本
        """
        # 如果不是长文，直接返回完整文本（可能已经过处理）
        if not is_long_text:
            # 添加标题/路径前缀
            prefix = self._get_title_prefix(text_unit)
            return f"{prefix}{text_unit.text}"
        
        # 获取 head 和 tail
        head = text_unit.head if text_unit.head else ""
        tail = text_unit.tail if text_unit.tail else ""
        
        # 检测重复并替换
        head, tail = self._detect_duplicate_and_replace(
            text_unit, head, tail, best_chunk
        )
        
        # 构建重组文本
        # 格式: [Title/Path] {head} ...[检索定位] {best_chunk} ... {tail}
        prefix = self._get_title_prefix(text_unit)
        
        # 如果 head 和 tail 都为空，直接返回最佳切片
        if not head and not tail:
            return f"{prefix}{best_chunk}"
        
        # 如果只有 head
        if head and not tail:
            return f"{prefix}{head} ...[检索定位] {best_chunk}"
        
        # 如果只有 tail
        if not head and tail:
            return f"{prefix}{best_chunk} ... {tail}"
        
        # head 和 tail 都存在
        return f"{prefix}{head} ...[检索定位] {best_chunk} ... {tail}"
    
    def _detect_duplicate_and_replace(self, text_unit: TextUnit, 
                                     head: str, tail: str, 
                                     best_chunk: str) -> Tuple[str, str]:
        """
        检测 head/tail 与 text_chunks 的重复，并进行替换。
        
        逻辑：
        1. 如果第一个 text_chunk 与 head 高度重叠（相似度 > 0.8），则丢弃 head，用该 chunk 代替 head 位置
        2. 如果最后一个 text_chunk 与 tail 高度重叠，则丢弃 tail，用该 chunk 代替 tail 位置
        3. 注意：如果 best_chunk 正好是第一个或最后一个 chunk，需要特殊处理
        
        Args:
            text_unit: 文本单元
            head: 头部文本
            tail: 尾部文本
            best_chunk: 最佳切片文本
            
        Returns:
            Tuple[str, str]: 处理后的 (head, tail)
        """
        if not text_unit.text_chunks:
            return head, tail
        
        # 计算文本相似度（简单重叠比例）
        def overlap_ratio(text1: str, text2: str) -> float:
            if not text1 or not text2:
                return 0.0
            # 简单实现：计算较短的文本在较长文本中的出现比例
            shorter = text1 if len(text1) < len(text2) else text2
            longer = text2 if len(text1) < len(text2) else text1
            if shorter in longer:
                return len(shorter) / len(longer)
            return 0.0
        
        # 检查第一个 chunk
        first_chunk = text_unit.text_chunks[0]
        if head and overlap_ratio(first_chunk, head) > 0.8:
            logger.debug(f"检测到 head 与第一个 chunk 重复，丢弃 head")
            head = ""  # 丢弃 head
        
        # 检查最后一个 chunk
        last_chunk = text_unit.text_chunks[-1]
        if tail and overlap_ratio(last_chunk, tail) > 0.8:
            logger.debug(f"检测到 tail 与最后一个 chunk 重复，丢弃 tail")
            tail = ""  # 丢弃 tail
        
        # 如果 best_chunk 是第一个或最后一个 chunk，且 head/tail 已被丢弃，
        # 则不需要在重组中重复显示，但这里已经通过 head/tail 为空处理了
        
        return head, tail
    
    def _get_title_prefix(self, text_unit: TextUnit) -> str:
        """
        获取标题/路径前缀。
        
        格式: [Title/Path] 
        例如: [天部·日] 
        
        Args:
            text_unit: 文本单元
            
        Returns:
            str: 标题前缀，如果为空则返回空字符串
        """
        if text_unit.prepend_source:
            return text_unit.prepend_source
        
        if text_unit.hierarchy_path:
            # 将层级路径转换为简洁格式
            path_parts = text_unit.hierarchy_path.split(' > ')
            if len(path_parts) >= 2:
                return f"[{path_parts[0]}·{path_parts[-1]}] "
            else:
                return f"[{text_unit.hierarchy_path}] "
        
        return ""
    
    def _generate_highlight_spans(self, text: str, 
                                 query_keywords: List[str]) -> List[Dict[str, Any]]:
        """
        生成高光标注片段（当前版本未使用，返回空列表）。
        
        保留此方法以保持接口兼容性。
        
        Args:
            text: 文本内容
            query_keywords: 查询关键词列表
            
        Returns:
            List[Dict[str, Any]]: 空列表
        """
        return []
    
    def _build_metadata(self, retrieval_result: RetrievalResult,
                       is_long_text: bool, chunk_score: float) -> Dict[str, Any]:
        """
        构建处理结果的元数据。
        
        Args:
            retrieval_result: 检索结果
            is_long_text: 是否为长文
            chunk_score: 切片相似度分数
            
        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = {
            "is_long_text": is_long_text,
            "chunk_score": chunk_score,
            "similarity_score": retrieval_result.similarity_score,
            "source_type": retrieval_result.source_type.value,
            "hierarchy_path": retrieval_result.text_unit.hierarchy_path,
            "community_id": retrieval_result.text_unit.community_id,
        }
        
        # 添加关系信息（如果是邻居扩展）
        if retrieval_result.source_relation:
            metadata["relation_type"] = retrieval_result.source_relation.relation_type
            metadata["relation_weight"] = retrieval_result.source_relation.weight
            metadata["relation_description"] = retrieval_result.source_relation.description
        
        # 添加锚点信息（如果是邻居扩展）
        if retrieval_result.anchor_community_id:
            metadata["anchor_community_id"] = retrieval_result.anchor_community_id
            metadata["anchor_community_title"] = retrieval_result.anchor_community_title
        
        return metadata
    
    def _create_fallback_result(self, retrieval_result: RetrievalResult) -> ProcessedResult:
        """
        创建兜底结果（当处理出错时使用）。
        
        返回原始文本作为处理结果。
        
        Args:
            retrieval_result: 检索结果
            
        Returns:
            ProcessedResult: 兜底处理结果
        """
        text_unit = retrieval_result.text_unit
        
        processed_result = ProcessedResult(
            text_unit_id=text_unit.id,
            processed_text=text_unit.text,
            original_length=len(text_unit.text),
            processed_length=len(text_unit.text),
            text_unit_type=TextUnitType.FULL_TEXT,
            highlight_spans=[],
            metadata={
                "is_fallback": True,
                "error": "处理失败，返回原始文本",
                "similarity_score": retrieval_result.similarity_score,
                "source_type": retrieval_result.source_type.value,
            }
        )
        
        return processed_result