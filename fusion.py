"""
Fusion & Context Builder 模块 - 融合与装填

本模块负责多路召回结果的归并、去重、排序、预算截断，以及最终 Prompt 模板的渲染。
严格遵循 project.md 2.4 节（Fusion & Context Builder：融合与装填）的技术规格和算法要求。

核心功能：
1. 归并与排序 (Merge & Rank)
2. 去重 (Deduplication)
3. 上下文装填 (Context Stuffing)
4. Prompt 模板渲染 (Template Rendering)

输入：List[ProcessedResult]（来自 Processor）
输出：格式化后的 Context 字符串（用于 LLM Prompt）
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum

from schemas import (
    ProcessedResult, SourceType, RetrievalMode,
    MicroRetrievalParams, MacroRetrievalParams
)
from config import GLOBAL_CHAR_BUDGET

logger = logging.getLogger(__name__)


class FusionMode(str, Enum):
    """融合模式枚举"""
    MICRO = "micro"      # 微观考据模式
    MACRO = "macro"      # 宏观综述模式


class Fusion:
    """
    融合与上下文构建器
    
    设计原则：
    1. 纯 CPU 同步计算，无需 Async
    2. 严格遵循 project.md 2.4 节的技术规格
    3. 使用 config.py 中的全局配置（如 GLOBAL_CHAR_BUDGET）
    4. 复用 schemas.py 中定义的数据模型
    """
    
    def __init__(self, mode: FusionMode, system_prompt: str = "", user_query: str = ""):
        """
        初始化 Fusion。
        
        Args:
            mode: 融合模式（MICRO 或 MACRO）
            system_prompt: 系统提示词（用于计算初始字符数）
            user_query: 用户查询（用于计算初始字符数）
        """
        self.mode = mode
        self.system_prompt = system_prompt
        self.user_query = user_query
        
        # 计算初始已用字符数
        self.initial_chars = len(system_prompt) + len(user_query)
        self.global_char_budget = GLOBAL_CHAR_BUDGET
        
        # 统计信息
        self.last_stats: Optional[Dict[str, Any]] = None
        
        logger.info(f"Fusion 初始化完成，模式: {mode.value}，初始字符: {self.initial_chars}，全局预算: {self.global_char_budget}")
    
    def fuse_and_build_context(
        self,
        processed_results: List[ProcessedResult],
        retrieval_params: Optional[Any] = None
    ) -> str:
        """
        融合处理结果并构建上下文。
        
        这是主入口方法，执行以下步骤：
        1. 归并（根据模式合并不同来源的结果）
        2. 去重（基于 text_unit_id）
        3. 排序（相关性优先）
        4. 上下文装填（贪心算法，遵守字符预算）
        5. Prompt 模板渲染
        
        Args:
            processed_results: 处理后的结果列表
            retrieval_params: 检索参数（MicroRetrievalParams 或 MacroRetrievalParams）
            
        Returns:
            str: 格式化后的 Context 字符串
        """
        if not processed_results:
            logger.warning("处理结果为空，返回空上下文")
            self.last_stats = {
                "input_count": 0,
                "merged_count": 0,
                "deduplicated_count": 0,
                "sorted_count": 0,
                "stuffed_count": 0,
                "total_chars": 0,
                "budget_usage_percent": 0.0,
                "exceeded_budget": False,
            }
            return ""
        
        logger.info(f"开始融合 {len(processed_results)} 个处理结果")
        
        # 1. 归并（根据模式合并不同来源的结果）
        merged_results = self._merge_results(processed_results, retrieval_params)
        
        # 2. 去重（基于 text_unit_id）
        deduplicated_results = self._deduplicate_results(merged_results)
        
        # 3. 排序（相关性优先）
        sorted_results = self._sort_results(deduplicated_results)
        
        # 4. 上下文装填（贪心算法，遵守字符预算）
        stuffed_results = self._stuff_context(sorted_results)
        
        # 5. Prompt 模板渲染
        context_str = self._render_template(stuffed_results)
        
        # 计算统计信息
        total_chars = len(context_str)
        budget_usage_percent = (total_chars / self.global_char_budget) * 100 if self.global_char_budget > 0 else 0
        exceeded_budget = total_chars > self.global_char_budget
        
        self.last_stats = {
            "input_count": len(processed_results),
            "merged_count": len(merged_results),
            "deduplicated_count": len(deduplicated_results),
            "sorted_count": len(sorted_results),
            "stuffed_count": len(stuffed_results),
            "total_chars": total_chars,
            "budget_usage_percent": budget_usage_percent,
            "exceeded_budget": exceeded_budget,
        }
        
        logger.info(f"上下文构建完成，最终字符数: {total_chars}")
        return context_str
    
    def _merge_results(
        self,
        processed_results: List[ProcessedResult],
        retrieval_params: Optional[Any]
    ) -> List[ProcessedResult]:
        """
        归并处理结果。
        
        根据模式执行不同的合并策略：
        - Mode A (Micro): 合并 Anchors + Neighbors + Fallback 三个阶段的结果
        - Mode B (Macro): 合并 Findings (Claims) + Evidence (TextUnits) + Global Supplement
        
        当前版本：简单返回所有结果（后续批次实现具体逻辑）
        
        Args:
            processed_results: 处理后的结果列表
            retrieval_params: 检索参数
            
        Returns:
            List[ProcessedResult]: 归并后的结果列表
        """
        logger.info(f"归并阶段（模式 {self.mode.value}），输入 {len(processed_results)} 个结果")
        
        # 简单实现：直接返回所有结果
        # 后续批次将实现具体的合并策略
        return processed_results
    
    def _deduplicate_results(
        self,
        merged_results: List[ProcessedResult]
    ) -> List[ProcessedResult]:
        """
        去重处理结果。
        
        基于 text_unit_id 进行硬去重。
        冲突解决：如果同一条记录同时出现在"Anchor"和"Neighbor"中，
        保留"Anchor"的身份，但叠加"Neighbor"的元数据。
        
        算法：
        1. 构建字典 id -> 结果列表（相同ID的所有结果）
        2. 对每个ID，根据来源优先级选择最佳结果
        3. 如果存在多个来源，合并元数据（叠加Neighbor的元数据到Anchor）
        
        来源优先级：Anchor > Neighbor > Fallback > 其他
        
        Args:
            merged_results: 归并后的结果列表
            
        Returns:
            List[ProcessedResult]: 去重后的结果列表
        """
        logger.info(f"去重阶段，输入 {len(merged_results)} 个结果")
        
        # 1. 按ID分组
        id_to_results: Dict[str, List[ProcessedResult]] = {}
        for result in merged_results:
            id_to_results.setdefault(result.text_unit_id, []).append(result)
        
        # 2. 对每个ID选择最佳结果并处理冲突
        deduplicated = []
        for text_unit_id, results in id_to_results.items():
            if len(results) == 1:
                # 无冲突，直接添加
                deduplicated.append(results[0])
                continue
            
            # 有冲突，记录日志
            logger.debug(f"发现重复记录: {text_unit_id}，共 {len(results)} 个版本")
            
            # 确定优先级最高的来源
            # 来源类型映射到优先级数值（越小优先级越高）
            source_priority = {
                SourceType.ANCHOR.value: 1,
                SourceType.NEIGHBOR.value: 2,
                SourceType.FALLBACK.value: 3,
                SourceType.FINDING.value: 4,
                SourceType.EVIDENCE.value: 5,
                SourceType.GLOBAL_SUPPLEMENT.value: 6,
            }
            
            # 找出优先级最高的结果
            best_result = None
            best_priority = float('inf')
            for result in results:
                source_type = result.metadata.get("source_type", "")
                priority = source_priority.get(source_type, 99)
                if priority < best_priority:
                    best_priority = priority
                    best_result = result
            
            if best_result is None:
                # 没有有效来源类型，选择第一个
                best_result = results[0]
                logger.warning(f"无法确定最佳来源，选择第一个: {text_unit_id}")
            
            # 3. 合并元数据（如果需要）
            # 如果存在Anchor和Neighbor，将Neighbor的关系信息叠加到Anchor
            if len(results) > 1:
                best_result = self._merge_metadata_for_duplicates(results, best_result)
            
            deduplicated.append(best_result)
        
        logger.info(f"去重后剩余 {len(deduplicated)} 个结果")
        return deduplicated
    
    def _merge_metadata_for_duplicates(
        self,
        all_results: List[ProcessedResult],
        best_result: ProcessedResult
    ) -> ProcessedResult:
        """
        合并重复记录的元数据。
        
        冲突解决逻辑：
        - 如果最佳结果是 Anchor，但存在 Neighbor 版本，将 Neighbor 的关系信息叠加到 Anchor
        - 叠加方式：在 metadata 中添加额外的字段，如 "additional_relations"
        - 保留最佳结果的身份（source_type 不变）
        
        Args:
            all_results: 相同ID的所有结果
            best_result: 选中的最佳结果
            
        Returns:
            ProcessedResult: 合并元数据后的结果（可能是新实例）
        """
        # 如果只有一个结果，无需合并
        if len(all_results) <= 1:
            return best_result
        
        # 确定最佳结果的来源类型
        best_source = best_result.metadata.get("source_type", "")
        
        # 收集其他来源的关系信息
        additional_relations = []
        for result in all_results:
            if result.text_unit_id != best_result.text_unit_id:
                continue  # 安全校验
            
            source_type = result.metadata.get("source_type", "")
            # 如果是 Neighbor 来源，且有关系信息，则收集
            if source_type == SourceType.NEIGHBOR.value:
                relation_info = {
                    "relation_type": result.metadata.get("relation_type"),
                    "relation_weight": result.metadata.get("relation_weight"),
                    "relation_description": result.metadata.get("relation_description"),
                    "anchor_community_id": result.metadata.get("anchor_community_id"),
                    "anchor_community_title": result.metadata.get("anchor_community_title"),
                }
                # 过滤空值
                relation_info = {k: v for k, v in relation_info.items() if v is not None}
                if relation_info:
                    additional_relations.append(relation_info)
        
        # 如果有额外的关系信息，添加到 metadata
        if additional_relations:
            # 创建新的 metadata 字典（避免修改原始对象）
            new_metadata = best_result.metadata.copy()
            new_metadata["additional_relations"] = additional_relations
            new_metadata["has_multiple_sources"] = True
            
            # 创建新的 ProcessedResult（不可变对象，需要重新创建）
            merged_result = ProcessedResult(
                text_unit_id=best_result.text_unit_id,
                processed_text=best_result.processed_text,
                original_length=best_result.original_length,
                processed_length=best_result.processed_length,
                text_unit_type=best_result.text_unit_type,
                highlight_spans=best_result.highlight_spans.copy() if best_result.highlight_spans else [],
                metadata=new_metadata
            )
            logger.debug(f"合并元数据完成，添加了 {len(additional_relations)} 个额外关系")
            return merged_result
        
        return best_result
    
    def _sort_results(
        self,
        deduplicated_results: List[ProcessedResult]
    ) -> List[ProcessedResult]:
        """
        排序处理结果。
        
        原则：相关性优先。
        主键：vector_score (降序)。
        (Mode A 特修)：对于 Neighbor 来源的记录，可以使用 weight * 0.9 进行降权排序。
        
        当前版本：简单按 similarity_score 排序（后续批次实现降权逻辑）
        
        Args:
            deduplicated_results: 去重后的结果列表
            
        Returns:
            List[ProcessedResult]: 排序后的结果列表
        """
        logger.info(f"排序阶段，输入 {len(deduplicated_results)} 个结果")
        
        # 简单实现：按 similarity_score 降序排序
        # 从 metadata 中提取 similarity_score
        def get_score(result: ProcessedResult) -> float:
            score = result.metadata.get("similarity_score", 0.0)
            # 如果是 Neighbor 来源，应用降权（后续批次实现）
            source_type = result.metadata.get("source_type")
            if source_type == SourceType.NEIGHBOR.value:
                weight = result.metadata.get("relation_weight", 1.0)
                score = score * weight * 0.9  # 降权因子
            return score
        
        sorted_results = sorted(
            deduplicated_results,
            key=get_score,
            reverse=True
        )
        
        logger.info(f"排序完成，最高分: {get_score(sorted_results[0]) if sorted_results else 0.0}")
        return sorted_results
    
    def _stuff_context(
        self,
        sorted_results: List[ProcessedResult]
    ) -> List[ProcessedResult]:
        """
        上下文装填（贪心算法）。
        
        严格遵守 GLOBAL_CHAR_BUDGET。
        初始化：current_chars = len(system_prompt) + len(user_query)
        贪心装填：
        - 遍历有序列表
        - 计算 entry_length = len(processed_text) + len(metadata_string)
        - 若 current_chars + entry_length <= GLOBAL_CHAR_BUDGET：加入 Context List
        - 否则：Break（保证高分优先）
        
        实现细节：
        1. 使用 _calculate_entry_length 精确计算每个条目在最终模板中的字符数
        2. 考虑换行符和分隔符
        3. 严格检查预算，确保最终 Prompt 总长度不超过 GLOBAL_CHAR_BUDGET
        
        Args:
            sorted_results: 排序后的结果列表
            
        Returns:
            List[ProcessedResult]: 装填后的结果列表（符合预算）
        """
        logger.info(f"上下文装填阶段，输入 {len(sorted_results)} 个结果")
        
        current_chars = self.initial_chars
        stuffed_results = []
        
        for i, result in enumerate(sorted_results):
            # 计算条目在最终模板中的长度
            entry_length = self._calculate_entry_length(result)
            
            # 检查是否超出预算
            if current_chars + entry_length <= self.global_char_budget:
                stuffed_results.append(result)
                current_chars += entry_length
                logger.debug(f"加入结果 {result.text_unit_id}，当前字符: {current_chars}")
            else:
                remaining = len(sorted_results) - i
                logger.info(f"预算不足，停止装填（剩余 {remaining} 个结果）")
                break
        
        logger.info(f"装填完成，选中 {len(stuffed_results)} 个结果，总字符: {current_chars}")
        return stuffed_results
    
    def _calculate_entry_length(self, result: ProcessedResult) -> int:
        """
        计算单个结果在最终模板中的字符长度。
        
        根据模式使用不同的计算逻辑：
        - Micro 模式：按照微观考据模板计算
        - Macro 模式：按照宏观综述模板计算
        
        Args:
            result: 处理结果
            
        Returns:
            int: 预估字符长度
        """
        if self.mode == FusionMode.MICRO:
            return self._calculate_micro_entry_length(result)
        else:
            return self._calculate_macro_entry_length(result)
    
    def _calculate_micro_entry_length(self, result: ProcessedResult) -> int:
        """
        计算 Micro 模式下的条目长度。
        """
        # 获取元数据
        prepend_source = result.metadata.get("prepend_source", "")
        hierarchy_path = result.metadata.get("hierarchy_path", "")
        relation_tag = self._generate_relation_tag(result)
        
        # 构建元数据字符串行（与 _format_micro_metadata_string 一致）
        metadata_parts = []
        if hierarchy_path:
            metadata_parts.append(f"[Source: {hierarchy_path}]")
        if prepend_source:
            metadata_parts.append(f"[{prepend_source}]")
        if relation_tag:
            metadata_parts.append(relation_tag)
        metadata_str = " ".join(metadata_parts)
        
        # 计算总长度
        total_length = 0
        if prepend_source:
            total_length += len(prepend_source) + 1  # 加上换行符
        total_length += len(metadata_str) + 1  # 元数据行 + 换行符
        total_length += len(result.processed_text) + 1  # 内容行 + 换行符
        # 结果之间可能保留空行，空行长度在装填时统一计算
        return total_length
    
    def _calculate_macro_entry_length(self, result: ProcessedResult) -> int:
        """
        计算 Macro 模式下的条目长度。
        
        简化估计：根据 source_type 决定在模板中的大致长度。
        - 如果是 Claim (Finding/Evidence)：需要更多字符（包含标题、元数据等）
        - 如果是 Supplementary：简单列表项
        
        当前版本：统一估计为 processed_text 长度 + 固定开销
        """
        source_type = result.metadata.get("source_type", "")
        base_length = len(result.processed_text)
        
        # 固定开销：换行符、编号、前缀等
        if source_type in [SourceType.FINDING.value, SourceType.EVIDENCE.value]:
            # Claim 格式：### Claim X: ... 加上多行元数据
            overhead = 100  # 估计值
        else:
            # Supplementary 格式：X. [ID]: "..."
            overhead = 50
        
        return base_length + overhead
    
    def _format_metadata_string(self, result: ProcessedResult) -> str:
        """
        格式化元数据字符串。
        
        用于计算 entry_length 和后续模板渲染。
        根据模式生成不同的元数据字符串：
        - Mode A (Micro): [Source: {hierarchy_path}][prepend_source] {relation_tag}
        - Mode B (Macro): 根据结构化模板生成
        
        Args:
            result: 处理结果
            
        Returns:
            str: 格式化后的元数据字符串
        """
        if self.mode == FusionMode.MICRO:
            return self._format_micro_metadata_string(result)
        else:
            return self._format_macro_metadata_string(result)
    
    def _format_micro_metadata_string(self, result: ProcessedResult) -> str:
        """
        格式化微观考据模式的元数据字符串。
        
        格式: [Source: {hierarchy_path}][prepend_source] {relation_tag}
        
        Args:
            result: 处理结果
            
        Returns:
            str: 格式化后的元数据字符串
        """
        # 获取层级路径
        hierarchy_path = result.metadata.get("hierarchy_path", "")
        # 获取 prepend_source（如果 metadata 中没有，则从 processed_text 提取？）
        # 暂时使用空字符串
        prepend_source = ""
        
        # 生成 relation_tag
        relation_tag = self._generate_relation_tag(result)
        
        # 构建字符串
        parts = []
        if hierarchy_path:
            parts.append(f"[Source: {hierarchy_path}]")
        if prepend_source:
            parts.append(f"[{prepend_source}]")
        if relation_tag:
            parts.append(relation_tag)
        
        metadata_str = " ".join(parts)
        return metadata_str
    
    def _format_macro_metadata_string(self, result: ProcessedResult) -> str:
        """
        格式化宏观综述模式的元数据字符串。
        
        根据结构化模板生成，暂时返回空字符串。
        
        Args:
            result: 处理结果
            
        Returns:
            str: 格式化后的元数据字符串
        """
        # 后续批次实现
        return ""
    
    def _generate_relation_tag(self, result: ProcessedResult) -> str:
        """
        生成关系标签 (relation_tag)。
        
        逻辑：
        - 若是直接命中 (Anchor/Fallback)：为空字符串
        - 若是图谱扩展 (Neighbor)：
          `[关联证据: 通过 "{relation_type}" 关联到 "{anchor_community_title}"]`
        
        Args:
            result: 处理结果
            
        Returns:
            str: 关系标签字符串
        """
        source_type = result.metadata.get("source_type", "")
        
        # 如果是 Neighbor 来源，生成关系标签
        if source_type == SourceType.NEIGHBOR.value:
            relation_type = result.metadata.get("relation_type", "")
            anchor_title = result.metadata.get("anchor_community_title", "")
            
            if relation_type and anchor_title:
                return f"[关联证据: 通过 \"{relation_type}\" 关联到 \"{anchor_title}\"]"
            elif relation_type:
                return f"[关联证据: 通过 \"{relation_type}\" 关联]"
            else:
                return "[关联证据]"
        
        # 如果是其他来源，返回空字符串
        return ""
    
    def _render_template(
        self,
        stuffed_results: List[ProcessedResult]
    ) -> str:
        """
        Prompt 模板渲染。
        
        根据模式渲染不同的模板：
        - Mode A (Micro): 使用微观考据模板
        - Mode B (Macro): 使用宏观综述模板
        
        当前版本：简单渲染（后续批次实现具体模板）
        
        Args:
            stuffed_results: 装填后的结果列表
            
        Returns:
            str: 格式化后的 Context 字符串
        """
        logger.info(f"模板渲染阶段，输入 {len(stuffed_results)} 个结果")
        
        if self.mode == FusionMode.MICRO:
            return self._render_micro_template(stuffed_results)
        else:
            return self._render_macro_template(stuffed_results)
    
    def _render_micro_template(
        self,
        stuffed_results: List[ProcessedResult]
    ) -> str:
        """
        渲染微观考据模板。
        
        Template for Mode A (Micro Search):
        ```
        # 史料证据列表 (Historical Evidence)
        
        prepend_source
        [Source: {hierarchy_path}][prepend_source] {relation_tag}
        {content}
        ```
        
        实现细节：
        1. 每个结果按照上述模板格式化
        2. 结果之间用空行分隔
        3. 确保最终字符串长度不超过预算（已在装填阶段保证）
        
        Args:
            stuffed_results: 装填后的结果列表
            
        Returns:
            str: 格式化后的微观考据 Context
        """
        if not stuffed_results:
            return ""
        
        lines = ["# 史料证据列表 (Historical Evidence)", ""]
        
        for i, result in enumerate(stuffed_results):
            # 获取元数据
            prepend_source = result.metadata.get("prepend_source", "")
            hierarchy_path = result.metadata.get("hierarchy_path", "")
            relation_tag = self._generate_relation_tag(result)
            
            # 构建元数据字符串（与 _format_micro_metadata_string 一致）
            metadata_parts = []
            if hierarchy_path:
                metadata_parts.append(f"[Source: {hierarchy_path}]")
            if prepend_source:
                metadata_parts.append(f"[{prepend_source}]")
            if relation_tag:
                metadata_parts.append(relation_tag)
            metadata_str = " ".join(metadata_parts)
            
            # 获取内容（processed_text 可能已包含前缀，但为了模板清晰，我们使用原始内容）
            # processed_text 可能已包含 prepend_source 前缀，此处按模板重新组织
            content = result.processed_text
            
            # 按照模板添加行
            if prepend_source:
                lines.append(prepend_source)
            lines.append(metadata_str)
            lines.append(content)
            
            # 结果之间添加空行（除非是最后一个）
            if i < len(stuffed_results) - 1:
                lines.append("")
        
        return "\n".join(lines)
    
    def _render_macro_template(
        self,
        stuffed_results: List[ProcessedResult]
    ) -> str:
        """
        渲染宏观综述模板。
        
        复用 2.2 节最后定义的结构化模板（区分 Claims 和 Supplementary），
        确保 Findings 部分包含 Logical Bridge 的描述。
        
        模板结构：
        # Macro Overview (宏观综述)
        
        ## I. Structural Logic & Claims (逻辑与观点)
        基于图谱分析，发现以下核心观点及其关联：
        
        ### Claim 1: [Finding Summary] (Score: 0.88)
        - **Context**: From Community [天部·日]
        - **Logical Bridge**: [天部] --[RespondsTo]--> [人事部] (Reference)
        - **Supporting Evidence**:
          1. [TextUnit ID]: "..."
          2. [TextUnit ID]: "..."
        
        ## II. Supplementary Historical Records (补充史料)
        原典记载：
        1. [TextUnit ID]: "..."
        2. [TextUnit ID]: "..."
        
        实现细节：
        1. 根据 source_type 将结果分类为 Claims (Findings) 和 Supplementary
        2. 为每个 Claim 提取 Logical Bridge 信息
        3. 为每个 Claim 关联 Supporting Evidence
        4. 将所有 Supplementary 结果列为简单列表
        
        当前版本：简化实现，将所有结果视为 Supplementary（后续批次完善）
        
        Args:
            stuffed_results: 装填后的结果列表
            
        Returns:
            str: 格式化后的宏观综述 Context
        """
        if not stuffed_results:
            return ""
        
        # 分类结果
        claims = []
        supplementary = []
        for result in stuffed_results:
            source_type = result.metadata.get("source_type", "")
            if source_type in [SourceType.FINDING.value, SourceType.EVIDENCE.value]:
                claims.append(result)
            else:
                supplementary.append(result)
        
        lines = ["# Macro Overview (宏观综述)", ""]
        
        # 第一部分：Claims（如果有）
        if claims:
            lines.append("## I. Structural Logic & Claims (逻辑与观点)")
            lines.append("基于图谱分析，发现以下核心观点及其关联：")
            lines.append("")
            
            for i, claim in enumerate(claims, 1):
                # 提取 Finding 摘要（从 metadata 或 processed_text）
                finding_summary = claim.metadata.get("finding_summary", claim.processed_text[:100] + "...")
                similarity_score = claim.metadata.get("similarity_score", 0.0)
                
                # 提取 Logical Bridge
                logical_bridge = claim.metadata.get("logical_bridge", "")
                community_info = claim.metadata.get("community_id", "")
                
                lines.append(f"### Claim {i}: {finding_summary} (Score: {similarity_score:.2f})")
                if community_info:
                    lines.append(f"- **Context**: From Community [{community_info}]")
                if logical_bridge:
                    lines.append(f"- **Logical Bridge**: {logical_bridge}")
                lines.append("- **Supporting Evidence**:")
                lines.append("  1. [证据占位]")
                lines.append("")
        
        # 第二部分：Supplementary Historical Records
        lines.append("## II. Supplementary Historical Records (补充史料)")
        lines.append("原典记载：")
        lines.append("")
        
        for i, supp in enumerate(supplementary, 1):
            # 提取文本预览
            preview = supp.processed_text[:150] + "..." if len(supp.processed_text) > 150 else supp.processed_text
            lines.append(f"{i}. [{supp.text_unit_id}]: \"{preview}\"")
        
        return "\n".join(lines)


# ==================== 辅助函数 ====================

def create_fusion(
    mode: FusionMode,
    system_prompt: str = "",
    user_query: str = ""
) -> Fusion:
    """
    创建 Fusion 实例的工厂函数。
    
    Args:
        mode: 融合模式
        system_prompt: 系统提示词
        user_query: 用户查询
        
    Returns:
        Fusion: Fusion 实例
    """
    return Fusion(mode, system_prompt, user_query)


if __name__ == "__main__":
    # 简单测试
    import sys
    logging.basicConfig(level=logging.INFO)
    
    print("Fusion 模块测试")
    
    # 创建模拟的 ProcessedResult
    from schemas import ProcessedResult, TextUnitType
    
    mock_results = [
        ProcessedResult(
            text_unit_id="test_001",
            processed_text="测试文本 1",
            original_length=100,
            processed_length=50,
            text_unit_type=TextUnitType.FULL_TEXT,
            highlight_spans=[],
            metadata={"similarity_score": 0.9, "source_type": "anchor"}
        ),
        ProcessedResult(
            text_unit_id="test_002",
            processed_text="测试文本 2",
            original_length=200,
            processed_length=100,
            text_unit_type=TextUnitType.FULL_TEXT,
            highlight_spans=[],
            metadata={"similarity_score": 0.8, "source_type": "neighbor"}
        ),
    ]
    
    # 创建 Fusion 实例
    fusion = Fusion(FusionMode.MICRO, system_prompt="系统提示", user_query="用户查询")
    
    # 执行融合
    context = fusion.fuse_and_build_context(mock_results)
    
    print(f"生成的上下文长度: {len(context)}")
    print("上下文内容:")
    print(context)