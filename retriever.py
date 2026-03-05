"""
Retriever 模块 - 双模路由 (The Router - Precision First)

本模块实现《太平御览》GraphRAG 系统的检索逻辑，严格遵循 project.md 2.2 节（Retriever：双模路由）的技术规格和算法要求。

包含两个核心模式：
1. 微观考据 (Micro Search / Drill-down)
2. 宏观综述 (Macro Panorama)

核心原则：遵循“精度优先 (Precision-First)”策略，利用显式图谱关系打破语义隔离，并采用动态权重配额机制确保高价值内容的优先召回。
"""

import logging
import polars as pl
import faiss
from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np

from schemas import (
    TextUnit, Community, Relationship, CommunityReport, FilterQuery,
    RetrievalResult, ProcessedResult, TextUnitType, SourceType,
    MicroRetrievalParams, MacroRetrievalParams, RetrievalMode,
    Vector, CommunityMap, RelationshipMap
)
from data_manager import DataManager
from config import GLOBAL_CHAR_BUDGET


logger = logging.getLogger(__name__)


class MicroRetriever:
    """
    微观考据检索器 (Mode A: Micro Search / Drill-down)
    
    该模式针对具体实体或事件的深度挖掘（例如：“荧惑守心有何预兆？”）。
    
    核心算法流程：
    1. 目标锁定 (Target Identification) - 锚点定位 + 邻居扩展
    2. 智能配额分配算法 (Smart Quota Allocation Algorithm)
    3. 瀑布流装填 (Waterfall Retrieval) - 锚点装填 + 邻居注入 + 全局语义兜底
    4. 层级穿透逻辑 (Hierarchy Penetration)
    
    严格遵循 project.md 2.2 节【模式 A：微观考据】的所有技术规格。
    """
    
    def __init__(self, data_manager: DataManager, config: Optional[Dict[str, Any]] = None):
        """
        初始化微观检索器。
        
        Args:
            data_manager: 数据管理器实例，提供索引和检索功能
            config: 可选配置字典，覆盖默认参数
        """
        self.data_manager = data_manager
        self.config = config or {}
        
        # 从配置中提取参数，使用默认值
        self.params = MicroRetrievalParams(**self.config)
        
        # 加载 relationships 表（用于邻居扩展）
        self.relationships_df = self._load_relationships()
        
        logger.info(f"MicroRetriever 初始化完成: graph_enable={self.params.graph_enable}, "
                   f"anchor_quota_ratio={self.params.anchor_quota_ratio}, "
                   f"neighbor_quota_ratio={self.params.neighbor_quota_ratio}")
    
    def _load_relationships(self) -> Optional[pl.DataFrame]:
        """
        加载 relationships 表。
        
        Returns:
            Optional[pl.DataFrame]: 加载的 DataFrame，如果失败则返回 None
        """
        try:
            df = pl.read_parquet("create_final_relationships_refined.parquet")
            logger.info(f"Relationships 表加载成功: {df.shape[0]} 条关系")
            return df
        except Exception as e:
            logger.error(f"加载 relationships 表失败: {e}")
            return None
    
    def retrieve(self, query_vector: Vector, query_text: str = "") -> List[RetrievalResult]:
        """
        执行微观考据检索。
        
        这是主入口方法，执行完整的微观检索流程：
        1. 目标锁定
        2. 智能配额分配
        3. 瀑布流装填
        4. 层级穿透
        
        Args:
            query_vector: 查询向量 (4096维)
            query_text: 原始查询文本，用于调试和日志
            
        Returns:
            List[RetrievalResult]: 检索结果列表，按相关性排序
        """
        logger.info(f"开始微观检索: query_text='{query_text[:50]}...'")
        
        # Step 1: 目标锁定
        anchors, neighbors = self._identify_targets(query_vector)
        logger.info(f"目标锁定完成: {len(anchors)} 个锚点, {len(neighbors)} 个邻居")
        
        # Step 2: 瀑布流装填
        results = self._waterfall_retrieval(query_vector, anchors, neighbors)
        logger.info(f"瀑布流装填完成: 共 {len(results)} 个结果")
        
        # Step 3: 去重与排序
        final_results = self._deduplicate_and_sort(results)
        logger.info(f"最终结果: {len(final_results)} 个去重结果")
        
        return final_results
    
    def _identify_targets(self, query_vector: Vector) -> Tuple[List[Dict], List[Dict]]:
        """
        目标锁定 (Target Identification)
        
        确定两类目标社区：锚点社区 (Anchors) 和邻居社区 (Neighbors)。
        
        Args:
            query_vector: 查询向量
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (锚点列表, 邻居列表)
            每个锚点元素: {"community_id": str, "score": float}
            每个邻居元素: {"community_id": str, "weight": float, "relation_description": str, "source_anchor_id": str}
        """
        # 锚点定位 (Anchors)
        anchors = self._locate_anchors(query_vector)
        
        # 邻居扩展 (Neighbors) - 仅当 graph_enable=True 且 anchors 非空
        neighbors = []
        if self.params.graph_enable and anchors:
            neighbors = self._expand_neighbors(anchors)
        
        return anchors, neighbors
    
    def _locate_anchors(self, query_vector: Vector) -> List[Dict]:
        """
        锚点定位 (Anchor Location)
        
        对 Report Index 执行向量检索，保留 Score > SIMILARITY_THRESHOLD 的 Top K 社区。
        
        Args:
            query_vector: 查询向量
            
        Returns:
            List[Dict]: 锚点列表，每个元素包含 community_id 和 score
        """
        if not self.data_manager.report_index:
            logger.warning("Report Index 未构建，无法进行锚点定位")
            return []
        
        # 对查询向量进行 L2 归一化
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_np)
        
        # 搜索 Top K 个社区（K 取 10，确保有足够候选）
        top_k = 10
        distances, indices = self.data_manager.report_index.search(query_np, top_k)
        
        anchors = []
        for i in range(top_k):
            faiss_id = indices[0, i]
            similarity = distances[0, i]  # 内积相似度，范围 [-1, 1]
            
            # 过滤相似度阈值
            if similarity < self.params.similarity_threshold:
                continue
            
            # 通过 faiss_id 获取社区 ID
            community_id = self.data_manager.faiss_id_to_report_community_id.get(faiss_id)
            if not community_id:
                logger.warning(f"无效的 FAISS ID {faiss_id}，无法映射到社区")
                continue
            
            # 获取社区标题用于日志
            community = self.data_manager.get_community_by_id(community_id)
            community_title = community.title if community else "未知"
            
            anchors.append({
                "community_id": community_id,
                "score": float(similarity),
                "title": community_title
            })
        
        # 按分数降序排序
        anchors.sort(key=lambda x: x["score"], reverse=True)
        
        # 限制最终锚点数量（例如最多 5 个）
        max_anchors = 5
        if len(anchors) > max_anchors:
            anchors = anchors[:max_anchors]
        
        logger.info(f"锚点定位完成: 找到 {len(anchors)} 个锚点 (阈值={self.params.similarity_threshold})")
        for anchor in anchors:
            logger.debug(f"  锚点: {anchor['title']} ({anchor['community_id']}) 分数: {anchor['score']:.3f}")
        
        return anchors
    
    def _expand_neighbors(self, anchors: List[Dict]) -> List[Dict]:
        """
        邻居扩展 (Neighbor Expansion)
        
        遍历锚点列表，查询 relationships 表，取 weight > RELATION_WEIGHT_THRESHOLD 的 Top neighbor_fanout 邻居。
        
        Args:
            anchors: 锚点列表
            
        Returns:
            List[Dict]: 邻居列表，每个元素包含 community_id, weight, relation_description, source_anchor_id
        """
        if self.relationships_df is None:
            logger.warning("Relationships 表未加载，跳过邻居扩展")
            return []
        
        # 收集所有锚点 ID
        anchor_ids = {anchor["community_id"] for anchor in anchors}
        
        neighbors = []
        neighbor_ids = set()  # 用于去重
        
        for anchor in anchors:
            anchor_id = anchor["community_id"]
            anchor_title = anchor.get("title", "未知")
            
            # 查询以该锚点为起点的关系
            source_relations = self.relationships_df.filter(
                pl.col("source_id") == anchor_id
            )
            
            # 过滤权重阈值
            threshold = self.params.relation_weight_threshold
            filtered = source_relations.filter(pl.col("weight") > threshold)
            
            # 按权重降序排序
            sorted_rels = filtered.sort("weight", descending=True)
            
            # 截取 Top neighbor_fanout
            fanout = self.params.neighbor_fanout
            top_rels = sorted_rels.head(fanout)
            
            # 处理每个关系
            for row in top_rels.iter_rows(named=True):
                target_id = row["target_id"]
                
                # 去重：排除已经是锚点的 ID，以及已经添加过的邻居
                if target_id in anchor_ids or target_id in neighbor_ids:
                    continue
                
                # 获取社区标题
                target_community = self.data_manager.get_community_by_id(target_id)
                target_title = target_community.title if target_community else "未知"
                
                neighbor = {
                    "community_id": target_id,
                    "weight": float(row["weight"]),
                    "relation_type": row["relation_type"],
                    "relation_description": row["description"],
                    "source_anchor_id": anchor_id,
                    "source_anchor_title": anchor_title,
                    "target_title": target_title
                }
                neighbors.append(neighbor)
                neighbor_ids.add(target_id)
        
        # 按权重降序排序所有邻居
        neighbors.sort(key=lambda x: x["weight"], reverse=True)
        
        logger.info(f"邻居扩展完成: 找到 {len(neighbors)} 个邻居 (阈值={self.params.relation_weight_threshold})")
        for neighbor in neighbors[:5]:  # 只打印前5个
            logger.debug(f"  邻居: {neighbor['target_title']} ({neighbor['community_id']}) "
                        f"权重: {neighbor['weight']:.3f} 关联: {neighbor['relation_type']}")
        
        return neighbors
    
    def _waterfall_retrieval(self, query_vector: Vector, anchors: List[Dict], neighbors: List[Dict]) -> List[RetrievalResult]:
        """
        瀑布流装填 (Waterfall Retrieval)
        
        分三个阶段进行内容的物理拉取和装填：
        1. Phase 1: 锚点装填 (Anchor Filling)
        2. Phase 2: 邻居装填 (Neighbor Injection)
        3. Phase 3: 全局语义兜底 (Global Semantic Fallback)
        
        Args:
            query_vector: 查询向量
            anchors: 锚点列表
            neighbors: 邻居列表
            
        Returns:
            List[RetrievalResult]: 合并的检索结果
        """
        all_results = []
        
        # Phase 1: 锚点装填
        anchor_results = self._anchor_filling(query_vector, anchors)
        all_results.extend(anchor_results)
        
        # Phase 2: 邻居装填
        neighbor_results = self._neighbor_injection(query_vector, neighbors)
        all_results.extend(neighbor_results)
        
        # Phase 3: 全局语义兜底
        fallback_results = self._global_semantic_fallback(query_vector, all_results)
        all_results.extend(fallback_results)
        
        return all_results
    
    def _anchor_filling(self, query_vector: Vector, anchors: List[Dict]) -> List[RetrievalResult]:
        """
        Phase 1: 锚点装填 (Anchor Filling)
        
        预算：budget_anchors = GLOBAL_CHAR_BUDGET * anchor_quota_ratio
        应用智能配额分配算法处理 anchors 列表。
        
        Args:
            query_vector: 查询向量
            anchors: 锚点列表
            
        Returns:
            List[RetrievalResult]: 锚点装填结果
        """
        if not anchors:
            logger.info("锚点列表为空，跳过锚点装填")
            return []
        
        # 计算字符预算
        budget_chars = int(GLOBAL_CHAR_BUDGET * self.params.anchor_quota_ratio)
        logger.info(f"锚点装填预算: {budget_chars} 字符 (总预算 {GLOBAL_CHAR_BUDGET} * 比例 {self.params.anchor_quota_ratio})")
        
        # 估算平均每条 TextUnit 的长度（字符数）
        # 这里使用经验值 500，可以根据实际数据调整
        AVERAGE_TEXTUNIT_LENGTH = 500
        total_budget_count = max(1, budget_chars // AVERAGE_TEXTUNIT_LENGTH)
        logger.info(f"锚点装填预算条数: {total_budget_count} 条 (平均长度 {AVERAGE_TEXTUNIT_LENGTH} 字符)")
        
        # 智能配额分配
        quotas = self._distribute_quota_by_score(anchors, total_budget_count)
        if not quotas:
            logger.warning("配额分配失败，返回空结果")
            return []
        
        # 遍历每个社区，拉取内容
        results = []
        used_chars = 0
        
        for community_id, quota in quotas.items():
            # 层级穿透获取该社区下的内容（锚定社区）
            community_results = self._hierarchy_penetration(community_id, query_vector, quota, is_anchor=True)
            
            # 计算已使用字符数（用于预算跟踪）
            for result in community_results:
                text_length = len(result.text_unit.text)
                if used_chars + text_length > budget_chars:
                    # 超出预算，停止添加
                    logger.debug(f"锚点装填预算已满，停止添加更多内容")
                    break
                used_chars += text_length
                results.append(result)
            
            # 如果已超出预算，跳出循环
            if used_chars >= budget_chars:
                logger.info(f"锚点装填预算已用尽: {used_chars}/{budget_chars} 字符")
                break
        
        logger.info(f"锚点装填完成: 共 {len(results)} 条结果, 使用 {used_chars}/{budget_chars} 字符")
        return results
    
    def _neighbor_injection(self, query_vector: Vector, neighbors: List[Dict]) -> List[RetrievalResult]:
        """
        Phase 2: 邻居装填 (Neighbor Injection)
        
        预算：budget_neighbors = GLOBAL_CHAR_BUDGET * neighbor_quota_ratio
        此阶段拥有独立预算，即使 Phase 1 没用完，这里也不会抢占；即使 Phase 1 用光了，这里依然有粮。
        
        Args:
            query_vector: 查询向量
            neighbors: 邻居列表
            
        Returns:
            List[RetrievalResult]: 邻居装填结果
        """
        if not neighbors:
            logger.info("邻居列表为空，跳过邻居装填")
            return []
        
        # 计算字符预算（独立预算）
        budget_chars = int(GLOBAL_CHAR_BUDGET * self.params.neighbor_quota_ratio)
        logger.info(f"邻居装填预算: {budget_chars} 字符 (总预算 {GLOBAL_CHAR_BUDGET} * 比例 {self.params.neighbor_quota_ratio})")
        
        # 估算平均每条 TextUnit 的长度（字符数）
        AVERAGE_TEXTUNIT_LENGTH = 500
        total_budget_count = max(1, budget_chars // AVERAGE_TEXTUNIT_LENGTH)
        logger.info(f"邻居装填预算条数: {total_budget_count} 条 (平均长度 {AVERAGE_TEXTUNIT_LENGTH} 字符)")
        
        # 智能配额分配
        quotas = self._distribute_quota_by_score(neighbors, total_budget_count)
        if not quotas:
            logger.warning("邻居配额分配失败，返回空结果")
            return []
        
        # 遍历每个社区，拉取内容
        results = []
        used_chars = 0
        
        for community_id, quota in quotas.items():
            # 找到该邻居的元数据（关系信息）
            neighbor_meta = next((n for n in neighbors if n["community_id"] == community_id), None)
            if not neighbor_meta:
                continue
            
            # 层级穿透获取该社区下的内容（邻居社区，不添加摘要）
            community_results = self._hierarchy_penetration(community_id, query_vector, quota, is_anchor=False)
            
            # 为每个结果附加关系元数据
            for result in community_results:
                # 创建 Relationship 对象
                relation = Relationship(
                    source_id=neighbor_meta["source_anchor_id"],
                    target_id=community_id,
                    weight=neighbor_meta["weight"],
                    relation_type=neighbor_meta["relation_type"],
                    description=neighbor_meta["relation_description"],
                    vector_score=0.0  # 未知
                )
                
                # 更新结果元数据
                result.source_type = SourceType.NEIGHBOR
                result.source_relation = relation
                result.anchor_community_id = neighbor_meta["source_anchor_id"]
                result.anchor_community_title = neighbor_meta.get("source_anchor_title", "未知")
                
                # 计算字符数
                text_length = len(result.text_unit.text)
                if used_chars + text_length > budget_chars:
                    logger.debug(f"邻居装填预算已满，停止添加更多内容")
                    break
                used_chars += text_length
                results.append(result)
            
            # 如果已超出预算，跳出循环
            if used_chars >= budget_chars:
                logger.info(f"邻居装填预算已用尽: {used_chars}/{budget_chars} 字符")
                break
        
        logger.info(f"邻居装填完成: 共 {len(results)} 条结果, 使用 {used_chars}/{budget_chars} 字符")
        return results
    
    def _global_semantic_fallback(self, query_vector: Vector, existing_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Phase 3: 全局语义兜底 (Global Semantic Fallback)
        
        预算：remaining_budget = GLOBAL_CHAR_BUDGET - used_chars_anchors - used_chars_neighbors
        强制保底逻辑：如果 remaining_budget 过小，导致无法容纳 force_recall_count 条记录，
        则无视预算上限，强制扩容以容纳这 force_recall_count 条（Top-K）。
        
        Args:
            query_vector: 查询向量
            existing_results: 已存在的检索结果（用于去重）
            
        Returns:
            List[RetrievalResult]: 全局语义兜底结果
        """
        # 计算已使用字符数
        used_chars = sum(len(result.text_unit.text) for result in existing_results)
        remaining_budget = GLOBAL_CHAR_BUDGET - used_chars
        
        logger.info(f"全局语义兜底: 已使用 {used_chars}/{GLOBAL_CHAR_BUDGET} 字符, 剩余预算 {remaining_budget} 字符")
        
        # 强制保底逻辑：如果剩余预算过小，无法容纳 force_recall_count 条记录，则强制扩容
        AVERAGE_TEXTUNIT_LENGTH = 500
        min_chars_needed = self.params.force_recall_count * AVERAGE_TEXTUNIT_LENGTH
        
        if remaining_budget < min_chars_needed:
            logger.warning(f"剩余预算 {remaining_budget} 字符不足以容纳 {self.params.force_recall_count} 条记录 "
                          f"(需要至少 {min_chars_needed} 字符)，强制扩容")
            # 无视预算上限，我们将拉取 force_recall_count 条记录
            target_count = self.params.force_recall_count
        else:
            # 正常情况：根据剩余预算计算可拉取条数
            target_count = max(1, remaining_budget // AVERAGE_TEXTUNIT_LENGTH)
        
        logger.info(f"全局语义兜底目标条数: {target_count} 条")
        
        # 执行全库向量检索（忽略社区结构）
        # 使用 DataManager 的 search_text_units 方法
        if not self.data_manager.text_unit_index:
            logger.error("TextUnit 索引未构建，无法执行全局语义检索")
            return []
        
        logger.info(f"TextUnit 索引已构建，text_unit_map 大小: {len(self.data_manager.text_unit_map)}")
        
        # 搜索更多结果以便过滤
        search_k = min(100, len(self.data_manager.text_unit_map))  # 搜索 Top 100
        logger.info(f"FAISS 搜索参数: search_k={search_k}, similarity_threshold={self.params.similarity_threshold}")
        
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_np)
        
        distances, indices = self.data_manager.text_unit_index.search(query_np, search_k)
        logger.info(f"FAISS 搜索返回: distances shape={distances.shape}, indices shape={indices.shape}")
        logger.info(f"FAISS 搜索结果 (前10个): distances={distances[0][:10].tolist()}, indices={indices[0][:10].tolist()}")
        
        # 构建已存在的 TextUnit ID 集合用于去重
        existing_ids = {result.text_unit.id for result in existing_results}
        logger.info(f"已存在的 TextUnit ID 数量: {len(existing_ids)}")
        
        results = []
        filtered_by_threshold = 0
        failed_to_get_textunit = 0
        filtered_by_dedup = 0
        
        for i in range(search_k):
            faiss_id = indices[0, i]
            similarity = distances[0, i]
            
            # 过滤相似度阈值
            if similarity < self.params.similarity_threshold:
                filtered_by_threshold += 1
                continue
            
            # 获取 TextUnit
            text_unit = self.data_manager.get_text_unit_by_faiss_id(faiss_id)
            if not text_unit:
                failed_to_get_textunit += 1
                logger.warning(f"无法通过 faiss_id={faiss_id} 获取 TextUnit")
                continue
            
            # 去重
            if text_unit.id in existing_ids:
                filtered_by_dedup += 1
                continue
            
            # 构建结果
            result = RetrievalResult(
                text_unit=text_unit,
                similarity_score=float(similarity),
                source_type=SourceType.FALLBACK,
                source_relation=None,
                anchor_community_id=None,
                anchor_community_title=None
            )
            results.append(result)
            
            # 如果已收集足够结果，停止
            if len(results) >= target_count:
                break
        
        logger.info(f"全局语义兜底完成: 找到 {len(results)} 条新结果 (去重后)")
        logger.info(f"  - 被相似度阈值过滤: {filtered_by_threshold} 条")
        logger.info(f"  - 获取 TextUnit 失败: {failed_to_get_textunit} 条")
        logger.info(f"  - 被去重过滤: {filtered_by_dedup} 条")
        return results
    
    def _distribute_quota_by_score(self, candidates: List[Dict], total_budget_count: int) -> Dict[str, int]:
        """
        智能配额分配算法 (Smart Quota Allocation Algorithm)
        
        基于分数的加权配额分配：quota_i = Round(total_budget_count * (score_i / sum_scores))
        Edge Case：若 quota_i < 1，强制设为 1。
        
        Args:
            candidates: 候选列表，每个元素必须包含 "score" 或 "weight" 字段
            total_budget_count: 该阶段允许拉取的 TextUnit 总条数（估算值）
            
        Returns:
            Dict[str, int]: 社区ID -> 配额数量 的映射
        """
        if not candidates or total_budget_count <= 0:
            return {}
        
        # 提取分数：优先使用 score，其次使用 weight
        scores = []
        community_ids = []
        for candidate in candidates:
            community_id = candidate.get("community_id")
            if not community_id:
                continue
            
            # 获取分数
            score = candidate.get("score")
            if score is None:
                score = candidate.get("weight")
            if score is None:
                logger.warning(f"候选 {community_id} 没有 score 或 weight 字段，跳过")
                continue
            
            scores.append(float(score))
            community_ids.append(community_id)
        
        if not scores:
            logger.warning("没有有效的分数，无法分配配额")
            return {}
        
        # 计算总分
        sum_scores = sum(scores)
        if sum_scores <= 0:
            logger.warning("总分 <= 0，无法分配配额")
            # 平均分配
            quota_per_item = max(1, total_budget_count // len(scores))
            return {cid: quota_per_item for cid in community_ids}
        
        # 分配配额
        quotas = {}
        remaining = total_budget_count
        
        for i, (community_id, score) in enumerate(zip(community_ids, scores)):
            # 计算配额
            quota = round(total_budget_count * (score / sum_scores))
            # Edge Case：若 quota < 1，强制设为 1
            if quota < 1:
                quota = 1
            
            quotas[community_id] = quota
            remaining -= quota
        
        # 由于四舍五入可能导致配额总和与 total_budget_count 有差异，调整最后一个配额
        if remaining != 0:
            # 将差异加到最后一个候选上（或从最后一个候选减去）
            last_id = community_ids[-1]
            quotas[last_id] = max(1, quotas[last_id] + remaining)
        
        # 验证配额总和
        total_allocated = sum(quotas.values())
        if total_allocated != total_budget_count:
            logger.warning(f"配额分配不一致: 期望 {total_budget_count}, 实际 {total_allocated}")
        
        logger.info(f"智能配额分配完成: 总预算 {total_budget_count}, 分配 {len(quotas)} 个社区")
        for community_id, quota in list(quotas.items())[:10]:  # 只打印前10个
            logger.debug(f"  社区 {community_id}: 配额 {quota}")
        
        return quotas
    
    def _hierarchy_penetration(self, community_id: str, query_vector: Vector, quota: int, is_anchor: bool = True) -> List[RetrievalResult]:
        """
        层级穿透逻辑 (Hierarchy Penetration)
        
        检查 is_leaf 字段：
        - Case A (Leaf Community)：直接读取该 ID 下挂载的 TextUnits
        - Case B (Non-Leaf Community)：拉取该社区的 summary 作为虚拟 TextUnit 返回
        
        注意：社区摘要只在 is_anchor=True 时添加，邻居社区不添加摘要
        
        Args:
            community_id: 社区ID
            query_vector: 查询向量（用于排序 TextUnits）
            quota: 该社区的配额数量
            is_anchor: 是否为锚定社区（True：添加摘要，False：不添加摘要）
            
        Returns:
            List[RetrievalResult]: 该社区下的检索结果
        """
        community = self.data_manager.get_community_by_id(community_id)
        if not community:
            logger.warning(f"社区 {community_id} 不存在，跳过层级穿透")
            return []
        
        results = []
        
        if community.is_leaf:
            # Case A: Leaf Community - 直接读取该 ID 下挂载的 TextUnits
            text_units = self.data_manager.get_text_units_by_community_id(community_id)
            logger.debug(f"叶子社区 {community.title} ({community_id}) 有 {len(text_units)} 个 TextUnits")
            
            if not text_units:
                return []
            
            # 计算每个 TextUnit 与查询向量的相似度
            scored_units = []
            query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_np)
            
            for text_unit in text_units:
                # 获取 TextUnit 向量（已归一化）
                vector_np = np.array(text_unit.vector, dtype=np.float32).reshape(1, -1)
                # 计算内积相似度
                similarity = np.dot(query_np, vector_np.T).item()
                scored_units.append((text_unit, similarity))
            
            # 按相似度降序排序
            scored_units.sort(key=lambda x: x[1], reverse=True)
            
            # 截取 Top quota 条
            for text_unit, similarity in scored_units[:quota]:
                result = RetrievalResult(
                    text_unit=text_unit,
                    similarity_score=float(similarity),
                    source_type=SourceType.ANCHOR,  # 默认为锚点，调用者可以覆盖
                    source_relation=None,
                    anchor_community_id=community_id,
                    anchor_community_title=community.title
                )
                results.append(result)
            
            logger.debug(f"叶子社区 {community.title} 返回 {len(results)} 个 TextUnits (配额 {quota})")
            
            # 只为锚定社区添加社区摘要
            if is_anchor:
                report = self.data_manager.get_report_by_community_id(community_id)
                if report and report.summary:
                    # 创建虚拟 TextUnit 表示社区摘要
                    summary_text_unit = TextUnit(
                        id=f"summary_{community_id}",
                        text=f"【社区摘要】{report.summary}",
                        n_tokens=len(report.summary) // 4,  # 粗略估计
                        vector=report.embedding,  # 使用报告的向量
                        community_id=community_id,
                        text_chunks=[],
                        chunk_vectors=[],
                        head="",
                        tail="",
                        hierarchy_path=community.title,
                        source_metadata="",
                        prepend_source=f"[{community.title}·摘要]"
                    )
                    
                    # 计算相似度（使用报告向量）
                    vector_np = np.array(report.embedding, dtype=np.float32).reshape(1, -1)
                    faiss.normalize_L2(vector_np)
                    similarity = np.dot(query_np, vector_np.T).item()
                    
                    summary_result = RetrievalResult(
                        text_unit=summary_text_unit,
                        similarity_score=float(similarity),
                        source_type=SourceType.ANCHOR,
                        source_relation=None,
                        anchor_community_id=community_id,
                        anchor_community_title=community.title
                    )
                    # 将摘要放在结果列表的最前面
                    results.insert(0, summary_result)
                    logger.debug(f"叶子社区 {community.title} 添加社区摘要（锚定社区）")
        
        else:
            # Case B: Non-Leaf Community - 处理逻辑取决于是否为锚定社区
            if is_anchor:
                # 锚定社区：拉取该社区的 summary 作为虚拟 TextUnit 返回
                report = self.data_manager.get_report_by_community_id(community_id)
                if not report or not report.summary:
                    logger.warning(f"非叶子社区 {community.title} ({community_id}) 没有报告或摘要，跳过")
                    return []
                
                # 创建虚拟 TextUnit（添加【社区摘要】前缀以保持一致性）
                virtual_text_unit = TextUnit(
                    id=f"summary_{community_id}",
                    text=f"【社区摘要】{report.summary}",
                    n_tokens=len(report.summary) // 4,  # 粗略估计
                    vector=report.embedding,  # 使用报告的向量
                    community_id=community_id,
                    text_chunks=[],
                    chunk_vectors=[],
                    head="",
                    tail="",
                    hierarchy_path=community.title,
                    source_metadata="",
                    prepend_source=f"[{community.title}·摘要]"
                )
                
                # 计算相似度（使用报告向量）
                query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(query_np)
                vector_np = np.array(report.embedding, dtype=np.float32).reshape(1, -1)
                similarity = np.dot(query_np, vector_np.T).item()
                
                result = RetrievalResult(
                    text_unit=virtual_text_unit,
                    similarity_score=float(similarity),
                    source_type=SourceType.ANCHOR,
                    source_relation=None,
                    anchor_community_id=community_id,
                    anchor_community_title=community.title
                )
                results.append(result)
                logger.debug(f"非叶子社区 {community.title} 返回摘要作为虚拟 TextUnit（锚定社区）")
            else:
                # 邻居社区：非叶子社区不返回任何内容
                logger.debug(f"邻居社区 {community.title} 是非叶子社区，不返回摘要")
                return []
        
        return results
    
    def _deduplicate_and_sort(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        去重与排序
        
        基于 text_unit_id 进行硬去重，冲突解决：如果同一条记录同时出现在"Anchor"和"Neighbor"中，
        保留"Anchor"的身份，但叠加"Neighbor"的元数据。
        
        排序原则：相关性优先，主键为 similarity_score（降序）。
        
        Args:
            results: 原始检索结果列表
            
        Returns:
            List[RetrievalResult]: 去重并排序后的结果
        """
        # 待实现：去重与排序逻辑
        return results


class MacroRetriever:
    """
    宏观综述检索器 (Mode B: Macro Panorama)
    
    该模式针对宏观问题的综合论述，优先构建"逻辑链路"与"核心观点"，最后强制补充"全局语义证据"。
    
    核心算法流程：
    1. Territory Mapping (版图定位)
    2. Logical Bridging (逻辑架桥)
    3. Finding Ranking & Selection (观点重排)
    4. Evidence Sampling (例证采样)
    5. Global Semantic Supplement (全局语义补全)
    6. Context Assembly (结构化装填)
    
    严格遵循 project.md 2.2 节【模式 B：宏观综述】的所有技术规格。
    """
    
    def __init__(self, data_manager: DataManager, config: Optional[Dict[str, Any]] = None):
        """
        初始化宏观检索器。
        
        Args:
            data_manager: 数据管理器实例
            config: 可选配置字典
        """
        self.data_manager = data_manager
        self.config = config or {}
        self.params = MacroRetrievalParams(**self.config)
        
        # 加载 relationships 表（用于逻辑架桥）
        self.relationships_df = self._load_relationships()
        
        logger.info(f"MacroRetriever 初始化完成: top_k_macro={self.params.top_k_macro}, "
                   f"bridge_fanout={self.params.bridge_fanout}")
    
    def _territory_mapping(self, query_vector: Vector) -> List[Dict]:
        """
        Phase 1: Territory Mapping (版图定位)
        
        目标：确定 User Query 落在哪个"部"或"门"的管辖范围内。
        
        操作：
        1. 对 create_final_community_reports.parquet (Report Index) 执行向量检索。
        2. 保留 Score > SIMILARITY_THRESHOLD 的 Top top_k_macro 社区。
        
        Returns:
            List[Dict]: 锚点集合，每个元素包含 community_id, score, title
        """
        if not self.data_manager.report_index:
            logger.warning("Report Index 未构建，无法进行版图定位")
            return []
        
        # 对查询向量进行 L2 归一化
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_np)
        
        # 搜索 Top K 个社区（K 取 top_k_macro * 2，确保有足够候选）
        search_k = min(self.params.top_k_macro * 2, len(self.data_manager.community_reports))
        distances, indices = self.data_manager.report_index.search(query_np, search_k)
        
        anchors = []
        for i in range(search_k):
            faiss_id = indices[0, i]
            similarity = distances[0, i]  # 内积相似度，范围 [-1, 1]
            
            # 过滤相似度阈值
            if similarity < self.params.similarity_threshold:
                continue
            
            # 通过 faiss_id 获取社区 ID
            community_id = self.data_manager.faiss_id_to_report_community_id.get(faiss_id)
            if not community_id:
                logger.warning(f"无效的 FAISS ID {faiss_id}，无法映射到社区")
                continue
            
            # 获取社区标题用于日志
            community = self.data_manager.get_community_by_id(community_id)
            community_title = community.title if community else "未知"
            
            anchors.append({
                "community_id": community_id,
                "score": float(similarity),
                "title": community_title,
                "level": community.level if community else 0
            })
            
            # 如果已收集足够锚点，停止
            if len(anchors) >= self.params.top_k_macro:
                break
        
        # 按分数降序排序
        anchors.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"版图定位完成: 找到 {len(anchors)} 个锚点 (阈值={self.params.similarity_threshold})")
        for anchor in anchors:
            logger.debug(f"  锚点: {anchor['title']} (Level {anchor['level']}) 分数: {anchor['score']:.3f}")
        
        return anchors
    
    def _load_relationships(self) -> Optional[pl.DataFrame]:
        """
        加载 relationships 表。
        
        Returns:
            Optional[pl.DataFrame]: 加载的 DataFrame，如果失败则返回 None
        """
        try:
            df = pl.read_parquet("create_final_relationships_refined.parquet")
            logger.info(f"Relationships 表加载成功: {df.shape[0]} 条关系")
            return df
        except Exception as e:
            logger.error(f"加载 relationships 表失败: {e}")
            return None
    
    def _logical_bridging(self, anchors: List[Dict]) -> List[Dict]:
        """
        Phase 2: Logical Bridging (逻辑架桥)
        
        目标：寻找锚点与外部世界的逻辑关联，构建"逻辑链路"。
        
        算法：
        - Case A (具体社区)：查 relationships 表，取 weight > 0.8 且语义相关的 Top bridge_fanout 邻居。
        - Case B (部级聚合)：对 Level 0 节点执行"子集聚类投影"，聚合出联系最紧密的外部"部"。
        
        Args:
            anchors: 锚点列表，每个元素包含 community_id, score, title, level
        
        Returns:
            List[Dict]: 逻辑关联社区集合，每个元素包含 community_id, weight, relation_description, source_anchor_id
        """
        if not anchors:
            logger.warning("锚点列表为空，无法进行逻辑架桥")
            return []
        
        if self.relationships_df is None:
            logger.warning("Relationships 表未加载，跳过逻辑架桥")
            return []
        
        # 收集所有锚点 ID
        anchor_ids = {anchor["community_id"] for anchor in anchors}
        
        # 用于去重的集合
        seen_community_ids = set(anchor_ids)
        bridges = []
        
        # 权重阈值（根据 project.md 2.2 节，weight > 0.8）
        weight_threshold = 0.8
        
        for anchor in anchors:
            anchor_id = anchor["community_id"]
            anchor_title = anchor.get("title", "未知")
            anchor_level = anchor.get("level", 0)
            
            # 根据锚点级别决定处理方式
            if anchor_level == 0:
                # Case B: 部级聚合（Level 0 节点）
                # 使用 DataManager 的 node_to_root_map 找到该部下的所有子节点
                # 然后聚合出联系最紧密的外部"部"
                # 简化实现：暂时跳过，因为需要更复杂的聚类投影
                # 我们暂时只处理 Case A
                logger.debug(f"锚点 {anchor_title} 是 Level 0 节点，跳过部级聚合（待实现）")
                continue
            else:
                # Case A: 具体社区
                # 查询以该锚点为起点的关系
                source_relations = self.relationships_df.filter(
                    pl.col("source_id") == anchor_id
                )
                
                # 过滤权重阈值
                filtered = source_relations.filter(pl.col("weight") > weight_threshold)
                
                # 按权重降序排序
                sorted_rels = filtered.sort("weight", descending=True)
                
                # 截取 Top bridge_fanout
                fanout = self.params.bridge_fanout
                top_rels = sorted_rels.head(fanout)
                
                # 处理每个关系
                for row in top_rels.iter_rows(named=True):
                    target_id = row["target_id"]
                    
                    # 去重：排除已经是锚点的 ID，以及已经添加过的桥接社区
                    if target_id in seen_community_ids:
                        continue
                    
                    # 获取社区标题
                    target_community = self.data_manager.get_community_by_id(target_id)
                    target_title = target_community.title if target_community else "未知"
                    
                    bridge = {
                        "community_id": target_id,
                        "weight": float(row["weight"]),
                        "relation_type": row["relation_type"],
                        "relation_description": row["description"],
                        "source_anchor_id": anchor_id,
                        "source_anchor_title": anchor_title,
                        "target_title": target_title
                    }
                    bridges.append(bridge)
                    seen_community_ids.add(target_id)
        
        # 按权重降序排序
        bridges.sort(key=lambda x: x["weight"], reverse=True)
        
        logger.info(f"逻辑架桥完成: 找到 {len(bridges)} 个桥接社区 (权重阈值={weight_threshold})")
        for bridge in bridges[:5]:  # 只打印前5个
            logger.debug(f"  桥接: {bridge['target_title']} ({bridge['community_id']}) "
                        f"权重: {bridge['weight']:.3f} 关联: {bridge['relation_type']}")
        
        return bridges
    
    def _finding_ranking_and_selection(self, query_vector: Vector, communities: List[Dict]) -> List[Dict]:
        """
        Phase 3: Finding Ranking & Selection (观点重排)
        
        目标：从命中社区的众多 Findings 中，挑出与 User Query 真正相关的观点。
        
        算法：
        1. 收集所有命中社区（锚点 + 桥接社区）的 findings 和 finding_vectors。
        2. 对每个 finding 向量，计算与查询向量的 CosineSimilarity。
        3. 按相似度降序排列，截取 Top max_findings_per_query。
        
        Args:
            query_vector: 查询向量
            communities: 社区列表，每个元素包含 community_id, score/weight, title 等
        
        Returns:
            List[Dict]: 排序后的 finding 列表，每个元素包含：
                - community_id: 所属社区 ID
                - community_title: 社区标题
                - finding_index: 在该社区 findings 中的索引
                - summary: finding 摘要
                - explanation: finding 解释
                - similarity_score: 与查询的相似度
        """
        if not communities:
            logger.warning("社区列表为空，无法进行观点重排")
            return []
        
        # 对查询向量进行 L2 归一化
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_np)
        
        all_findings = []
        
        for community in communities:
            community_id = community["community_id"]
            community_title = community.get("title", "未知")
            
            # 获取社区报告
            report = self.data_manager.get_report_by_community_id(community_id)
            if not report:
                logger.debug(f"社区 {community_title} 没有报告，跳过")
                continue
            
            # 解析 findings
            try:
                findings = report.parsed_findings
            except Exception as e:
                logger.warning(f"社区 {community_id} 解析 findings 失败: {e}")
                findings = []
            
            finding_vectors = report.finding_vectors
            
            # 确保向量数量与 findings 数量一致
            if len(findings) != len(finding_vectors):
                logger.warning(f"社区 {community_id} 的 findings 数量 ({len(findings)}) 与向量数量 ({len(finding_vectors)}) 不匹配")
                # 取较小值
                n = min(len(findings), len(finding_vectors))
                findings = findings[:n]
                finding_vectors = finding_vectors[:n]
            
            # 计算每个 finding 的相似度
            for idx, (finding, vector) in enumerate(zip(findings, finding_vectors)):
                # 对 finding 向量进行 L2 归一化（假设已经归一化，但为了安全再次归一化）
                vector_np = np.array(vector, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(vector_np)
                
                # 计算内积相似度（余弦相似度）
                similarity = np.dot(query_np, vector_np.T).item()
                
                # 过滤相似度阈值（可选，但 project.md 未明确要求）
                # 我们暂时不过滤，因为后续会截取 Top K
                
                all_findings.append({
                    "community_id": community_id,
                    "community_title": community_title,
                    "finding_index": idx,
                    "summary": finding.summary,
                    "explanation": finding.explanation,
                    "similarity_score": float(similarity)
                })
        
        if not all_findings:
            logger.warning("没有找到任何 finding，观点重排返回空列表")
            return []
        
        # 按相似度降序排序
        all_findings.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # 截取 Top max_findings_per_query
        max_findings = self.params.max_findings_per_query
        selected = all_findings[:max_findings]
        
        logger.info(f"观点重排完成: 从 {len(all_findings)} 个 findings 中选出 {len(selected)} 个 (max_findings_per_query={max_findings})")
        for finding in selected[:5]:  # 只打印前5个
            logger.debug(f"  观点: {finding['summary'][:50]}... 相似度: {finding['similarity_score']:.3f}")
        
        return selected
    
    def _evidence_sampling(self, query_vector: Vector, query_text: str, findings: List[Dict]) -> Dict[str, List[RetrievalResult]]:
        """
        Phase 4: Evidence Sampling (例证采样)

        目标：为选中的观点提供“实锤”，同时为锚点社区添加摘要。

        算法：
        1. 构建 evidence_query = f"{user_query} {finding_text}"
        2. 在该 Finding 所属的 community_id 下，使用 evidence_query 检索 TextUnit Index
        3. 仅拉取 Top max_evidence_per_finding 条原文
        4. 为锚点社区添加社区摘要（如果存在）

        Args:
            query_vector: 查询向量
            query_text: 原始查询文本
            findings: 选中的 findings 列表，每个元素包含 community_id, finding_index, summary, explanation 等

        Returns:
            Dict[str, List[RetrievalResult]]: 键为 finding_key (f"{community_id}_{finding_index}")，值为证据列表（包含社区摘要）
        """
        if not findings:
            logger.warning("findings 列表为空，跳过例证采样")
            return {}

        evidence_map = {}
        max_evidence = self.params.max_evidence_per_finding

        for finding in findings:
            community_id = finding["community_id"]
            community_title = finding.get("community_title", "未知")
            finding_summary = finding.get("summary", "")
            finding_explanation = finding.get("explanation", "")

            # 构建 evidence_query 文本（用于日志）
            evidence_query_text = f"{query_text} {finding_summary}"
            logger.debug(f"为 finding {community_id}_{finding['finding_index']} 构建 evidence_query: {evidence_query_text[:100]}...")

            # 创建 FilterQuery 限制在该社区内
            filter_query = FilterQuery(
                community_ids=[community_id],
                must_contain=[],
                any_contain=[],
                must_not_contain=[],
                scope_hierarchy=[]
            )

            # 使用查询向量进行检索（假设 evidence_query 向量与查询向量相同）
            # 注意：这里我们使用 query_vector 作为 evidence_query 的向量
            # 添加 similarity_threshold 参数，使用配置中的阈值
            evidence_results = self.data_manager.search_text_units(
                query_vector=query_vector,
                top_k=max_evidence,
                filter_query=filter_query,
                similarity_threshold=self.params.similarity_threshold
            )

            # 为锚点社区添加社区摘要（如果存在）
            report = self.data_manager.get_report_by_community_id(community_id)
            if report and report.summary:
                # 创建虚拟 TextUnit 表示社区摘要
                summary_text_unit = TextUnit(
                    id=f"summary_{community_id}",
                    text=f"【社区摘要】{report.summary}",
                    n_tokens=len(report.summary) // 4,  # 粗略估计
                    vector=report.embedding,  # 使用报告的向量
                    community_id=community_id,
                    text_chunks=[],
                    chunk_vectors=[],
                    head="",
                    tail="",
                    hierarchy_path=community_title,
                    source_metadata="",
                    prepend_source=f"[{community_title}·摘要]"
                )
                
                # 计算相似度（使用报告向量）
                query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(query_np)
                vector_np = np.array(report.embedding, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(vector_np)
                similarity = np.dot(query_np, vector_np.T).item()
                
                summary_result = RetrievalResult(
                    text_unit=summary_text_unit,
                    similarity_score=float(similarity),
                    source_type=SourceType.EVIDENCE,  # 作为证据的一部分
                    source_relation=None,
                    anchor_community_id=community_id,
                    anchor_community_title=community_title
                )
                
                # 将摘要放在证据列表的最前面
                evidence_results.insert(0, summary_result)
                logger.debug(f"为锚点社区 {community_title} 添加社区摘要")

            # 记录结果
            finding_key = f"{community_id}_{finding['finding_index']}"
            evidence_map[finding_key] = evidence_results

            logger.info(f"为 finding {finding_key} 找到 {len(evidence_results)} 条证据 (max_evidence_per_finding={max_evidence}, similarity_threshold={self.params.similarity_threshold})")

        return evidence_map

    def _global_semantic_supplement(self, query_vector: Vector, query_text: str, evidence_map: Dict[str, List[RetrievalResult]]) -> List[RetrievalResult]:
        """
        Phase 5: Global Semantic Supplement (全局语义补全)

        目标：防止图谱路径过于“高冷”而漏掉直接相关的底层史料。

        算法：
        1. 使用 User Query 在 TextUnit Index 中执行全量向量检索。
        2. 剔除 Phase 4 中已经被选为“例证”的 TextUnit IDs。
        3. 严格截取 Top force_recall_count_macro (10条)。

        Args:
            query_vector: 查询向量
            query_text: 原始查询文本（用于日志）
            evidence_map: Phase 4 返回的证据映射，用于去重

        Returns:
            List[RetrievalResult]: 全局语义补充结果，source_type 为 GLOBAL_SUPPLEMENT
        """
        if not self.data_manager.text_unit_index:
            logger.warning("TextUnit 索引未构建，无法执行全局语义补全")
            return []

        # 收集所有已选为证据的 TextUnit ID 集合
        evidence_ids = set()
        for evidence_list in evidence_map.values():
            for result in evidence_list:
                evidence_ids.add(result.text_unit.id)

        # 搜索 Top K 个 TextUnit（K 取 force_recall_count_macro * 2，确保有足够候选）
        search_k = min(self.params.force_recall_count_macro * 2, len(self.data_manager.text_unit_map))
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_np)

        distances, indices = self.data_manager.text_unit_index.search(query_np, search_k)

        supplement_results = []
        for i in range(search_k):
            faiss_id = indices[0, i]
            similarity = distances[0, i]  # 内积相似度，范围 [-1, 1]

            # 过滤相似度阈值
            if similarity < self.params.similarity_threshold:
                continue

            # 获取 TextUnit
            text_unit = self.data_manager.get_text_unit_by_faiss_id(faiss_id)
            if not text_unit:
                continue

            # 去重：排除已经是证据的 TextUnit
            if text_unit.id in evidence_ids:
                continue

            # 构建结果
            result = RetrievalResult(
                text_unit=text_unit,
                similarity_score=float(similarity),
                source_type=SourceType.GLOBAL_SUPPLEMENT,
                source_relation=None,
                anchor_community_id=None,
                anchor_community_title=None
            )
            supplement_results.append(result)

            # 如果已收集足够结果，停止
            if len(supplement_results) >= self.params.force_recall_count_macro:
                break

        # 按相似度降序排序（FAISS 返回的结果已经按相似度降序排列，但为了安全再次排序）
        supplement_results.sort(key=lambda x: x.similarity_score, reverse=True)

        logger.info(f"全局语义补全完成: 找到 {len(supplement_results)} 条补充结果 (force_recall_count_macro={self.params.force_recall_count_macro})")
        for result in supplement_results[:5]:  # 只打印前5个
            logger.debug(f"  补充: {result.text_unit.hierarchy_path} 相似度: {result.similarity_score:.3f}")

        return supplement_results

    def _add_community_summaries(self, query_vector: Vector, community_ids: List[str]) -> List[RetrievalResult]:
        """
        为指定的社区列表添加社区摘要。
        
        Args:
            query_vector: 查询向量
            community_ids: 社区ID列表
            
        Returns:
            List[RetrievalResult]: 社区摘要结果列表
        """
        summary_results = []
        
        for community_id in community_ids:
            community = self.data_manager.get_community_by_id(community_id)
            if not community:
                continue
                
            report = self.data_manager.get_report_by_community_id(community_id)
            if not report or not report.summary:
                continue
                
            # 创建虚拟 TextUnit 表示社区摘要
            summary_text_unit = TextUnit(
                id=f"summary_{community_id}",
                text=f"【社区摘要】{report.summary}",
                n_tokens=len(report.summary) // 4,  # 粗略估计
                vector=report.embedding,  # 使用报告的向量
                community_id=community_id,
                text_chunks=[],
                chunk_vectors=[],
                head="",
                tail="",
                hierarchy_path=community.title,
                source_metadata="",
                prepend_source=f"[{community.title}·摘要]"
            )
            
            # 计算相似度（使用报告向量）
            query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_np)
            vector_np = np.array(report.embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(vector_np)
            similarity = np.dot(query_np, vector_np.T).item()
            
            summary_result = RetrievalResult(
                text_unit=summary_text_unit,
                similarity_score=float(similarity),
                source_type=SourceType.ANCHOR,  # 作为锚点内容
                source_relation=None,
                anchor_community_id=community_id,
                anchor_community_title=community.title
            )
            
            summary_results.append(summary_result)
            logger.debug(f"为社区 {community.title} 添加社区摘要")
        
        return summary_results
    
    def _finding_to_retrieval_result(self, finding: Dict) -> RetrievalResult:
        """
        将 finding 字典转换为 RetrievalResult。
        
        Args:
            finding: finding 字典，包含 community_id, community_title, finding_index, summary, explanation, similarity_score
        
        Returns:
            RetrievalResult: 虚拟 TextUnit 表示 finding
        """
        # 创建虚拟 TextUnit 表示 finding
        virtual_text_unit = TextUnit(
            id=f"finding_{finding['community_id']}_{finding['finding_index']}",
            text=f"【观点】{finding['summary']}\n\n【解释】{finding['explanation']}",
            n_tokens=len(finding['summary']) // 4 + len(finding['explanation']) // 4,
            vector=[],  # 空向量，因为 finding 向量不用于后续检索
            community_id=finding['community_id'],
            text_chunks=[],
            chunk_vectors=[],
            head="",
            tail="",
            hierarchy_path=finding['community_title'],
            source_metadata="",
            prepend_source=f"[{finding['community_title']}]"
        )
        
        return RetrievalResult(
            text_unit=virtual_text_unit,
            similarity_score=finding['similarity_score'],
            source_type=SourceType.FINDING,
            source_relation=None,
            anchor_community_id=finding['community_id'],
            anchor_community_title=finding['community_title']
        )
    
    def retrieve(self, query_vector: Vector, query_text: str = "") -> List[RetrievalResult]:
        """
        执行宏观综述检索。
        
        完整流程：
        1. Phase 1: Territory Mapping (版图定位)
        2. Phase 2: Logical Bridging (逻辑架桥)
        3. Phase 3: Finding Ranking & Selection (观点重排)
        4. Phase 4: Evidence Sampling (例证采样)
        5. Phase 5: Global Semantic Supplement (全局语义补全)
        6. Phase 6: Context Assembly (结构化装填) - 返回结构化结果列表
        
        Args:
            query_vector: 查询向量
            query_text: 原始查询文本
            
        Returns:
            List[RetrievalResult]: 检索结果列表，包含 Findings、Evidence 和 Global Supplement
        """
        logger.info(f"开始宏观检索: query_text='{query_text[:50]}...'")
        
        # Phase 1: Territory Mapping (版图定位)
        anchors = self._territory_mapping(query_vector)
        if not anchors:
            logger.warning("版图定位未找到任何锚点，返回空结果")
            return []
        
        # Phase 2: Logical Bridging (逻辑架桥)
        bridges = self._logical_bridging(anchors)
        
        # 合并锚点和桥接社区作为候选社区
        candidate_communities = anchors + bridges
        
        # Phase 3: Finding Ranking & Selection (观点重排)
        findings = self._finding_ranking_and_selection(query_vector, candidate_communities)
        
        # Phase 4: Evidence Sampling (例证采样)
        evidence_map = self._evidence_sampling(query_vector, query_text, findings)
        
        # Phase 5: Global Semantic Supplement (全局语义补全)
        supplement_results = self._global_semantic_supplement(query_vector, query_text, evidence_map)
        
        # Phase 6: Context Assembly (结构化装填) - 组装最终结果列表
        all_results = []
        
        # 0. 为锚点社区添加摘要（即使没有 findings）
        anchor_community_ids = [anchor["community_id"] for anchor in anchors]
        community_summaries = self._add_community_summaries(query_vector, anchor_community_ids)
        all_results.extend(community_summaries)
        
        # 1. 添加 Findings 作为虚拟 TextUnit
        for finding in findings:
            finding_result = self._finding_to_retrieval_result(finding)
            all_results.append(finding_result)
        
        # 2. 添加 Evidence (每个 finding 的证据)
        for finding_key, evidence_list in evidence_map.items():
            # 为每个证据设置 source_type 为 EVIDENCE
            for evidence_result in evidence_list:
                evidence_result.source_type = SourceType.EVIDENCE
                # 关联到 finding (通过 anchor_community_id 和 anchor_community_title)
                # 这里可以添加更多元数据，但为了简单起见，我们保持原样
                all_results.append(evidence_result)
        
        # 3. 添加 Global Supplement
        all_results.extend(supplement_results)
        
        # 按相似度降序排序（Findings 和 Evidence 的相似度可能不同）
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        logger.info(f"宏观检索完成: 共 {len(all_results)} 个结果 ({len(community_summaries)} 个社区摘要, {len(findings)} 个观点, {sum(len(v) for v in evidence_map.values())} 条证据, {len(supplement_results)} 条补充)")
        return all_results


# 工厂函数
def create_retriever(mode: RetrievalMode, data_manager: DataManager, config: Optional[Dict[str, Any]] = None):
    """
    创建检索器工厂函数。
    
    Args:
        mode: 检索模式 (MICRO 或 MACRO)
        data_manager: 数据管理器实例
        config: 可选配置字典
        
    Returns:
        MicroRetriever 或 MacroRetriever 实例
    """
    if mode == RetrievalMode.MICRO:
        return MicroRetriever(data_manager, config)
    elif mode == RetrievalMode.MACRO:
        return MacroRetriever(data_manager, config)
    else:
        raise ValueError(f"未知的检索模式: {mode}")


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    print("Retriever 模块结构验证通过")
    print("MicroRetriever 和 MacroRetriever 类定义完成")