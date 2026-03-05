"""
DataManager 模块 - 预处理与索引层

本模块是《太平御览》GraphRAG 系统的核心基础模块，负责：
1. 资产加载与层级路由
2. 长文切片管理
3. 向量索引加速（FAISS）
4. 过滤器引擎
5. Root Lineage Mapping
6. Finding Index 构建
7. 原子化检索方法

严格遵循 project.md 2.1 节（DataManager：预处理与索引层）的技术规格和算法要求。
"""

import logging
import json
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from collections import defaultdict, deque

import polars as pl
import numpy as np
import faiss
import networkx as nx

from schemas import (
    TextUnit, Community, Relationship, CommunityReport, FilterQuery,
    RetrievalResult, ProcessedResult, TextUnitType, SourceType,
    CommunityMap, RelationshipMap, NodeToRootMap, Vector, Matrix
)
from config import settings, INPUT_COMMUNITIES, INPUT_TEXT_UNITS, INPUT_REPORTS


logger = logging.getLogger(__name__)


class DataManager:
    """
    数据管理器 - 负责所有数据资产的加载、索引构建和检索加速。

    设计原则：
    1. 一次性加载，全局共享
    2. 物理指针锁定（__faiss_id__ 作为唯一可信指针）
    3. 双索引加速（TextUnit Index + Report Index）
    4. 过滤器引擎支持布尔掩码裁切
    5. 层级染色法预计算 Root Lineage
    """

    def __init__(self, config=None):
        """
        初始化 DataManager。

        Args:
            config: 可选配置字典，覆盖默认配置
        """
        self.config = config or {}
        
        # 核心数据资产
        self.community_map: CommunityMap = {}
        self.leaf_communities: Set[str] = set()
        self.hierarchy_tree: nx.DiGraph = nx.DiGraph()
        self.text_unit_map: Dict[str, TextUnit] = {}
        self.text_unit_id_to_community_id: Dict[str, str] = {}
        self.community_reports: Dict[str, CommunityReport] = {}
        
        # 反向索引
        self.community_to_text_units: Dict[str, List[TextUnit]] = defaultdict(list)
        
        # 向量索引
        self.text_unit_index: Optional[faiss.IndexFlatIP] = None
        self.report_index: Optional[faiss.IndexFlatIP] = None
        self.finding_index: Optional[faiss.IndexFlatIP] = None
        
        # 向量矩阵
        self.text_unit_matrix: Optional[np.ndarray] = None
        self.report_matrix: Optional[np.ndarray] = None
        self.finding_matrix: Optional[np.ndarray] = None
        
        # 物理指针映射
        self.faiss_id_to_text_unit_id: Dict[int, str] = {}
        self.text_unit_id_to_faiss_id: Dict[str, int] = {}
        self.faiss_id_to_report_community_id: Dict[int, str] = {}
        self.report_community_id_to_faiss_id: Dict[str, int] = {}
        self.faiss_id_to_finding_metadata: Dict[int, Tuple[str, int]] = {}  # (community_id, finding_index)
        
        # 过滤器引擎相关
        self.node_to_root_map: NodeToRootMap = {}
        
        # Finding Index 全局元数据
        self.global_findings_metadata: List[Dict[str, Any]] = []
        self.global_findings_matrix: Optional[np.ndarray] = None
        
        # 加载标志
        self._loaded = False
        
        logger.info("DataManager 初始化完成")

    def load_all_assets(self) -> None:
        """
        加载所有数据资产，构建核心数据结构。

        步骤：
        1. 加载社区数据，构建 community_map 和 leaf_communities
        2. 构建 hierarchy_tree
        3. 加载文本单元数据，建立反向索引
        4. 加载社区报告数据
        5. 构建向量索引
        6. 预计算 Root Lineage Mapping
        7. 构建 Finding Index
        """
        if self._loaded:
            logger.warning("数据已加载，跳过重复加载")
            return
        
        logger.info("开始加载所有数据资产...")
        
        # 1. 加载社区数据
        self._load_communities()
        
        # 2. 构建层级树
        self._build_hierarchy_tree()
        
        # 3. 加载文本单元数据
        self._load_text_units()
        
        # 4. 加载社区报告数据
        self._load_community_reports()
        
        # 5. 构建向量索引
        self._build_vector_indices()
        
        # 6. 预计算 Root Lineage Mapping
        self._compute_root_lineage_mapping()
        
        # 7. 构建 Finding Index
        self._build_finding_index()
        
        self._loaded = True
        logger.info("所有数据资产加载完成")

    def _load_communities(self) -> None:
        """加载 create_final_communities.parquet，构建 community_map 和 leaf_communities 集合"""
        logger.info(f"加载社区数据: {INPUT_COMMUNITIES}")
        try:
            df = pl.read_parquet(INPUT_COMMUNITIES)
            logger.info(f"社区数据形状: {df.shape}")
            
            for row in df.iter_rows(named=True):
                # 处理 null 值
                parent_id = row['parent_id']
                if parent_id is None:
                    parent_id = ""
                
                child_ids = row['child_ids']
                if child_ids is None:
                    child_ids = []
                
                community = Community(
                    id=row['id'],
                    title=row['title'],
                    level=row['level'],
                    parent_id=parent_id,
                    child_ids=child_ids,
                    is_leaf=row['is_leaf']
                )
                self.community_map[community.id] = community
                
                if community.is_leaf:
                    self.leaf_communities.add(community.id)
            
            logger.info(f"加载了 {len(self.community_map)} 个社区，其中 {len(self.leaf_communities)} 个叶子社区")
            
        except Exception as e:
            logger.error(f"加载社区数据失败: {e}")
            raise

    def _build_hierarchy_tree(self) -> None:
        """利用 parent_id 和 child_ids 构建 hierarchy_tree（NetworkX 有向图）"""
        logger.info("构建层级树...")
        
        # 添加所有节点
        for community_id, community in self.community_map.items():
            self.hierarchy_tree.add_node(community_id, **community.model_dump())
        
        # 添加边（父 -> 子）
        for community_id, community in self.community_map.items():
            if community.parent_id and community.parent_id in self.community_map:
                self.hierarchy_tree.add_edge(community.parent_id, community_id)
        
        # 验证树结构
        if not nx.is_directed_acyclic_graph(self.hierarchy_tree):
            logger.warning("层级树包含环，可能数据有问题")
        
        logger.info(f"层级树构建完成: {self.hierarchy_tree.number_of_nodes()} 个节点, "
                   f"{self.hierarchy_tree.number_of_edges()} 条边")

    def _load_text_units(self) -> None:
        """加载 create_final_text_units.parquet，建立 text_unit_id -> community_id 反向索引"""
        logger.info(f"加载文本单元数据: {INPUT_TEXT_UNITS}")
        try:
            df = pl.read_parquet(INPUT_TEXT_UNITS)
            logger.info(f"文本单元数据形状: {df.shape}")
            
            for row in df.iter_rows(named=True):
                # 处理 text_chunks 和 chunk_vectors（可能是 JSON 字符串）
                text_chunks = row['text_chunks']
                chunk_vectors = row['chunk_vectors']
                
                if isinstance(text_chunks, str) and text_chunks.strip():
                    try:
                        text_chunks = json.loads(text_chunks)
                    except json.JSONDecodeError:
                        text_chunks = []
                elif not isinstance(text_chunks, list):
                    text_chunks = []
                
                if isinstance(chunk_vectors, str) and chunk_vectors.strip():
                    try:
                        chunk_vectors = json.loads(chunk_vectors)
                    except json.JSONDecodeError:
                        chunk_vectors = []
                elif not isinstance(chunk_vectors, list):
                    chunk_vectors = []
                
                # 处理 prepend_source 可能是布尔值的情况
                prepend_source = row.get('prepend_source', '')
                if isinstance(prepend_source, bool):
                    prepend_source = str(prepend_source)
                
                text_unit = TextUnit(
                    id=row['id'],
                    text=row['text'],
                    n_tokens=row['n_tokens'],
                    vector=row['vector'],
                    community_id=row['community_id'],
                    text_chunks=text_chunks,
                    chunk_vectors=chunk_vectors,
                    head=row.get('head', ''),
                    tail=row.get('tail', ''),
                    hierarchy_path=row.get('hierarchy_path', ''),
                    source_metadata=row.get('source_metadata', ''),
                    prepend_source=prepend_source
                )
                
                self.text_unit_map[text_unit.id] = text_unit
                self.text_unit_id_to_community_id[text_unit.id] = text_unit.community_id
                self.community_to_text_units[text_unit.community_id].append(text_unit)
            
            logger.info(f"加载了 {len(self.text_unit_map)} 个文本单元")
            
        except Exception as e:
            logger.error(f"加载文本单元数据失败: {e}")
            raise

    def _load_community_reports(self) -> None:
        """加载 create_final_community_reports.parquet，构建 community_reports 字典"""
        logger.info(f"加载社区报告数据: {INPUT_REPORTS}")
        try:
            df = pl.read_parquet(INPUT_REPORTS)
            logger.info(f"社区报告数据形状: {df.shape}")
            
            for row in df.iter_rows(named=True):
                # 处理 finding_vectors（可能是 JSON 字符串）
                finding_vectors = row['finding_vectors']
                if isinstance(finding_vectors, str) and finding_vectors.strip():
                    try:
                        finding_vectors = json.loads(finding_vectors)
                    except json.JSONDecodeError:
                        finding_vectors = []
                elif not isinstance(finding_vectors, list):
                    finding_vectors = []
                
                report = CommunityReport(
                    community_id=row['community_id'],
                    title=row['title'],
                    level=row['level'],
                    summary=row['summary'],
                    findings=row['findings'],
                    full_content=row['full_content'],
                    embedding=row['embedding'],
                    finding_vectors=finding_vectors
                )
                
                self.community_reports[report.community_id] = report
            
            logger.info(f"加载了 {len(self.community_reports)} 个社区报告")
            
        except Exception as e:
            logger.error(f"加载社区报告数据失败: {e}")
            raise

    # ==================== 长文切片管理 ====================
    
    def is_long_text(self, text_unit: TextUnit) -> bool:
        """
        判断文本单元是否为长文。
        
        根据 project.md 2.1 节 B 部分，长文阈值由配置决定。
        默认使用 config.LONG_TEXT_THRESHOLD（默认 1000 字符）。
        """
        from config import LONG_TEXT_THRESHOLD
        return len(text_unit.text) > LONG_TEXT_THRESHOLD
    
    def get_text_unit_chunks(self, text_unit: TextUnit) -> List[Tuple[str, List[float]]]:
        """
        获取文本单元的切片列表，每个切片包含文本和向量。
        
        对于长文（is_long_text 为 True），返回预计算的 text_chunks 和 chunk_vectors。
        对于短文，返回一个包含完整文本和完整向量的单一切片。
        
        Returns:
            List[Tuple[str, List[float]]]: 每个元素为 (chunk_text, chunk_vector)
        """
        if self.is_long_text(text_unit) and text_unit.text_chunks and text_unit.chunk_vectors:
            # 确保切片数量与向量数量一致
            chunks = list(zip(text_unit.text_chunks, text_unit.chunk_vectors))
            return chunks
        else:
            # 短文：返回完整文本作为单一切片
            return [(text_unit.text, text_unit.vector)]
    
    def get_best_chunk_for_query(self, text_unit: TextUnit, query_vector: Vector) -> Tuple[str, List[float], float]:
        """
        对于长文，选择与查询向量最相似的切片。
        
        Args:
            text_unit: 文本单元
            query_vector: 查询向量（4096维）
            
        Returns:
            Tuple[str, List[float], float]: (最佳切片文本, 最佳切片向量, 相似度分数)
            
        注意：如果文本单元不是长文，则返回完整文本和向量，相似度为 1.0。
        """
        chunks = self.get_text_unit_chunks(text_unit)
        if len(chunks) == 1:
            # 短文或没有切片
            return chunks[0][0], chunks[0][1], 1.0
        
        # 计算每个切片向量与查询向量的相似度（内积）
        best_score = -float('inf')
        best_chunk_text = ""
        best_chunk_vector = []
        
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        for chunk_text, chunk_vector in chunks:
            chunk_np = np.array(chunk_vector, dtype=np.float32).reshape(1, -1)
            # L2 归一化已在索引构建时完成，这里直接计算内积
            score = np.dot(query_np, chunk_np.T).item()
            if score > best_score:
                best_score = score
                best_chunk_text = chunk_text
                best_chunk_vector = chunk_vector
        
        return best_chunk_text, best_chunk_vector, best_score

    # ==================== 向量索引加速（FAISS） ====================
    
    def _build_vector_indices(self) -> None:
        """
        构建向量索引（TextUnit Index 和 Report Index）。
        
        步骤：
        1. 收集所有文本单元的向量，构建 TextUnit Index
        2. 收集所有社区报告的向量，构建 Report Index
        3. 执行 L2 Normalization（关键步骤）
        4. 建立物理指针映射（__faiss_id__）
        """
        logger.info("开始构建向量索引...")
        
        # 1. 构建 TextUnit Index
        self._build_text_unit_index()
        
        # 2. 构建 Report Index
        self._build_report_index()
        
        logger.info("向量索引构建完成")
    
    def _build_text_unit_index(self) -> None:
        """构建 TextUnit 向量索引"""
        from config import VECTOR_DIM
        
        if not self.text_unit_map:
            logger.warning("没有文本单元数据，跳过 TextUnit 索引构建")
            return
        
        # 收集向量和 ID
        vectors = []
        text_unit_ids = []
        
        for text_unit_id, text_unit in self.text_unit_map.items():
            vectors.append(text_unit.vector)
            text_unit_ids.append(text_unit_id)
        
        # 转换为 numpy 数组
        matrix = np.array(vectors, dtype=np.float32)
        logger.info(f"TextUnit 矩阵形状: {matrix.shape}")
        
        # L2 归一化（关键步骤）
        faiss.normalize_L2(matrix)
        
        # 创建 FAISS 索引
        dimension = VECTOR_DIM
        self.text_unit_index = faiss.IndexFlatIP(dimension)
        self.text_unit_index.add(matrix)
        
        # 保存矩阵用于后续操作
        self.text_unit_matrix = matrix
        
        # 建立物理指针映射
        for faiss_id, text_unit_id in enumerate(text_unit_ids):
            self.faiss_id_to_text_unit_id[faiss_id] = text_unit_id
            self.text_unit_id_to_faiss_id[text_unit_id] = faiss_id
            # 在 TextUnit 对象中设置 faiss_id
            self.text_unit_map[text_unit_id].faiss_id = faiss_id
        
        logger.info(f"TextUnit 索引构建完成: {len(text_unit_ids)} 个向量，维度 {dimension}")
    
    def _build_report_index(self) -> None:
        """构建 CommunityReport 向量索引"""
        from config import VECTOR_DIM
        
        if not self.community_reports:
            logger.warning("没有社区报告数据，跳过 Report 索引构建")
            return
        
        # 收集向量和社区 ID
        vectors = []
        community_ids = []
        
        for community_id, report in self.community_reports.items():
            vectors.append(report.embedding)
            community_ids.append(community_id)
        
        # 转换为 numpy 数组
        matrix = np.array(vectors, dtype=np.float32)
        logger.info(f"Report 矩阵形状: {matrix.shape}")
        
        # L2 归一化
        faiss.normalize_L2(matrix)
        
        # 创建 FAISS 索引
        dimension = VECTOR_DIM
        self.report_index = faiss.IndexFlatIP(dimension)
        self.report_index.add(matrix)
        
        # 保存矩阵
        self.report_matrix = matrix
        
        # 建立物理指针映射
        for faiss_id, community_id in enumerate(community_ids):
            self.faiss_id_to_report_community_id[faiss_id] = community_id
            self.report_community_id_to_faiss_id[community_id] = faiss_id
        
        logger.info(f"Report 索引构建完成: {len(community_ids)} 个向量，维度 {dimension}")
    
    # ==================== 物理指针锁定机制 ====================
    
    def get_text_unit_by_faiss_id(self, faiss_id: int) -> Optional[TextUnit]:
        """
        通过 FAISS ID 获取文本单元。
        
        FAISS ID 是唯一可信的物理指针，用于在检索后回查原始数据。
        """
        text_unit_id = self.faiss_id_to_text_unit_id.get(faiss_id)
        if text_unit_id is None:
            logger.warning(f"无效的 FAISS ID: {faiss_id}")
            return None
        return self.text_unit_map.get(text_unit_id)
    
    def get_report_by_faiss_id(self, faiss_id: int) -> Optional[CommunityReport]:
        """通过 FAISS ID 获取社区报告"""
        community_id = self.faiss_id_to_report_community_id.get(faiss_id)
        if community_id is None:
            logger.warning(f"无效的 FAISS ID: {faiss_id}")
            return None
        return self.community_reports.get(community_id)
    
    def validate_faiss_id(self, faiss_id: int, index_type: str = "text_unit") -> bool:
        """
        验证 FAISS ID 是否有效。
        
        Args:
            faiss_id: FAISS 物理行号
            index_type: 索引类型，可选 "text_unit" 或 "report"
        
        Returns:
            bool: 是否有效
        """
        if index_type == "text_unit":
            return faiss_id in self.faiss_id_to_text_unit_id
        elif index_type == "report":
            return faiss_id in self.faiss_id_to_report_community_id
        else:
            logger.error(f"未知的索引类型: {index_type}")
            return False
    
    # ==================== 过滤器引擎 ====================
    
    def apply_filter(self, filter_query: FilterQuery) -> np.ndarray:
        """
        根据 FilterQuery 生成布尔掩码。
        
        Args:
            filter_query: 过滤器查询对象
            
        Returns:
            np.ndarray: 布尔掩码，True 表示通过过滤的条目
        """
        if not filter_query:
            # 无过滤器，返回全 True 掩码
            return np.ones(len(self.text_unit_map), dtype=bool)
        
        masks = []
        
        # 1. 关键词过滤
        if (filter_query.must_contain or filter_query.any_contain or
            filter_query.must_not_contain):
            keyword_mask = self._filter_by_keywords(filter_query)
            masks.append(keyword_mask)
        
        # 2. 层级过滤
        if filter_query.scope_hierarchy:
            hierarchy_mask = self._filter_by_hierarchy(filter_query.scope_hierarchy)
            masks.append(hierarchy_mask)
        
        # 3. 社区 ID 过滤
        if filter_query.community_ids:
            community_mask = self._filter_by_community_ids(filter_query.community_ids)
            masks.append(community_mask)
        
        # 合并所有掩码
        if masks:
            final_mask = self._merge_masks(masks, filter_query)
        else:
            final_mask = np.ones(len(self.text_unit_map), dtype=bool)
        
        return final_mask
    
    def _filter_by_keywords(self, filter_query: FilterQuery) -> np.ndarray:
        """
        关键词过滤：根据 must_contain, any_contain, must_not_contain 生成掩码。
        
        逻辑：
        - must_contain: 文本必须包含所有关键词（AND）
        - any_contain: 文本至少包含一个关键词（OR）
        - must_not_contain: 文本不能包含任何关键词（NOT）
        """
        mask = np.ones(len(self.text_unit_map), dtype=bool)
        
        # 获取文本单元列表（按 FAISS ID 顺序）
        text_units = [self.get_text_unit_by_faiss_id(i) for i in range(len(self.text_unit_map))]
        
        for i, text_unit in enumerate(text_units):
            if text_unit is None:
                mask[i] = False
                continue
            
            text = text_unit.text.lower()
            
            # must_contain 检查
            if filter_query.must_contain:
                for keyword in filter_query.must_contain:
                    if keyword.lower() not in text:
                        mask[i] = False
                        break
            
            # any_contain 检查（如果 must_contain 已失败，跳过）
            if mask[i] and filter_query.any_contain:
                any_match = False
                for keyword in filter_query.any_contain:
                    if keyword.lower() in text:
                        any_match = True
                        break
                if not any_match:
                    mask[i] = False
            
            # must_not_contain 检查
            if mask[i] and filter_query.must_not_contain:
                for keyword in filter_query.must_not_contain:
                    if keyword.lower() in text:
                        mask[i] = False
                        break
        
        return mask
    
    def _filter_by_hierarchy(self, scope_hierarchy: List[str]) -> np.ndarray:
        """
        层级过滤：只保留层级路径前缀匹配的文本单元。
        
        Args:
            scope_hierarchy: 层级路径前缀列表，例如 ["天部", "天部 > 日"]
            
        Returns:
            np.ndarray: 布尔掩码
        """
        mask = np.zeros(len(self.text_unit_map), dtype=bool)
        
        for i in range(len(self.text_unit_map)):
            text_unit = self.get_text_unit_by_faiss_id(i)
            if text_unit is None:
                continue
            
            hierarchy_path = text_unit.hierarchy_path
            if not hierarchy_path:
                # 如果没有层级路径，跳过过滤（保留）
                mask[i] = True
                continue
            
            # 检查是否匹配任意前缀
            for prefix in scope_hierarchy:
                if hierarchy_path.startswith(prefix):
                    mask[i] = True
                    break
        
        return mask
    
    def _filter_by_community_ids(self, community_ids: List[str]) -> np.ndarray:
        """
        社区 ID 过滤：只保留属于指定社区的文本单元。
        """
        mask = np.zeros(len(self.text_unit_map), dtype=bool)
        
        for i in range(len(self.text_unit_map)):
            text_unit = self.get_text_unit_by_faiss_id(i)
            if text_unit is None:
                continue
            
            if text_unit.community_id in community_ids:
                mask[i] = True
        
        return mask
    
    def _merge_masks(self, masks: List[np.ndarray], filter_query: FilterQuery) -> np.ndarray:
        """
        合并多个掩码。
        
        默认使用 AND 逻辑：所有条件必须同时满足。
        根据 project.md，过滤器引擎支持复合逻辑结构。
        """
        if not masks:
            return np.ones(len(self.text_unit_map), dtype=bool)
        
        # 初始掩码为全 True
        final_mask = np.ones(len(self.text_unit_map), dtype=bool)
        
        # 应用 AND 逻辑
        for mask in masks:
            final_mask = np.logical_and(final_mask, mask)
        
        return final_mask

    def _compute_root_lineage_mapping(self) -> None:
        """
        使用染色法预计算 node_to_root_map。
        
        算法：
        1. 找到所有根节点（parent_id 为空）
        2. 对每个根节点进行 BFS/DFS 染色，将其所有后代标记为该根节点
        3. 记录 node_to_root_map[后代节点] = 根节点 ID
        
        这样可以在 O(1) 时间内查询任意节点的根节点。
        """
        logger.info("开始计算 Root Lineage Mapping...")
        
        # 清空现有映射
        self.node_to_root_map.clear()
        
        # 找到所有根节点（parent_id 为空）
        root_nodes = []
        for community_id, community in self.community_map.items():
            if not community.parent_id:
                root_nodes.append(community_id)
        
        logger.info(f"找到 {len(root_nodes)} 个根节点")
        
        # 对每个根节点进行 BFS 染色
        for root_id in root_nodes:
            # BFS 队列
            queue = deque([root_id])
            
            while queue:
                current_id = queue.popleft()
                
                # 如果已经染色过（理论上不应该发生，因为树结构无环）
                if current_id in self.node_to_root_map:
                    continue
                
                # 染色
                self.node_to_root_map[current_id] = root_id
                
                # 将子节点加入队列
                current_community = self.community_map.get(current_id)
                if current_community:
                    for child_id in current_community.child_ids:
                        if child_id in self.community_map:
                            queue.append(child_id)
        
        logger.info(f"Root Lineage Mapping 计算完成: {len(self.node_to_root_map)} 个节点已染色")
        
        # 验证所有节点都有映射
        missing = set(self.community_map.keys()) - set(self.node_to_root_map.keys())
        if missing:
            logger.warning(f"有 {len(missing)} 个节点未染色，可能是孤立节点")

    def _build_finding_index(self) -> None:
        """
        扁平化 findings 和 finding_vectors，构建全局统一的 Finding Index。
        
        步骤：
        1. 遍历所有社区报告，解析 findings JSON
        2. 收集每个 finding 的向量（来自 finding_vectors）
        3. 构建全局元数据列表和向量矩阵
        4. 创建 FAISS 索引并建立物理指针映射
        """
        logger.info("开始构建 Finding Index...")
        
        from config import VECTOR_DIM
        
        # 清空现有数据
        self.global_findings_metadata.clear()
        self.faiss_id_to_finding_metadata.clear()
        
        vectors = []
        metadata_list = []
        
        for community_id, report in self.community_reports.items():
            # 解析 findings，捕获可能的验证错误
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
            
            # 收集每个 finding
            for idx, (finding, vector) in enumerate(zip(findings, finding_vectors)):
                metadata = {
                    "community_id": community_id,
                    "community_title": report.title,
                    "finding_index": idx,
                    "summary": finding.summary,
                    "explanation": finding.explanation,
                    "report_level": report.level
                }
                metadata_list.append(metadata)
                vectors.append(vector)
        
        if not vectors:
            logger.warning("没有找到任何 finding 向量，跳过 Finding Index 构建")
            return
        
        # 转换为 numpy 数组
        matrix = np.array(vectors, dtype=np.float32)
        logger.info(f"Finding 矩阵形状: {matrix.shape}")
        
        # L2 归一化
        faiss.normalize_L2(matrix)
        
        # 创建 FAISS 索引
        dimension = VECTOR_DIM
        self.finding_index = faiss.IndexFlatIP(dimension)
        self.finding_index.add(matrix)
        
        # 保存矩阵
        self.finding_matrix = matrix
        
        # 建立物理指针映射
        for faiss_id, metadata in enumerate(metadata_list):
            self.faiss_id_to_finding_metadata[faiss_id] = (metadata["community_id"], metadata["finding_index"])
        
        # 保存全局元数据
        self.global_findings_metadata = metadata_list
        
        logger.info(f"Finding Index 构建完成: {len(metadata_list)} 个 findings，维度 {dimension}")

    def search_text_units(self, query_vector: Vector, top_k: int = 10,
                         filter_query: Optional[FilterQuery] = None,
                         similarity_threshold: float = 0.0) -> List[RetrievalResult]:
        """
        原子化检索方法：在 TextUnit Index 中搜索最相似的文本单元。
        
        确保物理指针回查安全（使用 __faiss_id__ 作为唯一可信的物理指针）。
        
        Args:
            query_vector: 查询向量（4096维）
            top_k: 返回结果数量
            filter_query: 可选过滤器，用于裁切搜索空间
            similarity_threshold: 相似度阈值，过滤掉低于此值的条目（默认 0.0 表示不过滤）
            
        Returns:
            List[RetrievalResult]: 检索结果列表，按相似度降序排列
        """
        if not self.text_unit_index:
            logger.error("TextUnit 索引未构建，请先调用 load_all_assets()")
            return []
        
        # 1. 对查询向量进行 L2 归一化
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_np)
        
        # 2. 生成过滤器掩码（如果有）
        if filter_query:
            mask = self.apply_filter(filter_query)
            # 计算通过过滤的条目数量
            valid_count = np.sum(mask)
            if valid_count == 0:
                logger.warning("过滤器导致零个有效条目，返回空结果")
                return []
            
            # 如果有效条目少于 top_k，调整搜索数量
            search_k = min(top_k * 5, len(self.text_unit_map))
        else:
            mask = None
            search_k = top_k * 5  # 搜索更多结果以便后续过滤
        
        # 3. 使用 FAISS 搜索
        # 注意：FAISS 不支持直接掩码，我们搜索更多结果然后过滤
        search_k = min(search_k, len(self.text_unit_map))
        distances, indices = self.text_unit_index.search(query_np, search_k)
        
        # 4. 处理结果
        results = []
        filtered_by_threshold = 0
        for i in range(search_k):
            faiss_id = indices[0, i]
            similarity = distances[0, i]  # 内积相似度，范围 [-1, 1]
            
            # 检查相似度阈值
            if similarity_threshold > 0 and similarity < similarity_threshold:
                filtered_by_threshold += 1
                continue
            
            # 检查掩码过滤
            if mask is not None and not mask[faiss_id]:
                continue
            
            # 通过 faiss_id 回查文本单元
            text_unit = self.get_text_unit_by_faiss_id(faiss_id)
            if text_unit is None:
                logger.warning(f"无法找到 FAISS ID {faiss_id} 对应的文本单元，跳过")
                continue
            
            # 构建 RetrievalResult
            result = RetrievalResult(
                text_unit=text_unit,
                similarity_score=float(similarity),
                source_type=SourceType.ANCHOR,  # 默认锚点类型，后续可由上层调整
                source_relation=None,
                anchor_community_id=None,
                anchor_community_title=None
            )
            results.append(result)
            
            # 如果已收集足够结果，停止
            if len(results) >= top_k:
                break
        
        if filtered_by_threshold > 0:
            logger.debug(f"相似度阈值过滤: {filtered_by_threshold} 条 (阈值={similarity_threshold})")
        
        logger.info(f"检索完成: 找到 {len(results)} 个结果 (请求 {top_k} 个, 阈值={similarity_threshold})")
        return results

    def get_text_unit_by_id(self, text_unit_id: str) -> Optional[TextUnit]:
        """根据 ID 获取文本单元"""
        return self.text_unit_map.get(text_unit_id)

    def get_community_by_id(self, community_id: str) -> Optional[Community]:
        """根据 ID 获取社区"""
        return self.community_map.get(community_id)

    def get_report_by_community_id(self, community_id: str) -> Optional[CommunityReport]:
        """根据社区 ID 获取报告"""
        return self.community_reports.get(community_id)

    def get_text_units_by_community_id(self, community_id: str) -> List[TextUnit]:
        """获取属于指定社区的所有文本单元"""
        return self.community_to_text_units.get(community_id, [])

    def is_loaded(self) -> bool:
        """检查数据是否已加载"""
        return self._loaded

    def embed_text(self, text: str) -> List[float]:
        """
        将文本转换为向量嵌入。

        使用 SiliconFlow API 调用嵌入模型（Qwen/Qwen3-Embedding-8B）。

        Args:
            text: 要嵌入的文本

        Returns:
            List[float]: 4096维向量
        """
        from openai import OpenAI
        from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, EMBEDDING_MODEL, EMBEDDING_TIMEOUT

        client = OpenAI(
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL,
            timeout=EMBEDDING_TIMEOUT
        )

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"嵌入文本失败: {e}")
            raise


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    dm = DataManager()
    dm.load_all_assets()
    print(f"DataManager 加载完成: {len(dm.text_unit_map)} 个文本单元, {len(dm.community_map)} 个社区")