"""
《太平御览》GraphRAG 系统数据模型定义

本模块定义了系统核心数据结构，严格遵循 project.md 1.1 节（数据资产字典）和 2.1 节（DataManager）的规范。
所有模型使用 Pydantic v2 进行验证，确保类型安全。

注意：
- 向量字段使用 List[float] 表示，维度为 4096（Qwen/BGE 标准）
- JSON 字符串字段在需要时会进行解析
- 所有字段名称与 Parquet 文件中的列名保持一致
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


# ==================== 枚举类型 ====================

class RelationType(str, Enum):
    """关系类型枚举，基于 create_final_relationships_refined.parquet 中的 relation_type 字段"""
    PORTENDS = "Portends"           # 预兆
    CORRESPONDS_TO = "CorrespondsTo" # 对应
    RESPONDS_TO = "RespondsTo"      # 响应
    INFLUENCES = "Influences"       # 影响
    SIMILAR_TO = "SimilarTo"        # 相似
    PART_OF = "PartOf"              # 部分
    CAUSES = "Causes"               # 导致
    OTHER = "Other"                 # 其他


class RetrievalMode(str, Enum):
    """检索模式枚举"""
    MICRO = "micro"      # 微观考据模式
    MACRO = "macro"      # 宏观综述模式


class TextUnitType(str, Enum):
    """文本单元类型"""
    FULL_TEXT = "full_text"      # 完整文本
    CHUNK = "chunk"              # 切片文本
    SUMMARY = "summary"          # 社区摘要（用于非叶子节点）


class SourceType(str, Enum):
    """检索结果来源类型"""
    ANCHOR = "anchor"              # 锚点社区直接命中
    NEIGHBOR = "neighbor"          # 图谱关联社区
    FALLBACK = "fallback"          # 全局语义兜底
    FINDING = "finding"            # 宏观模式的观点
    EVIDENCE = "evidence"          # 观点的例证
    GLOBAL_SUPPLEMENT = "global_supplement"  # 宏观模式的全局补充


# ==================== 核心数据模型 ====================

class TextUnit(BaseModel):
    """
    核心史料库 - 对应 create_final_text_units.parquet 结构
    
    存储《太平御览》的原始文献切片及预计算索引。
    """
    id: str = Field(..., description="唯一标识符 (UUID)")
    text: str = Field(..., description="原始文本内容，检索的最终目标")
    n_tokens: int = Field(..., description="Token 数量，用于计算 Context Window 预算")
    vector: List[float] = Field(..., description="核心向量索引 (4096维)，对应全文或 Search Text 的嵌入")
    community_id: str = Field(..., description="归属社区 ID，外键，指向 communities 表")
    text_chunks: List[str] = Field(default=[], description="长文切片，针对长文预先切分好的段落列表")
    chunk_vectors: List[List[float]] = Field(default=[], description="切片向量列表，对应 text_chunks 的向量")
    head: str = Field(default="", description="头部预计算 (前 ~200 字符)，用于快速构建上下文预览")
    tail: str = Field(default="", description="尾部预计算 (后 ~100 字符)，用于快速构建上下文预览")
    hierarchy_path: str = Field(default="", description="层级路径/标题 (e.g., '天部 > 日')，替代原 title 字段")
    source_metadata: str = Field(default="", description="元数据字符串，包含卷数、类别等原始来源信息")
    prepend_source: str = Field(default="", description="预格式化的来源引注字符串，可直接拼接到 LLM Prompt 中")
    
    # 内部字段（不在 Parquet 中，但用于系统处理）
    faiss_id: Optional[int] = Field(default=None, description="FAISS 索引中的物理行号，系统内部使用", exclude=True)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "《太平御览》卷三引《淮南子》曰：日者，阳之精也。",
                "n_tokens": 25,
                "vector": [0.1, 0.2, ...],  # 4096维
                "community_id": "community_001",
                "text_chunks": ["《太平御览》卷三引《淮南子》曰：日者，阳之精也。"],
                "chunk_vectors": [[0.1, 0.2, ...]],
                "head": "《太平御览》卷三引《淮南子》曰：日者",
                "tail": "阳之精也。",
                "hierarchy_path": "天部 > 日",
                "source_metadata": "{\"卷\": 3, \"类别\": \"天部\"}",
                "prepend_source": "[天部·日]"
            }
        }
    )


class Community(BaseModel):
    """
    层级与路由表 - 对应 create_final_communities.parquet 结构
    
    定义知识的树状分类体系及节点属性。
    """
    id: str = Field(..., description="社区唯一标识符")
    title: str = Field(..., description="社区名称（如'天部'、'日'）")
    level: int = Field(..., description="层级深度 (0: Root, 1: 部, 2: 门, 3: 子目)")
    parent_id: str = Field(default="", description="父社区 ID")
    child_ids: List[str] = Field(default=[], description="子社区 ID 列表")
    is_leaf: bool = Field(..., description="核心路由标志，True 表示该社区是挂载 TextUnits 的最底层单元")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "community_001",
                "title": "日",
                "level": 3,
                "parent_id": "community_002",
                "child_ids": [],
                "is_leaf": True
            }
        }
    )


class Relationship(BaseModel):
    """
    图谱关联层 - 对应 create_final_relationships_refined.parquet 结构
    
    存储经过拓扑剪枝后的高质量横向关联。
    """
    source_id: str = Field(..., description="起点 ID")
    target_id: str = Field(..., description="终点 ID")
    weight: float = Field(..., description="关联强度 (0-1)，由 LLM 打分经过拓扑剪枝保留的权重")
    relation_type: str = Field(..., description="关系类型 (e.g., 'Portends', 'CorrespondsTo')")
    description: str = Field(default="", description="关联描述，解释两个节点关联的具体逻辑")
    vector_score: float = Field(default=0.0, description="原始向量相似度，仅作为参考数据")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_id": "community_001",
                "target_id": "community_002",
                "weight": 0.85,
                "relation_type": "Portends",
                "description": "日食预兆国家有难",
                "vector_score": 0.78
            }
        }
    )


class FindingItem(BaseModel):
    """Finding 条目结构，对应 findings JSON 中的单个条目"""
    summary: str = Field(..., description="发现摘要")
    explanation: str = Field(..., description="详细解释")


class CommunityReport(BaseModel):
    """
    摘要与索引层 - 对应 create_final_community_reports.parquet 结构
    
    存储 LLM 生成的社区综述与洞察，用于宏观检索。
    """
    community_id: str = Field(..., description="主键，对应 communities 表")
    title: str = Field(..., description="社区标题")
    level: int = Field(..., description="冗余字段，方便快速过滤不同层级的报告")
    summary: str = Field(..., description="社区综述，用于 Macro 模式的直接回答")
    findings: str = Field(default="", description="深度洞察，JSON 字符串格式的列表，包含 summary 和 explanation")
    full_content: str = Field(default="", description="完整报告，包含 Markdown 格式的标题、Summary 和 Findings")
    embedding: List[float] = Field(..., description="向量索引 (4096维)，用于 FAISS 构建 Report_Index")
    finding_vectors: List[List[float]] = Field(default=[], description="预计算的 Finding 向量列表，与 findings 列表一一对应")
    
    # 计算属性
    @property
    def parsed_findings(self) -> List[FindingItem]:
        """解析 findings JSON 字符串为 FindingItem 列表"""
        import json
        if not self.findings or self.findings.strip() == "":
            return []
        try:
            data = json.loads(self.findings)
            if isinstance(data, list):
                return [FindingItem(**item) for item in data]
            else:
                return []
        except json.JSONDecodeError:
            return []
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "community_id": "community_001",
                "title": "日",
                "level": 3,
                "summary": "日者，阳之精，君之象也。",
                "findings": '[{"summary": "日食预兆", "explanation": "日食通常被视为君主失德或国家有难的预兆"}]',
                "full_content": "# 日\n\n## Summary\n\n日者，阳之精...",
                "embedding": [0.1, 0.2, ...],
                "finding_vectors": [[0.1, 0.2, ...]]
            }
        }
    )


# ==================== 查询与过滤模型 ====================

class FilterQuery(BaseModel):
    """
    过滤器查询结构 - 用于在向量检索之前或之中对搜索空间进行布尔掩码裁切
    
    支持复合逻辑结构，详见 project.md 2.1 节 D 部分。
    """
    must_contain: List[str] = Field(default=[], description="AND 逻辑：必须同时包含这些词")
    any_contain: List[str] = Field(default=[], description="OR 逻辑：包含任意一个即可")
    must_not_contain: List[str] = Field(default=[], description="NOT 逻辑：排除包含此词的条目")
    scope_hierarchy: List[str] = Field(default=[], description="层级限制：只在指定层级路径前缀下搜索")
    community_ids: List[str] = Field(default=[], description="社区限制：只在指定社区 ID 列表中搜索")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "must_contain": ["日", "月"],
                "any_contain": ["神话", "典故"],
                "must_not_contain": ["俗"],
                "scope_hierarchy": ["天部"],
                "community_ids": ["community_001", "community_002"]
            }
        }
    )


# ==================== 检索结果模型 ====================

class RetrievalResult(BaseModel):
    """
    检索结果 - Retriever 返回的原始结果
    
    包含完整的 TextUnit 信息以及检索相关的元数据。
    """
    text_unit: TextUnit = Field(..., description="检索到的文本单元")
    similarity_score: float = Field(..., description="向量相似度分数")
    source_type: SourceType = Field(..., description="来源类型")
    source_relation: Optional[Relationship] = Field(default=None, description="如果是邻居扩展，关联的关系信息")
    anchor_community_id: Optional[str] = Field(default=None, description="如果是邻居扩展，对应的锚点社区 ID")
    anchor_community_title: Optional[str] = Field(default=None, description="如果是邻居扩展，对应的锚点社区标题")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text_unit": TextUnit(
                    id="123e4567-e89b-12d3-a456-426614174000",
                    text="《太平御览》卷三引《淮南子》曰：日者，阳之精也。",
                    n_tokens=25,
                    vector=[0.1, 0.2],
                    community_id="community_001"
                ),
                "similarity_score": 0.85,
                "source_type": "anchor",
                "source_relation": None,
                "anchor_community_id": None,
                "anchor_community_title": None
            }
        }
    )


class ProcessedResult(BaseModel):
    """
    处理结果 - Processor 处理后的结果
    
    包含截断后的文本、高亮元数据，用于最终 Context 装填。
    """
    text_unit_id: str = Field(..., description="原始 TextUnit ID")
    processed_text: str = Field(..., description="处理后的文本（可能被截断或重组）")
    original_length: int = Field(..., description="原始文本长度")
    processed_length: int = Field(..., description="处理后文本长度")
    text_unit_type: TextUnitType = Field(..., description="文本类型：完整文本、切片或摘要")
    highlight_spans: List[Dict[str, Any]] = Field(default=[], description="高亮片段位置信息")
    metadata: Dict[str, Any] = Field(default={}, description="附加元数据")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text_unit_id": "123e4567-e89b-12d3-a456-426614174000",
                "processed_text": "[天部·日] 《太平御览》卷三引《淮南子》曰：日者，阳之精也。",
                "original_length": 1000,
                "processed_length": 500,
                "text_unit_type": TextUnitType.FULL_TEXT,
                "highlight_spans": [{"start": 10, "end": 20, "type": "keyword"}],
                "metadata": {"hierarchy_path": "天部 > 日"}
            }
        }
    )


# ==================== 检索参数模型 ====================

class MicroRetrievalParams(BaseModel):
    """微观考据模式检索参数"""
    graph_enable: bool = Field(default=True, description="是否启用横向图谱扩展")
    anchor_quota_ratio: float = Field(default=0.3, description="锚点配额比例，分配给向量直接命中社区的预算比例")
    neighbor_quota_ratio: float = Field(default=0.3, description="邻居配额比例，分配给图谱关联社区的独立预算比例")
    force_recall_count: int = Field(default=20, description="语义保底数量，必须通过全局语义检索补充的最低条数")
    neighbor_fanout: int = Field(default=5, description="每个锚点允许扩展的最大邻居数量")
    similarity_threshold: float = Field(default=0.65, description="相似度阈值，所有向量检索结果必须通过的第一道关卡")
    relation_weight_threshold: float = Field(default=0.8, description="关系权重阈值，仅信任 weight > 此值的高置信度边")


class MacroRetrievalParams(BaseModel):
    """宏观综述模式检索参数"""
    top_k_macro: int = Field(default=3, description="宏观锚点数，最初锁定的高层社区数量")
    bridge_fanout: int = Field(default=5, description="桥接扇出数，每个锚点允许寻找的关联社区数量")
    max_findings_per_query: int = Field(default=5, description="核心观点数，最终选送给 LLM 的高相关性 Findings 数量")
    max_evidence_per_finding: int = Field(default=2, description="观点例证数，为每个 Finding 搭配的原文数量")
    force_recall_count_macro: int = Field(default=10, description="强制补全数，最后必须输出的全局语义匹配 TextUnits 数量")
    similarity_threshold: float = Field(default=0.65, description="相似度阈值")


# ==================== 系统配置模型 ====================

class SystemConfig(BaseModel):
    """系统全局配置"""
    global_char_budget: int = Field(default=32000, description="全局字符预算，可配置项，预设 32,000 字符")
    embedding_dim: int = Field(default=4096, description="向量维度，Qwen/BGE 标准")
    faiss_index_type: str = Field(default="IndexFlatIP", description="FAISS 索引类型，使用内积索引")
    long_text_threshold: int = Field(default=1000, description="长文阈值，超过此长度进入切片选择流程")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "global_char_budget": 32000,
                "embedding_dim": 4096,
                "faiss_index_type": "IndexFlatIP",
                "long_text_threshold": 1000
            }
        }
    )


# ==================== 辅助函数 ====================

def text_unit_to_dict(text_unit: TextUnit) -> Dict[str, Any]:
    """将 TextUnit 转换为字典，用于序列化"""
    return text_unit.model_dump()


def dict_to_text_unit(data: Dict[str, Any]) -> TextUnit:
    """从字典创建 TextUnit"""
    return TextUnit(**data)


# ==================== 类型别名 ====================

Vector = List[float]
Matrix = List[List[float]]
CommunityMap = Dict[str, Community]
RelationshipMap = Dict[str, List[Relationship]]
NodeToRootMap = Dict[str, str]  # 用于 Root Lineage Mapping


if __name__ == "__main__":
    # 简单测试
    print("数据模型定义验证通过")
    print(f"TextUnit 字段数: {len(TextUnit.model_fields)}")
    print(f"Community 字段数: {len(Community.model_fields)}")
    print(f"Relationship 字段数: {len(Relationship.model_fields)}")
    print(f"CommunityReport 字段数: {len(CommunityReport.model_fields)}")