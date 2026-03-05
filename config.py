"""
配置文件 - 用于存储API密钥和其他配置
建议将敏感信息存储在环境变量中，避免硬编码
使用 Pydantic BaseSettings 进行配置管理，支持环境变量覆盖
"""

import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings

# ==================== Pydantic 配置模型 ====================
class Settings(BaseSettings):
    """系统配置模型，支持环境变量覆盖"""
    
    # ==================== API 配置 ====================
    # SiliconFlow API 配置
    siliconflow_api_key: str = Field(
        default="<你的SiliconFlow API密钥>", # 请在此处填入你的API密钥，例如 "sk-..."
        description="SiliconFlow API 密钥",
        env="SILICONFLOW_API_KEY"
    )
    
    siliconflow_base_url: str = Field(
        default="<API服务提供商链接>", # 请在此处填入供应商链接，例如 "https://api.siliconflow.cn/v1"
        description="SiliconFlow API 基础 URL",
        env="SILICONFLOW_BASE_URL"
    )
    
    # ==================== 模型配置 ====================
    # LLM 模型
    llm_model: str = Field(
        default="Pro/deepseek-ai/DeepSeek-V3.2",
        description="LLM 模型名称",
        env="LLM_MODEL"
    )
    
    # LLM 温度
    llm_temperature: float = Field(
        default=0.1,
        description="LLM 温度参数，控制生成随机性",
        env="LLM_TEMPERATURE"
    )
    
    # LLM 最大令牌数
    llm_max_tokens: int = Field(
        default=8000,
        description="LLM 最大生成令牌数",
        env="LLM_MAX_TOKENS"
    )
    
    # LLM 超时时间（秒）
    llm_timeout: int = Field(
        default=120,
        description="LLM API 调用超时时间（秒）",
        env="LLM_TIMEOUT"
    )
    
    # 嵌入模型
    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-8B",
        description="嵌入模型名称",
        env="EMBEDDING_MODEL"
    )
    
    # 嵌入模型超时时间（秒）
    embedding_timeout: int = Field(
        default=60,
        description="嵌入模型 API 调用超时时间（秒）",
        env="EMBEDDING_TIMEOUT"
    )
    
    # ==================== 检索相关参数 ====================
    similarity_threshold: float = Field(
        default=0.4,
        description="向量相似度阈值，所有检索结果必须通过的第一道关卡",
        env="SIMILARITY_THRESHOLD"
    )
    
    global_char_budget: int = Field(
        default=32000,
        description="全局字符预算，用于上下文装填",
        env="GLOBAL_CHAR_BUDGET"
    )
    
    relation_weight_threshold: float = Field(
        default=0.7,
        description="关系权重阈值，仅信任 weight > 此值的高置信度边",
        env="RELATION_WEIGHT_THRESHOLD"
    )
    
    # ==================== Retriever 参数 (Micro Search) ====================
    graph_enable: bool = Field(
        default=True,
        description="是否启用横向图谱扩展",
        env="GRAPH_ENABLE"
    )
    
    anchor_quota_ratio: float = Field(
        default=0.4,
        description="锚点配额比例，分配给向量直接命中社区的预算比例",
        env="ANCHOR_QUOTA_RATIO"
    )
    
    neighbor_quota_ratio: float = Field(
        default=0.2,
        description="邻居配额比例，分配给图谱关联社区的独立预算比例",
        env="NEIGHBOR_QUOTA_RATIO"
    )
    
    force_recall_count: int = Field(
        default=20,
        description="语义保底数量，在任何情况下必须通过全局语义检索补充的最低条数",
        env="FORCE_RECALL_COUNT"
    )
    
    neighbor_fanout: int = Field(
        default=5,
        description="每个锚点允许扩展的最大邻居数量",
        env="NEIGHBOR_FANOUT"
    )
    
    # ==================== MacroRetriever 参数 ====================
    top_k_macro: int = Field(
        default=3,
        description="宏观锚点数，最初锁定的高层社区数量",
        env="TOP_K_MACRO"
    )
    
    bridge_fanout: int = Field(
        default=5,
        description="桥接扇出数，每个锚点允许寻找的关联社区数量",
        env="BRIDGE_FANOUT"
    )
    
    max_findings_per_query: int = Field(
        default=5,
        description="核心观点数，最终选送给 LLM 的高相关性 Findings 数量",
        env="MAX_FINDINGS_PER_QUERY"
    )
    
    max_evidence_per_finding: int = Field(
        default=6,
        description="观点例证数，为每个 Finding 搭配的原文数量",
        env="MAX_EVIDENCE_PER_FINDING"
    )
    
    force_recall_count_macro: int = Field(
        default=10,
        description="强制补全数，最后必须输出的全局语义匹配 TextUnits 数量",
        env="FORCE_RECALL_COUNT_MACRO"
    )
    
    # ==================== QueryProcessor 参数 ====================
    enable_query_enhancement: bool = Field(
        default=True,
        description="是否启用查询增强（将关键词附加到原始查询后）",
        env="ENABLE_QUERY_ENHANCEMENT"
    )
    
    # ==================== Processor 参数 ====================
    long_text_threshold: int = Field(
        default=700,
        description="长文阈值，超过此长度的文本将进行切片选择",
        env="LONG_TEXT_THRESHOLD"
    )
    
    # ==================== 文件路径配置 ====================
    # 输入文件
    input_file: str = Field(
        default="taipingyulan.csv",
        description="原始输入 CSV 文件",
        env="INPUT_FILE"
    )
    
    input_communities: str = Field(
        default="create_final_communities.parquet",
        description="社区 Parquet 文件",
        env="INPUT_COMMUNITIES"
    )
    
    input_text_units: str = Field(
        default="create_final_text_units.parquet",
        description="文本单元 Parquet 文件",
        env="INPUT_TEXT_UNITS"
    )
    
    input_reports: str = Field(
        default="create_final_community_reports.parquet",
        description="社区报告 Parquet 文件",
        env="INPUT_REPORTS"
    )
    
    # 输出文件
    output_communities: str = Field(
        default="create_final_communities.parquet",
        description="输出社区 Parquet 文件",
        env="OUTPUT_COMMUNITIES"
    )
    
    output_text_units: str = Field(
        default="create_final_text_units.parquet",
        description="输出文本单元 Parquet 文件",
        env="OUTPUT_TEXT_UNITS"
    )
    
    output_reports_jsonl: str = Field(
        default="create_final_community_reports.jsonl",
        description="输出社区报告 JSONL 文件",
        env="OUTPUT_REPORTS_JSONL"
    )
    
    output_reports_parquet: str = Field(
        default="create_final_community_reports.parquet",
        description="输出社区报告 Parquet 文件",
        env="OUTPUT_REPORTS_PARQUET"
    )
    
    output_text_units_updated: str = Field(
        default="create_final_text_units.parquet",
        description="更新后的文本单元 Parquet 文件",
        env="OUTPUT_TEXT_UNITS_UPDATED"
    )
    
    output_relationships_jsonl: str = Field(
        default="create_final_relationships.jsonl",
        description="输出关系 JSONL 文件",
        env="OUTPUT_RELATIONSHIPS_JSONL"
    )
    
    output_relationships_parquet: str = Field(
        default="create_final_relationships.parquet",
        description="输出关系 Parquet 文件",
        env="OUTPUT_RELATIONSHIPS_PARQUET"
    )
    
    # 中间文件
    vector_output_jsonl: str = Field(
        default="vectorization_vectors.jsonl",
        description="向量化输出 JSONL 文件",
        env="VECTOR_OUTPUT_JSONL"
    )
    
    vector_progress_jsonl: str = Field(
        default="vectorization_progress.jsonl",
        description="向量化进度 JSONL 文件",
        env="VECTOR_PROGRESS_JSONL"
    )
    
    checkpoint_file: str = Field(
        default="checkpoint_step3.jsonl",
        description="检查点文件",
        env="CHECKPOINT_FILE"
    )
    
    # ==================== 处理参数 ====================
    # 向量维度
    vector_dim: int = Field(
        default=4096,
        description="向量维度，Qwen Embedding 维度",
        env="VECTOR_DIM"
    )
    
    # 切片策略
    chunk_size: int = Field(
        default=300,
        description="切片长度",
        env="CHUNK_SIZE"
    )
    
    chunk_overlap: int = Field(
        default=100,
        description="重叠长度",
        env="CHUNK_OVERLAP"
    )
    
    chunk_step: int = Field(
        default=200,
        description="步长，自动计算为 chunk_size - chunk_overlap",
        env="CHUNK_STEP"
    )
    
    # Map-Reduce 策略
    map_reduce_threshold: int = Field(
        default=15000,
        description="Token 阈值，超过此值触发 Map-Reduce",
        env="MAP_REDUCE_THRESHOLD"
    )
    
    target_chunk_size: int = Field(
        default=12000,
        description="单段目标容量",
        env="TARGET_CHUNK_SIZE"
    )
    
    max_tokens_in_flight: int = Field(
        default=85000,
        description="动态令牌桶",
        env="MAX_TOKENS_IN_FLIGHT"
    )
    
    max_concurrent_net_reqs: int = Field(
        default=8,
        description="并发硬限制",
        env="MAX_CONCURRENT_NET_REQS"
    )
    
    # 图构建参数
    threshold_min: float = Field(
        default=0.60,
        description="低于此值不看",
        env="THRESHOLD_MIN"
    )
    
    threshold_max: float = Field(
        default=0.96,
        description="高于此值视为同义反复/模板废话，丢弃",
        env="THRESHOLD_MAX"
    )
    
    shadow_node_sim: float = Field(
        default=0.92,
        description="子类与父类相似度超过此值，子类被视为影子",
        env="SHADOW_NODE_SIM"
    )
    
    parent_child_diff: float = Field(
        default=0.00,
        description="父子竞争阈值",
        env="PARENT_CHILD_DIFF"
    )
    
    top_k_candidates: int = Field(
        default=50,
        description="每个节点最多送审 50 个",
        env="TOP_K_CANDIDATES"
    )
    
    max_concurrent_reqs: int = Field(
        default=16,
        description="并发请求数",
        env="MAX_CONCURRENT_REQS"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 创建全局配置实例
settings = Settings()

# ==================== 向后兼容的模块级变量 ====================
# 以下变量保持原有命名，以便现有代码无需修改即可使用
# API 配置
SILICONFLOW_API_KEY: str = settings.siliconflow_api_key
SILICONFLOW_BASE_URL: str = settings.siliconflow_base_url

# 模型配置
LLM_MODEL: str = settings.llm_model
LLM_TEMPERATURE: float = settings.llm_temperature
LLM_MAX_TOKENS: int = settings.llm_max_tokens
LLM_TIMEOUT: int = settings.llm_timeout
EMBEDDING_MODEL: str = settings.embedding_model
EMBEDDING_TIMEOUT: int = settings.embedding_timeout

# 检索相关参数
SIMILARITY_THRESHOLD: float = settings.similarity_threshold
GLOBAL_CHAR_BUDGET: int = settings.global_char_budget
RELATION_WEIGHT_THRESHOLD: float = settings.relation_weight_threshold

# Retriever 参数
GRAPH_ENABLE: bool = settings.graph_enable
ANCHOR_QUOTA_RATIO: float = settings.anchor_quota_ratio
NEIGHBOR_QUOTA_RATIO: float = settings.neighbor_quota_ratio
FORCE_RECALL_COUNT: int = settings.force_recall_count
NEIGHBOR_FANOUT: int = settings.neighbor_fanout

# MacroRetriever 参数
TOP_K_MACRO: int = settings.top_k_macro
BRIDGE_FANOUT: int = settings.bridge_fanout
MAX_FINDINGS_PER_QUERY: int = settings.max_findings_per_query
MAX_EVIDENCE_PER_FINDING: int = settings.max_evidence_per_finding
FORCE_RECALL_COUNT_MACRO: int = settings.force_recall_count_macro

# QueryProcessor 参数
ENABLE_QUERY_ENHANCEMENT: bool = settings.enable_query_enhancement

# Processor 参数
LONG_TEXT_THRESHOLD: int = settings.long_text_threshold

# 文件路径配置
INPUT_FILE: str = settings.input_file
INPUT_COMMUNITIES: str = settings.input_communities
INPUT_TEXT_UNITS: str = settings.input_text_units
INPUT_REPORTS: str = settings.input_reports

OUTPUT_COMMUNITIES: str = settings.output_communities
OUTPUT_TEXT_UNITS: str = settings.output_text_units
OUTPUT_REPORTS_JSONL: str = settings.output_reports_jsonl
OUTPUT_REPORTS_PARQUET: str = settings.output_reports_parquet
OUTPUT_TEXT_UNITS_UPDATED: str = settings.output_text_units_updated
OUTPUT_RELATIONSHIPS_JSONL: str = settings.output_relationships_jsonl
OUTPUT_RELATIONSHIPS_PARQUET: str = settings.output_relationships_parquet

VECTOR_OUTPUT_JSONL: str = settings.vector_output_jsonl
VECTOR_PROGRESS_JSONL: str = settings.vector_progress_jsonl
CHECKPOINT_FILE: str = settings.checkpoint_file

# 处理参数
VECTOR_DIM: int = settings.vector_dim
CHUNK_SIZE: int = settings.chunk_size
CHUNK_OVERLAP: int = settings.chunk_overlap
CHUNK_STEP: int = settings.chunk_step

MAP_REDUCE_THRESHOLD: int = settings.map_reduce_threshold
TARGET_CHUNK_SIZE: int = settings.target_chunk_size
MAX_TOKENS_IN_FLIGHT: int = settings.max_tokens_in_flight
MAX_CONCURRENT_NET_REQS: int = settings.max_concurrent_net_reqs

THRESHOLD_MIN: float = settings.threshold_min
THRESHOLD_MAX: float = settings.threshold_max
SHADOW_NODE_SIM: float = settings.shadow_node_sim
PARENT_CHILD_DIFF: float = settings.parent_child_diff
TOP_K_CANDIDATES: int = settings.top_k_candidates
MAX_CONCURRENT_REQS: int = settings.max_concurrent_reqs

# ==================== 辅助函数 ====================
def validate_config() -> None:
    """验证配置是否有效"""
    if not SILICONFLOW_API_KEY or SILICONFLOW_API_KEY.startswith("sk-"):
        # 这里可以添加更严格的验证
        pass
    
    # 检查必要的目录是否存在
    required_files = [INPUT_FILE]
    for file in required_files:
        if not os.path.exists(file):
            print(f"警告: 输入文件 {file} 不存在")

# 在导入时自动验证
if __name__ != "__main__":
    validate_config()