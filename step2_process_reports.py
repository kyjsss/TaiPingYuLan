import pandas as pd
import json
import re
import os
import asyncio
import math
import zhconv  # 核心新增：用于实时繁简转换
from openai import AsyncOpenAI
from collections import defaultdict
import json_repair
import numpy as np
import config  # 导入配置文件

# --- 配置区 ---
# 从配置文件读取
API_KEY = config.SILICONFLOW_API_KEY
BASE_URL = config.SILICONFLOW_BASE_URL

# 模型配置
LLM_MODEL = config.LLM_MODEL
EMBEDDING_MODEL = config.EMBEDDING_MODEL

# 文件路径
INPUT_COMMUNITIES = config.INPUT_COMMUNITIES
INPUT_TEXT_UNITS = config.INPUT_TEXT_UNITS
OUTPUT_REPORTS_JSONL = config.OUTPUT_REPORTS_JSONL
OUTPUT_REPORTS_PARQUET = config.OUTPUT_REPORTS_PARQUET
OUTPUT_TEXT_UNITS_UPDATED = config.OUTPUT_TEXT_UNITS_UPDATED
VECTOR_OUTPUT_JSONL = config.VECTOR_OUTPUT_JSONL

# Map-Reduce 策略配置
MAP_REDUCE_THRESHOLD = config.MAP_REDUCE_THRESHOLD
TARGET_CHUNK_SIZE = config.TARGET_CHUNK_SIZE
MAX_TOKENS_IN_FLIGHT = config.MAX_TOKENS_IN_FLIGHT
MAX_CONCURRENT_NET_REQS = config.MAX_CONCURRENT_NET_REQS

# --- 辅助函数 ---
def count_tokens(text):
    """计算 Token 数量"""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        return len(text)

def is_placeholder_vector(vec):
    """检测向量是否为占位数据（全零或无效）"""
    if vec is None:
        return True
    # 处理 numpy.ndarray
    if isinstance(vec, np.ndarray):
        if vec.size == 0:
            return True
        # 检查是否全为零（或接近零）
        # 使用 bool 转换确保返回 Python bool 类型
        return bool(np.all(np.abs(vec) < 1e-7))
    # 处理列表
    if isinstance(vec, list):
        if len(vec) == 0:
            return True
        # 检查是否全为零
        # 使用 all 并转换为 bool
        return bool(all(abs(v) < 1e-7 for v in vec))
    # 其他类型视为无效
    return True

# --- Prompts ---

# 1. 叶子节点 - 标准模式 (> 200 Tokens)
PROMPT_STANDARD = """
你是一位历史文献学家，你的任务是根据提供的相关文献，写一份既有学术价值又能支持精准检索的综述。
当前处理的相关文献是：【{path_str}】（仅供背景参考，综述中非必要不要提到父节点，保持独立性）

请仔细研读以下原始文献片段：
{context_text}

任务目标：
1. **Summary (摘要)**：
   - **定义与定性**：开门见山地定义该主题（Community Title）在古代知识体系中的核心含义。
   - 必须涵盖：关键人物/地名、以及文献中描述的主要事件或现象。
   - 不要流水账式说明都有哪些文献，但适当的时候可以做分析，如经史类文献（政治/礼制语境）与志怪/方术类文献（神话/实用语境）的描述是否有别分析不同语境下的差异。如果不同来源对同一事实有截然相反的记载，请明确指出这种矛盾。
   - 可选：是否存在时代演变（如先秦至汉唐的观念变化）
   - 语言风格：简练、客观、学术化。 
   - 关键内容要指出出处（如“据《史记》记载...”），但不必每句都标注。

2. **Findings (洞察)**：
   - 提取 **3-5 个** 具体的知识图谱洞察点。如五行生克、天人感应、礼制变迁、兵法策略、历史因果。
   - 每个洞察必须有原文支撑，**严禁脑补**。

**严禁**输出任何开场白、自我介绍或任务复述，严禁包含 Markdown 代码块标记，直接按格式要求陈述内容。严禁编造外部知识，所有论述必须基于给定的 Text Units。
- Summary 字数严格控制在 **300-600字** 之间。超过 800 字会导致向量检索精度下降，请务必精炼。
- Findings 数量控制在 **3-5 条**。每条 "explanation" 必须在 **50字以内**，一针见血。
- 必须使用 **简体中文**。JSON 字符串内部如果需要引用，请严格使用中文引号，严禁在 value 内部使用未转义的双引号（"）。
- 所有文献都不晚于宋初，请论述时不要犯时空上的低级错误。
返回格式 (JSON)：
{{
    "summary": "...",
    "findings": [
        {{"summary": "洞察标题", "explanation": "详细解释..."}}
    ]
}}
"""

# 2. 叶子节点 - 贫瘠模式 (< 200 Tokens)
PROMPT_SPARSE = """
你是一位严谨的历史文献学家和校勘学家，你的任务是根据提供的相关文献，写一份既有学术价值又能支持精准检索的综述。
当前处理的相关文献是：【{path_str}】（仅供背景参考，综述中非必要不要提到父节点，保持独立性）
相关文献内容**非常简短**：
{context_text}

任务目标：
1. **Summary (摘要)**：
   - **定义与定性**：开门见山地定义该主题（Community Title）在古代知识体系中的核心含义。
   - 由于原文简短，**请勿过度发挥**。可以仅对原文进行**现代汉语意译**或**概念解释**。
   -关键内容要指出出处（如“据《史记》记载...”）
   - 字数控制在 **100字以内**。

2. **Findings (洞察)**：
   - **严格检查**：原文是否包含足够的逻辑以支持深层洞察？
   - 如果原文仅为定义或简单记录，Findings 数组请返回 **空列表** []。
   - 只有当确有明确的因果/五行/分类逻辑时，才允许提取 1-2个洞察，且每条 "explanation" 字数应控制在 **50字以内**。

输出为简体中文，**严禁**输出任何开场白、自我介绍或任务复述，严禁包含 Markdown 代码块标记，直接按格式要求陈述内容。严禁编造外部知识，所有论述必须基于给定的 Text Units。
- 所有文献都不晚于宋初，请论述时不要犯时空上的低级错误。
- JSON 字符串内部如果需要引用，请严格使用中文引号，严禁在 value 内部使用未转义的双引号（"）。
返回格式 (JSON)：
{{
    "summary": "...",
    "findings": [
        {{"summary": "洞察标题", "explanation": "详细解释..."}}
    ]
}}
"""

# 3. Map-Reduce 分段 Prompt
PROMPT_MAP_PARTIAL = """
你是一位在进行于古籍数据化的历史文献学家。
当前任务是对【{path_str}】（仅供背景参考，非必要不要提到父节点，保持独立性）相关文献的长文本进行**分段预处理**（当前为第 {part_idx} 部分）。

原始文献片段：
{context_text}

任务目标：
请将这段文献转化为一份**高密度的信息摘要**，为后续的最终报告合成提供详实的原料。

请执行以下步骤：
1. **Partial Summary (分段摘要)**：
   - 必须涵盖：核心主题、关键人物/地名、以及文献中描述的主要事件或现象。
   - 不要流水账式说明都有哪些文献，但适当的时候可以做分析，如经史类文献（政治/礼制语境）与志怪/方术类文献（神话/实用语境）的描述是否有别分析不同语境下的差异。如果不同来源对同一事实有截然相反的记载，请明确指出这种矛盾。
   - 可选：是否存在时代演变（如先秦至汉唐的观念变化）
   - 关键内容要指出出处（如“据《史记》记载...”），但不必每句都标注。
   - 风格要求：紧凑、客观，不要遗漏重要细节。

2. **Partial Findings (分段发现线索)**：
   - 提取 3-5 个**有潜力的洞察线索**。
   - 格式要求：以“观点+证据”的形式记录。例如：“五行关联：文中多次提到北方与水的对应关系”。
   - 此为中间素材，需尽可能捕捉深层逻辑（五行、灾异、兵法、治国），以供后续筛选。

输出为简体中文，**严禁**输出任何开场白、自我介绍或任务复述，严禁包含 Markdown 代码块标记，直接按格式要求陈述内容。
- 所有文献都不晚于宋初，请论述时不要犯时空上的低级错误。
- JSON 字符串内部如果需要引用，请严格使用中文引号，严禁在 value 内部使用未转义的双引号（"）。
-返回格式 (JSON)：
{{
    "partial_summary": "...",
    "partial_findings": ["...", "..."]
}}    
"""

PROMPT_REDUCE_FINAL = """
你是一位历史文献学家，你的任务是为相关文献写一份既有学术价值又能支持精准检索的社区综述。
当前处理的相关文献是：【{path_str}】。（仅供背景参考，综述中非必要不要提到父节点，保持独立性）

我们已经将超长文献切分处理，并生成了以下分段摘要（Partial Summaries）和线索：

{context_text}

任务目标：
你需要将上述碎片化的信息，**重铸**为一份逻辑严密、学术价值极高的最终社区报告。

请执行以下步骤：
1. **Summary (最终综述 )**：
   - **定义与定性**：开门见山地定义该主题（Community Title）在古代知识体系中的核心含义。
   - **逻辑重组**：不要简单拼接分段摘要！请打通段落界限，按主题逻辑（如“定义-源流-事件-影响”）重新组织内容。
   - **去重与融合**：合并重复引用的典籍或重复描述的事件，形成连贯的叙事。


2. **Findings (最终洞察)**：
   - **深度筛选**：从分段线索中，提炼出 **3-5 个** 最具学术深度的洞察。
   - **合并同类项**：例如多个分段都提到了“五行”，请将其合并为一个名为“五行宇宙观”的深度洞察。
   - **结构化输出**：每个 Findings 必须包含 `summary` (精炼标题) 和 `explanation` (基于原文的详细论证)。

**严禁**输出任何开场白、自我介绍或任务复述，严禁包含 Markdown 代码块标记，直接按格式要求陈述内容。
- Summary 字数严格控制在 **300-600字** 之间。超过 800 字会导致向量检索精度下降，请务必精炼。
- Findings 数量控制在 **3-5 条**。每条 "explanation" 必须在 **50字以内**，一针见血。
- 必须使用 **简体中文**。JSON 字符串内部如果需要引用，请严格使用中文引号，严禁在 value 内部使用未转义的双引号（"）。
- 所有文献都不晚于宋初，请论述时不要犯时空上的低级错误。
-返回格式 (JSON)：
{{
    "summary": "...",
    "findings": [
        {{"summary": "洞察标题", "explanation": "详细解释..."}}
    ]
}}
"""

# 4. Level 0 (部) Prompt
PROMPT_LEVEL0 = """
你是一位非常了解中国古代知识分类学（Taxonomy）的历史文献学家。你的任务是为相关部类写一份既有学术价值又能支持精准检索的社区综述。
当前处理的是部类（Level 0）：【{path_str}】。

该部类下属所有“子目”的摘要汇总：
{context_text}

任务目标：
请跳出具体的细节，从**知识本体论（Ontology）**的高度，撰写该部类的宏观综述。

请执行以下步骤：
1. **Summary (宏观综述 - 简体中文)**：
   - **核心定义**：阐述该“部”在古代知识体系中的定义与地位。
   - **分类逻辑**：分析该部类下属子目是如何组织的？
   - **内容概览**：高度概括下属子目的核心议题，避免罗列式流水账。
   - 不要流水账式说明都有哪些文献，但适当的时候可以做分析，如经史类文献（政治/礼制语境）与志怪/方术类文献（神话/实用语境）的描述是否有别分析不同语境下的差异。如果不同来源对同一事实有截然相反的记载，请简单指出这种矛盾。
   - 可选：存在的时代演变（如先秦至汉唐的观念变化）
   - 语言风格：简练、客观、学术化。 
   - 关键内容要指出出处（如“据《史记》记载...”），但不必每句都标注。

2. **Findings (宏观洞察)**：
   - 提取跨子类的**高维共性**。
   - 关注以下维度：
     - **内容覆盖**：列出主要的实体类别。（如：包含兵器、阵法、攻守器械三类）。
     - **[地理/时间分布]**：如果子摘要中体现了明显的时空特征，请归纳。（如：多引汉代经学文献，或多引南方异物志）。
     - **宇宙图景**：该部类构建了怎样的古代世界观？

输出为简体中文，**严禁**输出任何开场白、自我介绍或任务复述，严禁包含 Markdown 代码块标记，直接按格式要求陈述内容。
字数控制在 **600-900字** 之间。**不要超过 1000 字**，以免稀释向量特征。
数量 3-5 条，每条 "explanation" 必须在 **75字以内**，一针见血。
- 所有文献都不晚于宋初，请论述时不要犯时空上的低级错误。
- JSON 字符串内部如果需要引用，请严格使用中文引号，严禁在 value 内部使用未转义的双引号（"）。
返回格式 (JSON)：
{{
    "summary": "...",
    "findings": [
        {{"summary": "洞察标题", "explanation": "详细解释..."}}
    ]
}}
"""

# 5. Level 0 Map-Reduce 分段 Prompt (Map 阶段)
PROMPT_LEVEL0_MAP_PARTIAL = """
你是一位在进行于古籍数据化的历史文献学家。
当前任务是对部类【{path_str}】下属子目摘要的长文本进行**分段预处理**（当前为第 {part_idx} 部分）。

该部分的子目摘要片段：
{context_text}

任务目标：
请将这段摘要转化为一份**高密度的信息摘要**，为后续的最终部类综述合成提供详实的原料。

请执行以下步骤：
1. **Partial Summary (分段摘要)**：
   - 必须涵盖：核心主题、关键人物/地名、以及文献中描述的主要事件或现象。
   - 不要流水账式说明都有哪些文献，但适当的时候可以做分析，如经史类文献（政治/礼制语境）与志怪/方术类文献（神话/实用语境）的描述是否有别分析不同语境下的差异。如果不同来源对同一事实有截然相反的记载，请明确指出这种矛盾。
   - 可选：是否存在时代演变（如先秦至汉唐的观念变化）
   - 关键内容要指出出处（如“据《史记》记载...”），但不必每句都标注。
   - 风格要求：紧凑、客观，不要遗漏重要细节。

2. **Partial Findings (分段发现线索)**：
   - 提取 3-5 个**有潜力的洞察线索**。
   - 格式要求：以“观点+证据”的形式记录。例如：“五行关联：文中多次提到北方与水的对应关系”。
   - 此为中间素材，需尽可能捕捉深层逻辑（五行、灾异、兵法、治国），以供后续筛选。

输出为简体中文，**严禁**输出任何开场白、自我介绍或任务复述，严禁包含 Markdown 代码块标记，直接按格式要求陈述内容。
- 所有文献都不晚于宋初，请论述时不要犯时空上的低级错误。
- JSON 字符串内部如果需要引用，请严格使用中文引号，严禁在 value 内部使用未转义的双引号（"）。
返回格式 (JSON)：
{{
    "partial_summary": "...",
    "partial_findings": ["...", "..."]
}}    
"""

# 6. Level 0 Map-Reduce 分段 Prompt (Reduce 阶段)
PROMPT_LEVEL0_REDUCE_FINAL = """
你是一位非常了解中国古代知识分类学（Taxonomy）的历史文献学家。你的任务是为相关部类写一份既有学术价值又能支持精准检索的社区综述。
当前处理的是部类（Level 0）：【{path_str}】。

我们已经将超长子目摘要切分处理，并生成了以下分段摘要（Partial Summaries）和线索：

{context_text}

任务目标：
你需要将上述碎片化的信息，**重铸**为一份逻辑严密、学术价值极高的最终部类报告。

请执行以下步骤：
1. **Summary (最终综述)**：
   - **核心定义**：阐述该“部”在古代知识体系中的定义与地位。
   - **分类逻辑**：分析该部类下属子目是如何组织的？
   - **内容概览**：高度概括下属子目的核心议题，避免罗列式流水账。
   - 不要流水账式说明都有哪些文献，但适当的时候可以做分析，如经史类文献（政治/礼制语境）与志怪/方术类文献（神话/实用语境）的描述是否有别分析不同语境下的差异。如果不同来源对同一事实有截然相反的记载，请简单指出这种矛盾。
   - 可选：存在的时代演变（如先秦至汉唐的观念变化）
   - 语言风格：简练、客观、学术化。 
   - 关键内容要指出出处（如“据《史记》记载...”），但不必每句都标注。

2. **Findings (最终洞察)**：
   - **深度筛选**：从分段线索中，提炼出 **3-5 个** 最具学术深度的洞察。
   - **合并同类项**：例如多个分段都提到了“五行”，请将其合并为一个名为“五行宇宙观”的深度洞察。
   - **结构化输出**：每个 Findings 必须包含 `summary` (精炼标题) 和 `explanation` (基于原文的详细论证)。

输出为简体中文，**严禁**输出任何开场白、自我介绍或任务复述，严禁包含 Markdown 代码块标记，直接按格式要求陈述内容。
字数控制在 **600-900字** 之间。**不要超过 1000 字**，以免稀释向量特征。
数量 3-5 条，每条 "explanation" 必须在 **75字以内**，一针见血。
- 所有文献都不晚于宋初，请论述时不要犯时空上的低级错误。
- JSON 字符串内部如果需要引用，请严格使用中文引号，严禁在 value 内部使用未转义的双引号（"）。
返回格式 (JSON)：
{{
    "summary": "...",
    "findings": [
        {{"summary": "洞察标题", "explanation": "详细解释..."}}
    ]
}}
"""

# --- 初始化 ---
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- 辅助类：动态令牌桶调度器 ---
class TokenScheduler:
    def __init__(self, max_tokens, max_reqs):
        self.max_tokens = max_tokens
        self.max_reqs = max_reqs
        self.current_tokens = 0
        self.current_reqs = 0
        self.condition = asyncio.Condition()

    async def acquire(self, estimated_tokens):
        async with self.condition:
            while True:
                if self.current_reqs < self.max_reqs:
                    if self.current_tokens == 0 or (self.current_tokens + estimated_tokens <= self.max_tokens):
                        self.current_reqs += 1
                        self.current_tokens += estimated_tokens
                        return
                await self.condition.wait()

    async def release(self, estimated_tokens):
        async with self.condition:
            self.current_reqs -= 1
            self.current_tokens -= estimated_tokens
            self.condition.notify_all()

# --- 核心逻辑函数 ---

async def get_embedding(text, max_retries=3):
    """获取向量，带指数退避重试"""
    import asyncio
    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            # BGE-M3 建议截断，但 search_text 一般不超长
            text = text[:8000]
            resp = await client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return resp.data[0].embedding
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Embedding Error after {max_retries} retries: {e}")
                return [0.0] * config.VECTOR_DIM # 维度需根据实际模型调整 (如Qwen是4096，BGE是1024)
            delay = base_delay * (2 ** attempt)
            print(f"Embedding Error attempt {attempt+1}/{max_retries}: {e}, retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
    # 理论上不会执行到这里
    return [0.0] * config.VECTOR_DIM

async def get_findings_vectors(findings_list, max_concurrent=None):
    """
    计算 findings 向量列表。
    findings_list: 列表，每个元素为 {"summary": "...", "explanation": "..."} 或字符串。
    max_concurrent: 并发数，默认使用 config.MAX_CONCURRENT_NET_REQS
    返回: List[List[Float]]，与 findings_list 一一对应。
    """
    if not findings_list:
        return []
    
    if max_concurrent is None:
        max_concurrent = config.MAX_CONCURRENT_NET_REQS
    
    # 合并 summary 和 explanation 文本，转换为简体中文
    texts = []
    for f in findings_list:
        if isinstance(f, dict):
            combined = f"{f.get('summary', '')} {f.get('explanation', '')}".strip()
        elif isinstance(f, str):
            # 如果元素是字符串，直接使用该字符串作为文本
            combined = f.strip()
        else:
            # 其他类型，转换为字符串
            combined = str(f)
        simplified = zhconv.convert(combined, 'zh-cn')
        texts.append(simplified)
    
    # 并发获取向量
    sem = asyncio.Semaphore(max_concurrent)
    async def embed_one(text):
        async with sem:
            return await get_embedding(text)
    
    tasks = [embed_one(t) for t in texts]
    vectors = await asyncio.gather(*tasks)
    return vectors

def bin_packing_text_units(text_units, total_tokens):
    """装箱法分段"""
    if total_tokens <= MAP_REDUCE_THRESHOLD:
        return ["\n\n".join(text_units)]
    
    num_chunks = math.ceil(total_tokens / TARGET_CHUNK_SIZE)
    dynamic_threshold = total_tokens / num_chunks
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for text in text_units:
        text_len = count_tokens(text)
        if current_chunk and (current_len + text_len > dynamic_threshold * 1.1):
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(text)
        current_len += text_len
        
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

def clean_and_parse_json(llm_output):
    """
    使用 json_repair 进行超强鲁棒解析：
    1. 自动修复未闭合的引号/括号 (解决 Unterminated string)
    2. 自动转义内容中的双引号 (解决 Expecting delimiter)
    3. 自动剥离 Markdown
    """
    try:
        # 尝试使用 json_repair 直接解析
        # skip_json_loads=True 表示不使用标准 json.loads，完全依赖 repair 逻辑
        decoded_object = json_repair.repair_json(llm_output, return_objects=True, skip_json_loads=True)
        
        # json_repair 有时返回列表，有时返回字典，确保我们需要的是字典
        if isinstance(decoded_object, list):
            # 如果 LLM 返回了 [ {result} ]，取第一个
            if len(decoded_object) > 0:
                return decoded_object[0]
            else:
                return {}
        return decoded_object

    except Exception as e:
        # 如果连 json_repair 都修不好，那只能抛出异常触发重试了
        # 但通常这种情况极少发生
        raise ValueError(f"JSON Repair Failed: {e} | Content Head: {llm_output[:100]}...")

async def llm_call(prompt, task_type="direct", max_retries=5):
    """通用 LLM 调用，带指数退避重试"""
    import asyncio
    base_delay = 2.0 
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个严谨的古籍研究助手。请只输出 JSON。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=4096  # <---【关键修改】显式增加输出上限！
            )
            content = response.choices[0].message.content
            
            # 使用新的修复函数
            parsed_json = clean_and_parse_json(content)
            
            # 简单的完整性检查：确保 summary 和 findings 字段存在
            if not isinstance(parsed_json, dict):
                 raise ValueError("Parsed result is not a dictionary")
            
            return parsed_json
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"LLM Critical Error ({task_type}) after {max_retries} retries: {e}")
                return {"summary": "调用错误", "findings": []}
            
            delay = base_delay * (2 ** attempt)
            # 打印更清晰的错误日志
            err_msg = str(e).split('|')[0]
            print(f"⚠️ Retry {attempt+1}/{max_retries} ({task_type}): {err_msg} -> Wait {delay:.1f}s")
            await asyncio.sleep(delay)
            
    return {"summary": "调用错误", "findings": []}

async def process_community_node(
    cid, comm_map, text_map, parent_to_children, 
    scheduler, path_helper, output_queue
):
    """处理单个节点"""
    node = comm_map[cid]
    title = node['title']
    level = node['level']
    is_leaf = node['is_leaf']
    path_str = path_helper(cid)
    
    context_chunks = []
    total_estimated_tokens = 0
    
    # 1. 准备 Context
    if is_leaf:
        # 真正的叶子节点
        raw_texts = text_map.get(cid, [])
        clean_texts = [t['text'] for t in raw_texts] 
        total_estimated_tokens = sum(count_tokens(t) for t in clean_texts)
        context_chunks = bin_packing_text_units(clean_texts, total_estimated_tokens)
        
    elif level == 1 and not is_leaf:
        # 特殊 Level 1: 聚合子类原文
        child_ids = parent_to_children.get(cid, [])
        aggregated_texts = []
        for child_id in child_ids:
            child_node = comm_map[child_id]
            child_title = child_node['title']
            texts = text_map.get(child_id, [])
            for t in texts:
                aggregated_texts.append(f"【小类：{child_title}】\n{t['text']}")
        
        total_estimated_tokens = sum(count_tokens(t) for t in aggregated_texts)
        context_chunks = bin_packing_text_units(aggregated_texts, total_estimated_tokens)
    else:
        return # Level 0 不在此处处理
        
    # 2. 生成流程
    final_result = {}
    first_chunk_len = count_tokens(context_chunks[0]) if context_chunks else 0
    await scheduler.acquire(first_chunk_len)
    
    try:
        if not context_chunks or (len(context_chunks) == 1 and not context_chunks[0].strip()):
            final_result = {"summary": "无有效内容", "findings": []}
        
        elif len(context_chunks) == 1:
            # === 核心逻辑修改：根据 Token 量选择 Prompt ===
            chunk_content = context_chunks[0]
            if count_tokens(chunk_content) < 200:
                # 贫瘠模式
                prompt = PROMPT_SPARSE.format(path_str=path_str, context_text=chunk_content)
            else:
                # 标准模式
                prompt = PROMPT_STANDARD.format(path_str=path_str, context_text=chunk_content)
            
            final_result = await llm_call(prompt, "Direct")
            
        else:
            # Map-Reduce 模式
            print(f"Node {title} 触发分段: {total_estimated_tokens} tokens -> {len(context_chunks)} 段")
            partial_summaries = []
            
            for idx, chunk in enumerate(context_chunks):
                prompt = PROMPT_MAP_PARTIAL.format(
                    path_str=path_str, part_idx=idx+1, context_text=chunk
                )
                res = await llm_call(prompt, f"Map-{idx}")
                partial_summaries.append(
                    f"--- 部分 {idx+1} ---\n摘要: {res.get('partial_summary','')}\n发现: {res.get('partial_findings',[])}"
                )
            
            reduce_context = "\n\n".join(partial_summaries)
            prompt = PROMPT_REDUCE_FINAL.format(path_str=path_str, context_text=reduce_context)
            final_result = await llm_call(prompt, "Reduce")
            
        # 生成摘要向量 (Report)
        # Report 的 full_content 建议转简体后再算向量以保持一致。
        # 转换为简体 search_content 用作 embedding
        full_content_raw = f"# {title}\n\n## Summary\n{final_result.get('summary','')}\n\n## Findings\n{json.dumps(final_result.get('findings',[]), ensure_ascii=False)}"
        full_content_search = zhconv.convert(full_content_raw, 'zh-cn')
        
        embedding = await get_embedding(full_content_search)
        
        # 计算 findings 向量
        findings_list = final_result.get('findings', [])
        finding_vectors = await get_findings_vectors(findings_list)
        
        result_record = {
            "community_id": cid,
            "title": title,
            "level": level,
            "summary": final_result.get('summary', ''),
            "findings": json.dumps(findings_list, ensure_ascii=False),
            "full_content": full_content_raw,
            "embedding": embedding,
            "finding_vectors": finding_vectors
        }
        
        output_queue.put_nowait(result_record)
        
    finally:
        await scheduler.release(first_chunk_len)

# --- 主流程 ---

async def run_vectorization():
    """独立运行向量化阶段，支持 JSONL 进度续传"""
    print(">>> 开始 TextUnit 向量化 (实时 zhconv 转简体)...")
    
    # 重新读取以获取完整数据 (特别是 chunks)
    df_tu_final = pd.read_parquet(INPUT_TEXT_UNITS)
    total_rows = len(df_tu_final)
    
    # 准备容器
    updated_main_vectors = [None] * total_rows
    updated_chunk_vectors = [None] * total_rows
    
    # === 向量化续传：加载已有向量和进度文件 ===
    VECTOR_PROGRESS_JSONL = "vectorization_progress.jsonl"
    processed_indices = set()
    
    if os.path.exists(VECTOR_PROGRESS_JSONL):
        print(f">>> 检测到进度文件 {VECTOR_PROGRESS_JSONL}，加载已处理索引...")
        with open(VECTOR_PROGRESS_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    idx = record.get('index')
                    status = record.get('status')
                    if status == 'success':
                        processed_indices.add(idx)
                except Exception as e:
                    print(f"警告：无法解析进度行: {e}")
        print(f"  已加载 {len(processed_indices)} 个已处理索引")
    
    # 加载已有向量文件（如果存在）以填充已处理索引的向量
    if os.path.exists(OUTPUT_TEXT_UNITS_UPDATED):
        print(f">>> 检测到已有向量文件 {OUTPUT_TEXT_UNITS_UPDATED}，加载向量数据...")
        df_existing = pd.read_parquet(OUTPUT_TEXT_UNITS_UPDATED)
        if len(df_existing) == total_rows:
            # 先加载所有向量
            for idx in range(total_rows):
                vec = df_existing.iloc[idx]['vector']
                chunk_vecs = df_existing.iloc[idx]['chunk_vectors']
                # 检查向量是否为占位向量
                if is_placeholder_vector(vec):
                    # 如果是占位向量，即使索引在 processed_indices 中，也视为无效，需要重新计算
                    if idx in processed_indices:
                        print(f"  警告：索引 {idx} 的向量为占位向量，将从已处理集合中移除")
                        processed_indices.discard(idx)
                    # 不更新 updated_main_vectors，保持为 None
                    continue
                # 向量有效
                # 如果该索引已处理，则使用现有向量
                if idx in processed_indices:
                    updated_main_vectors[idx] = vec
                    if chunk_vecs is not None and len(chunk_vecs) > 0:
                        valid_chunks = [cv for cv in chunk_vecs if not is_placeholder_vector(cv)]
                        if valid_chunks:
                            updated_chunk_vectors[idx] = valid_chunks
                # 如果索引未处理，但向量有效，我们也可以使用（避免重复计算）
                else:
                    updated_main_vectors[idx] = vec
                    if chunk_vecs is not None and len(chunk_vecs) > 0:
                        valid_chunks = [cv for cv in chunk_vecs if not is_placeholder_vector(cv)]
                        if valid_chunks:
                            updated_chunk_vectors[idx] = valid_chunks
            print(f"  已加载 {sum(1 for v in updated_main_vectors if v is not None)} 个主向量和 {sum(1 for c in updated_chunk_vectors if c is not None)} 个 chunk 向量")
            # 额外检查：确保加载的向量中没有占位向量
            placeholder_count = 0
            for idx, vec in enumerate(updated_main_vectors):
                if vec is not None and is_placeholder_vector(vec):
                    placeholder_count += 1
                    updated_main_vectors[idx] = None
            if placeholder_count:
                print(f"  警告：加载的向量中发现 {placeholder_count} 个占位向量，已替换为 None")
        else:
            print("警告：现有文件行数不匹配，将重新计算全部向量")
    else:
        print(">>> 未检测到已有向量文件，从头开始计算向量。")
    
    # === 加载向量 JSONL 文件作为保底措施 ===
    if os.path.exists(VECTOR_OUTPUT_JSONL):
        print(f">>> 检测到向量 JSONL 文件 {VECTOR_OUTPUT_JSONL}，加载向量数据...")
        loaded_from_jsonl = 0
        with open(VECTOR_OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    idx = record.get('index')
                    if idx is None:
                        continue
                    main_vec = record.get('main_vector')
                    chunk_vecs = record.get('chunk_vectors')
                    # 检查向量是否有效且非占位
                    if main_vec is not None and not is_placeholder_vector(main_vec):
                        # 如果当前 updated_main_vectors 中为 None，则使用 JSONL 中的向量
                        if updated_main_vectors[idx] is None:
                            updated_main_vectors[idx] = main_vec
                            loaded_from_jsonl += 1
                    if chunk_vecs is not None:
                        # 过滤占位向量
                        valid_chunks = [cv for cv in chunk_vecs if not is_placeholder_vector(cv)]
                        if valid_chunks and updated_chunk_vectors[idx] is None:
                            updated_chunk_vectors[idx] = valid_chunks
                except Exception as e:
                    print(f"警告：无法解析向量 JSONL 行: {e}")
        print(f"  从 JSONL 文件加载了 {loaded_from_jsonl} 个主向量")
    
    # 确定需要计算向量的索引：向量为 None 的索引
    need_compute = [idx for idx, vec in enumerate(updated_main_vectors) if vec is None]
    print(f"需要计算向量的 TextUnit 数量: {len(need_compute)} / {total_rows}")
    
    if not need_compute:
        print(">>> 所有向量已存在，跳过向量化阶段。")
    else:
        tasks_vec = []
        
        sem_vec = asyncio.Semaphore(20) # 向量化并发
        progress_lock = asyncio.Lock()
        
        async def process_tu_vector(index, row):
            async with sem_vec:
                main_vec = None
                chunk_vecs = None
                try:
                    # 1. 主向量：转简体 -> Embedding
                    orig_text = row['text']
                    search_text = zhconv.convert(orig_text, 'zh-cn')
                    main_vec = await get_embedding(search_text)
                    
                    # 检查主向量是否为占位向量
                    if is_placeholder_vector(main_vec):
                        raise ValueError("主向量为占位向量（全零），视为失败")
                    
                    # 2. Chunk 向量：遍历 -> 转简体 -> Embedding
                    chunks = row['text_chunks']
                    chunk_vecs = []
                    if chunks is not None and len(chunks) > 0:
                        for c in chunks:
                            c_simplified = zhconv.convert(c, 'zh-cn')
                            cv = await get_embedding(c_simplified)
                            if is_placeholder_vector(cv):
                                raise ValueError(f"Chunk 向量为占位向量（全零），视为失败")
                            chunk_vecs.append(cv)
                    
                    # 写入进度（包含向量信息）和向量 JSONL 作为保底
                    async with progress_lock:
                        # 计算向量摘要
                        vec_array = np.array(main_vec)
                        norm = float(np.linalg.norm(vec_array))
                        sample = vec_array[:3].tolist()
                        is_zero = is_placeholder_vector(main_vec)
                        with open(VECTOR_PROGRESS_JSONL, 'a', encoding='utf-8') as pf:
                            pf.write(json.dumps({
                                "index": index,
                                "status": "success",
                                "timestamp": pd.Timestamp.now().isoformat(),
                                "vector_norm": norm,
                                "sample": sample,
                                "is_zero": is_zero,
                                "chunk_count": len(chunk_vecs) if chunk_vecs is not None else 0
                            }, ensure_ascii=False) + "\n")
                        # 写入向量 JSONL 文件（完整向量）
                        with open(VECTOR_OUTPUT_JSONL, 'a', encoding='utf-8') as vf:
                            vf.write(json.dumps({
                                "index": index,
                                "main_vector": main_vec,
                                "chunk_vectors": chunk_vecs if chunk_vecs is not None else [],
                                "timestamp": pd.Timestamp.now().isoformat()
                            }, ensure_ascii=False) + "\n")
                    
                    return index, main_vec, chunk_vecs
                except Exception as e:
                    print(f"向量化失败 index {index}: {e}")
                    # 写入失败进度（包含向量信息）
                    async with progress_lock:
                        # 尝试计算向量摘要（如果 main_vec 存在）
                        norm = None
                        sample = None
                        is_zero = None
                        if main_vec is not None:
                            vec_array = np.array(main_vec)
                            norm = float(np.linalg.norm(vec_array))
                            sample = vec_array[:3].tolist()
                            is_zero = is_placeholder_vector(main_vec)
                        with open(VECTOR_PROGRESS_JSONL, 'a', encoding='utf-8') as pf:
                            pf.write(json.dumps({
                                "index": index,
                                "status": "failed",
                                "error": str(e),
                                "timestamp": pd.Timestamp.now().isoformat(),
                                "vector_norm": norm,
                                "sample": sample,
                                "is_zero": is_zero,
                                "chunk_count": len(chunk_vecs) if chunk_vecs is not None else 0
                            }, ensure_ascii=False) + "\n")
                    # 返回占位向量，但标记为需要重新计算（通过返回 None）
                    return index, None, None

        # 创建任务
        print(f"正在创建 {len(need_compute)} 个向量化任务...")
        for idx in need_compute:
            row = df_tu_final.iloc[idx]
            tasks_vec.append(process_tu_vector(idx, row))
            
        # 分批执行
        batch_size = 100
        for i in range(0, len(tasks_vec), batch_size):
            batch = tasks_vec[i:i+batch_size]
            results = await asyncio.gather(*batch)
            
            for idx, m_vec, c_vecs in results:
                if m_vec is not None:
                    updated_main_vectors[idx] = m_vec
                if c_vecs is not None:
                    updated_chunk_vectors[idx] = c_vecs
                
            print(f"向量化进度: {min(i+batch_size, len(tasks_vec))}/{len(tasks_vec)}")
    
    # 调试：检查前几个向量的范数
    non_zero_count = 0
    zero_count = 0
    for idx, vec in enumerate(updated_main_vectors):
        if vec is None:
            continue
        if is_placeholder_vector(vec):
            zero_count += 1
        else:
            non_zero_count += 1
        if idx < 5:
            vec_array = np.array(vec)
            norm = np.linalg.norm(vec_array)
            print(f"调试：索引 {idx} 向量范数 = {norm:.6f}")
    print(f"调试：非零向量 {non_zero_count}，零向量 {zero_count}，总计 {len(updated_main_vectors)}")

    # 安全措施：将占位向量替换为 None，避免保存零向量
    placeholder_indices = []
    for idx, vec in enumerate(updated_main_vectors):
        if vec is not None and is_placeholder_vector(vec):
            placeholder_indices.append(idx)
            updated_main_vectors[idx] = None
    if placeholder_indices:
        print(f"警告：发现 {len(placeholder_indices)} 个占位向量，已替换为 None（索引示例：{placeholder_indices[:10]}）")

    # 回填
    df_tu_final['vector'] = updated_main_vectors
    df_tu_final['chunk_vectors'] = updated_chunk_vectors

    df_tu_final.to_parquet(OUTPUT_TEXT_UNITS_UPDATED, index=False)
    print(">>> 向量化完成！文件已更新。")

async def main():
    print(">>> 正在加载数据...")
    df_comm = pd.read_parquet(INPUT_COMMUNITIES)
    df_text = pd.read_parquet(INPUT_TEXT_UNITS)
    
    # === 检查是否已存在完整的社区摘要 Parquet 文件 ===
    if os.path.exists(OUTPUT_REPORTS_PARQUET):
        print(f">>> 检测到已有社区摘要 Parquet 文件 {OUTPUT_REPORTS_PARQUET}，检查完整性...")
        df_reports = pd.read_parquet(OUTPUT_REPORTS_PARQUET)
        # 检查行数是否匹配社区总数
        if len(df_reports) == len(df_comm):
            # 检查是否有错误摘要
            error_summaries = df_reports['summary'].isin(["调用错误", "处理错误"])
            if not error_summaries.any():
                print(">>> 社区摘要已完整且无错误，跳过摘要生成阶段，直接进入向量化。")
                await run_vectorization()
                return
            else:
                print(f">>> 发现 {error_summaries.sum()} 个错误摘要，需要重新处理。")
        else:
            print(f">>> 社区摘要行数不匹配（现有 {len(df_reports)}，期望 {len(df_comm)}），需要重新处理。")
    else:
        print(">>> 未检测到社区摘要 Parquet 文件，将进行摘要生成。")
    
    comm_map = df_comm.set_index("id").to_dict(orient="index")
    parent_to_children = defaultdict(list)
    child_to_parent = {}
    dependency_count = defaultdict(int) 
    
    for _, row in df_comm.iterrows():
        cid = row['id']
        pid = row['parent_id']
        if pid:
            parent_to_children[pid].append(cid)
            child_to_parent[cid] = pid
            if comm_map[pid]['level'] == 0:
                dependency_count[pid] += 1

    text_map = defaultdict(list)
    for _, row in df_text.iterrows():
        text_map[row['community_id']].append({
            "text": row['text'],
            "chunk_vectors": row.get("chunk_vectors", [])
        })

    def get_path_chain(cid):
        path = []
        curr = cid
        while curr:
            path.append(comm_map[curr]['title'])
            curr = child_to_parent.get(curr)
        return " > ".join(reversed(path))

    # === 断点续传：读取已有 JSONL ===
    processed_set = set()
    finished_reports_cache = {}
    if os.path.exists(OUTPUT_REPORTS_JSONL):
        print(f">>> 检测到已有输出文件 {OUTPUT_REPORTS_JSONL}，加载已处理节点...")
        success_count = 0
        error_count = 0
        with open(OUTPUT_REPORTS_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    cid = record['community_id']
                    summary = record.get('summary', '')
                    # 检查是否为错误摘要
                    if summary in ["调用错误", "处理错误"]:
                        error_count += 1
                        print(f"  跳过错误节点 {cid}: {summary}")
                        continue  # 不加入 processed_set，后续重新处理
                    # 成功处理
                    processed_set.add(cid)
                    success_count += 1
                    if record['level'] == 1:
                        finished_reports_cache[cid] = record
                except Exception as e:
                    print(f"警告：无法解析行: {e}")
        print(f"  已加载 {success_count} 个成功节点，跳过 {error_count} 个错误节点")
        
        # 根据成功处理节点更新 dependency_count
        for cid in processed_set:
            pid = child_to_parent.get(cid)
            if pid and comm_map[pid]['level'] == 0:
                dependency_count[pid] -= 1
                if dependency_count[pid] < 0:
                    dependency_count[pid] = 0  # 防止负数
    else:
        print(">>> 未检测到已有输出文件，从头开始处理。")

    # 初始化
    scheduler = TokenScheduler(MAX_TOKENS_IN_FLIGHT, MAX_CONCURRENT_NET_REQS)
    task_queue = asyncio.Queue()
    result_queue = asyncio.Queue()
    
    # 初始任务：未处理且依赖计数为0的节点
    initial_tasks = []
    for cid in df_comm['id']:
        if cid in processed_set:
            continue  # 已处理，跳过
        if dependency_count[cid] == 0:
            initial_tasks.append(cid)
    
    for cid in initial_tasks:
        task_queue.put_nowait(cid)
    
    # === 核心修改：退出机制 ===
    total_nodes_count = len(df_comm)
    processed_count = len(processed_set)  # 已处理节点数
    all_done_event = asyncio.Event() # 用于通知主进程所有任务完成
    
    print(f"初始化任务数: {len(initial_tasks)} / 总数: {total_nodes_count} (已处理: {processed_count})")

    # 文件写入器
    async def file_writer():
        nonlocal processed_count
        # 追加模式，保留已有数据
        with open(OUTPUT_REPORTS_JSONL, "a", encoding="utf-8") as f:
            while True:
                record = await result_queue.get()
                if record is None: break
                
                # 安全检查：如果该节点已处理（极罕见情况），跳过写入
                if record['community_id'] in processed_set:
                    print(f"警告：节点 {record['community_id']} 已处理，跳过写入")
                else:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                
                if record['level'] == 1:
                    finished_reports_cache[record['community_id']] = record
                
                # DAG 解锁逻辑：仅当子节点成功时才减少父节点依赖计数
                pid = child_to_parent.get(record['community_id'])
                if pid and comm_map[pid]['level'] == 0:
                    summary = record.get('summary', '')
                    if summary not in ["调用错误", "处理错误"]:
                        dependency_count[pid] -= 1
                        if dependency_count[pid] == 0:
                            task_queue.put_nowait(pid)
                    else:
                        print(f"  子节点 {record['community_id']} 摘要为错误，不减少父节点 {pid} 的依赖计数")
                
                # 更新全局计数器
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"进度: {processed_count}/{total_nodes_count}")
                
                # 检查是否全部完成
                if processed_count == total_nodes_count:
                    print(">>> 所有节点处理完毕！设置结束信号。")
                    all_done_event.set()

                result_queue.task_done()
    
    writer_task = asyncio.create_task(file_writer())

    # Worker
    async def worker():
        while not all_done_event.is_set():
            try:
                # 阻塞获取，但不需要 timeout
                cid = await task_queue.get()
            except asyncio.CancelledError:
                return

            try:
                if comm_map[cid]['level'] == 0:
                    # ===========================
                    # Level 0 处理 (Level 0 Logic) - 支持 Map-Reduce 分段
                    # ===========================
                    path_str = get_path_chain(cid)
                    child_ids = parent_to_children.get(cid, [])
                    child_summaries = []
                    
                    # 从缓存读取子节点摘要
                    for child_id in child_ids:
                        rep = finished_reports_cache.get(child_id)
                        if rep:
                            child_summaries.append(f"【{rep['title']}】: {rep['summary']}")
                    
                    # 过滤空摘要
                    child_summaries = [cs for cs in child_summaries if cs.strip()]
                    if not child_summaries:
                        context_text = "无内容"
                        total_tokens = 0
                    else:
                        # 计算总 Token 数
                        total_tokens = sum(count_tokens(cs) for cs in child_summaries)
                    
                    # 申请资源（使用总 Token 数作为预估）
                    await scheduler.acquire(total_tokens)
                    try:
                        # 判断是否需要 Map-Reduce 分段
                        if total_tokens <= MAP_REDUCE_THRESHOLD:
                            # 直接使用 Level 0 标准提示词
                            context_text = "\n\n".join(child_summaries) if child_summaries else "无内容"
                            prompt = PROMPT_LEVEL0.format(path_str=path_str, context_text=context_text)
                            final_result = await llm_call(prompt, "Level-0")
                        else:
                            # Map-Reduce 分段处理
                            print(f"部 {comm_map[cid]['title']} 触发分段: {total_tokens} tokens")
                            # 使用装箱法分段
                            chunks = bin_packing_text_units(child_summaries, total_tokens)
                            print(f"  切分为 {len(chunks)} 段")
                            
                            partial_summaries = []
                            for idx, chunk in enumerate(chunks):
                                prompt = PROMPT_LEVEL0_MAP_PARTIAL.format(
                                    path_str=path_str, part_idx=idx+1, context_text=chunk
                                )
                                res = await llm_call(prompt, f"Level0-Map-{idx}")
                                partial_summaries.append(
                                    f"--- 部分 {idx+1} ---\n摘要: {res.get('partial_summary','')}\n发现: {res.get('partial_findings',[])}"
                                )
                            
                            reduce_context = "\n\n".join(partial_summaries)
                            prompt = PROMPT_LEVEL0_REDUCE_FINAL.format(path_str=path_str, context_text=reduce_context)
                            final_result = await llm_call(prompt, "Level0-Reduce")
                        
                        # 向量化 (转简体)
                        full_content_raw = f"# {comm_map[cid]['title']}\n\n## Summary\n{final_result.get('summary','')}\n\n## Findings\n{json.dumps(final_result.get('findings',[]), ensure_ascii=False)}"
                        full_content_search = zhconv.convert(full_content_raw, 'zh-cn')
                        embedding = await get_embedding(full_content_search)
                        
                        # 计算 findings 向量
                        findings_list = final_result.get('findings', [])
                        finding_vectors = await get_findings_vectors(findings_list)
                        
                        res_record = {
                            "community_id": cid,
                            "title": comm_map[cid]['title'],
                            "level": 0,
                            "summary": final_result.get('summary', ''),
                            "findings": json.dumps(findings_list, ensure_ascii=False),
                            "full_content": full_content_raw,
                            "embedding": embedding,
                            "finding_vectors": finding_vectors
                        }
                        result_queue.put_nowait(res_record)
                    finally:
                        # 释放资源
                        await scheduler.release(total_tokens)
                else:
                    # ===========================
                    # Level 1/2 处理 (Level 1/2 Logic)
                    # ===========================
                    await process_community_node(
                        cid, comm_map, text_map, parent_to_children,
                        scheduler, get_path_chain, result_queue
                    )
            except Exception as e:
                print(f"Worker Error ID {cid}: {e}")
                # 出错也要增加计数，放入 Dummy 数据防止死锁
                level = comm_map[cid]['level']
                dummy = {
                    "community_id": cid,
                    "title": comm_map[cid]['title'],
                    "level": level,
                    "summary": "处理错误",
                    "findings": "[]",
                    "full_content": "",
                    "embedding": [],
                    "finding_vectors": []
                }
                result_queue.put_nowait(dummy)
            finally:
                task_queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(20)]
    
    # === 等待结束 ===
    print("正在等待所有任务完成...")
    await all_done_event.wait()
    
    print("任务完成信号已触发，正在停止 Workers...")
    for w in workers: w.cancel()
    result_queue.put_nowait(None) # 停止 writer
    await writer_task

    # 转存 Parquet
    print(">>> 正在转存 Reports Parquet...")
    data = []
    with open(OUTPUT_REPORTS_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    pd.DataFrame(data).to_parquet(OUTPUT_REPORTS_PARQUET, index=False)
    
    # === 向量化阶段 (Step 2 Part B) ===
    await run_vectorization()

if __name__ == "__main__":
    asyncio.run(main())