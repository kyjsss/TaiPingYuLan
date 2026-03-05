import pandas as pd
import numpy as np
import json
import os
import asyncio
import json_repair
from openai import AsyncOpenAI
from collections import defaultdict
import config

# --- 配置区 ---
# 从配置文件读取
API_KEY = config.SILICONFLOW_API_KEY
BASE_URL = config.SILICONFLOW_BASE_URL

# 全局速率限制协调
rate_limit_condition = asyncio.Condition()
rate_limit_triggered = False

# 模型配置
LLM_MODEL = config.LLM_MODEL

# 文件路径
INPUT_COMMUNITIES = config.INPUT_COMMUNITIES
INPUT_REPORTS = config.INPUT_REPORTS
OUTPUT_RELATIONSHIPS_JSONL = config.OUTPUT_RELATIONSHIPS_JSONL
OUTPUT_RELATIONSHIPS_PARQUET = config.OUTPUT_RELATIONSHIPS_PARQUET

# 漏斗 1: 数学过滤参数
THRESHOLD_MIN = config.THRESHOLD_MIN
THRESHOLD_MAX = config.THRESHOLD_MAX
SHADOW_NODE_SIM = config.SHADOW_NODE_SIM
PARENT_CHILD_DIFF = config.PARENT_CHILD_DIFF
TOP_K_CANDIDATES = config.TOP_K_CANDIDATES

# 漏斗 2&3: LLM 审计参数
MAX_TOKENS_IN_FLIGHT = config.MAX_TOKENS_IN_FLIGHT
MAX_CONCURRENT_REQS = config.MAX_CONCURRENT_REQS

# 断点续传检查点文件
CHECKPOINT_FILE = config.CHECKPOINT_FILE

# --- 初始化 ---
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Prompt (核心) ---
PROMPT_RELATION_MINER = """
你是一位精通中国古代文献、知识分类学及博物学的专家。
你的任务是审核《太平御览》中两个条目（社区）之间是否存在**深层逻辑联系**。

--- 条目 A ---
标题：{title_a}
内容摘要：{content_a}

--- 条目 B ---
标题：{title_b}
内容摘要：{content_b}

### 核心任务
请判断 A 和 B 是否符合以下**特定关系定义**。如果符合，请打高分；如果不符合或仅为字面重复，请打低分。

### 关系定义表 (Schema)
**1. 有向关系 (必须区分 Source/Target，方向性明确)**
- **预兆 (Portends)**: 自然异象(A) 预示 人事/政治结果(B)。(例: 彗星 -> 兵灾)
- **应感 (RespondsTo)**: 人事行为(A) 引发 自然反馈(B)。(例: 冤狱 -> 大旱)
- **主治 (Treats)**: 药物/物质(A) 治疗 疾病(B)。(例: 麻黄 -> 伤寒)
- **材质 (MaterialFor)**: 原材料(A) 制成 器物(B)。(例: 桐木 -> 琴)
- **出产 (Produces)**: 地点(A) 出产 特产(B)。(例: 蓝田 -> 玉)
- **进贡 (Tributes)**: 政治实体(A) 进献 物品(B)。(例: 波斯 -> 珊瑚)
- **用于礼 (UsedInRitual)**: 物品(A) 专用于 仪式(B)。(例: 鼎 -> 祭祀)
- **象征 (Symbolizes)**: 物(A) 隐喻 抽象品德/身份(B)。(例: 松柏 -> 贞节)

**2. 双向关系 (A与B平等或互通，输出 direction="bidirectional")**
- **对应 (CorrespondsTo)**: 五行/天人感应上的同构。(例: 春 <-> 木 <-> 肝 <-> 仁)
- **相克 (IncompatibleWith)**: 五行相克或药物/食物禁忌。(例: 水 <-> 火, 甘草 <-> 大戟)
- **并称 (PairedWith)**: 典故或习惯中常成对出现。(例: 龙 <-> 凤, 钟 <-> 鼓)
- **齐名 (PeerOf)**: 属于同组著名集合。(例: 泰山 <-> 华山 [五岳], 金 <-> 银)
- **别名 (AliasOf)**: 名异实同。(例: 伏羲 <-> 太昊, 杜康 <-> 酒)
- **形似 (Resembles)**: 物理形态上的类比。(例: 鱟 <-> 扇)

### 评分标准 (Score 0-10)
- **0-6分 (驳回)**: 
  - 仅有字面重叠（如都出现了“水”字但无逻辑关联）。
  - 简单的层级/包含关系（如“虎”属于“兽部”，这已在目录树中，无需建立边）。
  - 关系模糊，不符合上述任何定义。
- **7-10分 (保留)**: 
  - 关系清晰，精准符合上述 Schema 中的某一项。
  - 具有学术价值或历史逻辑深度。

### 输出格式 (JSON)
{{
    "score": <0-10>,
    "relation_type": "<必须严格从上述英文代码中选一个，如 Portends 或 CorrespondsTo>",
    "direction": "<forward (A->B) / backward (B->A) / bidirectional>",
    "thought_process": "<简短分析理由>",
    "description": "<用一句话描述该关系，如：A 是 B 的五行对应色（但不要说AB,说标题即可）>"
}}
"""

# --- 辅助类：调度器 ---
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

# --- 核心类：候选生成器 (Matrix Math + Keyword Rescue) ---
class CandidateGenerator:
    def __init__(self, df_comm, df_reports):
        print(">>> 正在初始化矩阵计算引擎...")
        # 1. 合并数据：我们需要 parent_id (从 comm) 和 embedding (从 reports)
        self.df = pd.merge(
            df_reports[['community_id', 'title', 'embedding', 'full_content']],
            df_comm[['id', 'parent_id', 'level']],
            left_on='community_id', right_on='id', how='inner'
        )

        initial_len = len(self.df)
        self.df = self.df[self.df['level'] > 0].reset_index(drop=True)
        print(f"已剔除 Level 0 节点。参与计算的节点数: {initial_len} -> {len(self.df)}")
        
        # 2. 建立索引映射
        self.ids = self.df['community_id'].values
        self.id_to_idx = {id_: i for i, id_ in enumerate(self.ids)}
        self.parent_ids = self.df['parent_id'].values
        self.titles = self.df['title'].values
        
        # 预处理：将 full_content 转为用于搜索的字符串（建议转简体以提高匹配率）
        # 这里假设 full_content 已经是清洗过的
        self.contents = self.df['full_content'].values
        
        # 3. 准备 Embedding 矩阵
        print("正在堆叠向量矩阵...")
        self.matrix = np.stack(self.df['embedding'].values)
        # 归一化以便计算余弦相似度
        norm = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / (norm + 1e-9)
        
        self.N = len(self.df)
        print(f"矩阵就绪: {self.N} x {self.matrix.shape[1]}")

    def pre_calculate_shadow_nodes(self):
        """
        预计算“影子节点”：即子类与父类相似度极高 (> SHADOW_NODE_SIM)。
        这些子类将被标记，不主动作为 Source 发起连接。
        """
        print("正在识别影子节点...")
        shadow_indices = set()
        
        for i in range(self.N):
            pid = self.parent_ids[i]
            if pid and pid in self.id_to_idx:
                p_idx = self.id_to_idx[pid]
                # 计算 Child - Parent 相似度
                sim = np.dot(self.matrix[i], self.matrix[p_idx])
                if sim > SHADOW_NODE_SIM:
                    shadow_indices.add(i)
                    
        print(f"发现 {len(shadow_indices)} 个影子节点 (将跳过主动连接)。")
        return shadow_indices

    def generate_candidates(self):
        shadow_indices = self.pre_calculate_shadow_nodes()
        candidates = []
        BATCH_SIZE = 500
        rescued_count = 0
        
        print("开始全量矩阵扫描 (含标题互现捞人策略)...")
        for start_row in range(0, self.N, BATCH_SIZE):
            end_row = min(start_row + BATCH_SIZE, self.N)
            sim_batch = np.dot(self.matrix[start_row:end_row], self.matrix.T)
            
            for i_local in range(end_row - start_row):
                i_global = start_row + i_local
                
                # 影子节点不作为 Source
                if i_global in shadow_indices:
                    continue
                
                scores = sim_batch[i_local]
                # 诊断：检查索引范围
                if start_row == 0 and i_local == 0:
                    print(f"[DEBUG] 第一个节点: i_global={i_global}, scores.shape={scores.shape}, N={self.N}")
                    print(f"[DEBUG] scores[:i_global+1] 将影响前 {i_global+1} 个元素")
                    # 打印前几个相似度值
                    print(f"[DEBUG] scores[0:5] = {scores[0:5]}")
                
                scores[:i_global + 1] = -1.0 # 排除自身和下三角
                
                # === 策略 A: 正常的向量过滤 ===
                # 满足 0.78 < score < 0.94 的
                vector_indices = np.where((scores > THRESHOLD_MIN) & (scores < THRESHOLD_MAX))[0]
                
                # === 策略 B: 标题互现 (Keyword Rescue) ===
                # 针对 [0.35, 0.70] 之间的“潜在遗珠”做关键词检查
                low_score_indices = np.where((scores > 0.35) & (scores <= THRESHOLD_MIN))[0]
                
                rescued_indices = []
                my_title = self.titles[i_global]
                my_content = self.contents[i_global]
                
                # 只有标题长度 > 1 才启用互现检查，防止单字误判
                if len(my_title) > 1 or any(len(self.titles[j]) > 1 for j in low_score_indices):
                    for j in low_score_indices:
                        other_title = self.titles[j]
                        other_content = self.contents[j]
                        
                        hit = False
                        if len(my_title) > 1 and my_title in other_content:
                            hit = True
                        elif len(other_title) > 1 and other_title in my_content:
                            hit = True
                        
                        if hit:
                            rescued_indices.append(j)
                            # 给一个“虚高”的分数，保证它能排进 Top K
                            # 0.99 几乎可以打败所有父类竞争
                            scores[j] = 0.99 
                
                if len(rescued_indices) > 0:
                    rescued_count += len(rescued_indices)

                # 合并索引
                combined_indices = np.union1d(vector_indices, rescued_indices)
                
                if len(combined_indices) == 0:
                    continue
                
                # 3. 同父过滤 & 父子竞争
                final_targets = []
                my_parent = self.parent_ids[i_global]
                
                for j in combined_indices:
                    j = int(j) 
                    
                    # 规则: 排除同父
                    if my_parent and my_parent == self.parent_ids[j]:
                        continue
                    
                    # 规则: 父子竞争
                    should_keep = True
                    # 只有当分数不是“保送分(0.99)”时，才进行父子PK
                    if scores[j] < 0.98:
                        if my_parent and my_parent in self.id_to_idx:
                            p_idx = self.id_to_idx[my_parent]
                            sim_p_t = np.dot(self.matrix[p_idx], self.matrix[j])
                            sim_c_t = scores[j]
                            
                            # 诊断日志
                            if start_row == 0 and i_local == 0 and j < 5:
                                print(f"[DEBUG] 父子竞争: src={i_global}, parent={my_parent}, target={j}")
                                print(f"[DEBUG]   sim_c_t (score) = {sim_c_t}, sim_p_t = {sim_p_t}, diff = {sim_c_t - sim_p_t}")
                            
                            if sim_c_t - sim_p_t < PARENT_CHILD_DIFF:
                                should_keep = False
                    
                    if should_keep:
                        final_targets.append((j, scores[j]))
                
                # 4. Top-K 截断
                final_targets.sort(key=lambda x: x[1], reverse=True)
                final_targets = final_targets[:TOP_K_CANDIDATES]
                
                for tgt_idx, score in final_targets:
                    candidates.append({
                        "src_idx": i_global,
                        "tgt_idx": tgt_idx,
                        "score": float(score)
                    })
        
        print(f"候选生成完成。通过关键词互现策略，额外挽救了 {rescued_count} 个低分关联。")
        print(f"最终送审候选对数: {len(candidates)}")
        return candidates

# --- 核心逻辑：LLM 挖掘 ---
async def call_llm_with_retry(prompt, max_retries=5, initial_delay=1.0):
    """
    带指数退避重试的 LLM 调用函数。
    返回解析后的 JSON 结果，如果所有重试都失败则返回 None。
    """
    global rate_limit_triggered
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            # 在每次尝试前检查全局速率限制标志
            async with rate_limit_condition:
                while rate_limit_triggered:
                    print(f"[Worker] 检测到全局速率限制已触发，等待恢复...")
                    await rate_limit_condition.wait()
            
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个严谨的古籍研究助手。请只输出 JSON。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1024
            )
            # 使用 json_repair 解析
            result = json_repair.repair_json(response.choices[0].message.content, return_objects=True)
            if isinstance(result, list):
                result = result[0] if result else {}
            return result
        except Exception as e:
            # 检查是否为速率限制错误（429）或服务器错误（5xx）或网络错误
            error_str = str(e)
            if attempt == max_retries - 1:
                print(f"LLM 调用失败，已达最大重试次数 {max_retries}: {error_str}")
                return None
            # 如果是认证错误、内容过长等不可恢复错误，直接退出重试
            if "authentication" in error_str.lower() or "too long" in error_str.lower():
                print(f"不可恢复错误，停止重试: {error_str}")
                return None
            
            # 特殊处理 429 错误：触发全局等待
            if "429" in error_str or "rate limiting" in error_str.lower():
                async with rate_limit_condition:
                    if not rate_limit_triggered:
                        rate_limit_triggered = True
                        print(f"检测到速率限制 (429)，触发全局暂停 60 秒 (尝试 {attempt+1}/{max_retries})")
                        # 通知所有等待的 worker
                        rate_limit_condition.notify_all()
                    else:
                        print(f"速率限制已由其他 worker 触发，本 worker 等待恢复")
                
                # 等待 60 秒
                await asyncio.sleep(60.0)
                
                # 恢复标志并通知
                async with rate_limit_condition:
                    rate_limit_triggered = False
                    rate_limit_condition.notify_all()
                    print("全局速率限制已解除，恢复处理")
                
                # 重置延迟，避免指数退避导致过长等待
                delay = initial_delay
                continue
            
            print(f"LLM 调用失败 (尝试 {attempt+1}/{max_retries})，等待 {delay:.1f} 秒后重试: {error_str}")
            await asyncio.sleep(delay)
            delay *= 2  # 指数退避

async def process_candidates(generator, candidates):
    scheduler = TokenScheduler(MAX_TOKENS_IN_FLIGHT, MAX_CONCURRENT_REQS)
    
    # --- 断点续传：加载已处理的候选对 ---
    processed_pairs = set()
    if os.path.exists(CHECKPOINT_FILE):
        print(f"发现检查点文件 {CHECKPOINT_FILE}，正在加载...")
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    src = record.get('src_idx')
                    tgt = record.get('tgt_idx')
                    if src is not None and tgt is not None:
                        processed_pairs.add((src, tgt))
                except:
                    pass
        print(f"已加载 {len(processed_pairs)} 个已处理候选对。")
    
    # 过滤掉已处理的候选对
    filtered_candidates = []
    for c in candidates:
        if (c['src_idx'], c['tgt_idx']) not in processed_pairs:
            filtered_candidates.append(c)
    
    print(f"候选对总数: {len(candidates)}，已处理: {len(processed_pairs)}，待处理: {len(filtered_candidates)}")
    
    if not filtered_candidates:
        print("所有候选对均已处理，无需继续。")
        return
    
    # 结果写入器（追加模式，避免覆盖已有结果）
    output_file = open(OUTPUT_RELATIONSHIPS_JSONL, "a", encoding="utf-8")
    # 检查点写入器（追加模式）
    checkpoint_file = open(CHECKPOINT_FILE, "a", encoding="utf-8")
    
    async def worker(candidate):
        src_idx = candidate['src_idx']
        tgt_idx = candidate['tgt_idx']
        
        # 获取内容
        row_a = generator.df.iloc[src_idx]
        row_b = generator.df.iloc[tgt_idx]
        
        # 估算 Token (Input)
        text_a = row_a['full_content'][:3000] # 截断保护
        text_b = row_b['full_content'][:3000]
        cost = len(text_a) + len(text_b) + 500
        
        await scheduler.acquire(cost)
        try:
            prompt = PROMPT_RELATION_MINER.format(
                title_a=row_a['title'], content_a=text_a,
                title_b=row_b['title'], content_b=text_b
            )
            
            result = await call_llm_with_retry(prompt)
            if result is None:
                print(f"警告：候选对 {src_idx}-{tgt_idx} 处理失败，已跳过")
                return
            
            # --- 漏斗 3: 审计过滤 ---
            score = result.get("score", 0)
            if score >= 7:
                direction = result.get("direction", "bidirectional").lower()
                
                base_record = {
                    "weight": score / 10.0,
                    "relation_type": result.get("relation_type", "related"),
                    "description": result.get("description", ""),
                    "vector_score": candidate['score']
                }
                
                records_to_save = []
                
                # --- 方向处理逻辑 ---
                if direction == "forward":
                    rec = base_record.copy()
                    rec["source_id"] = row_a['community_id']
                    rec["target_id"] = row_b['community_id']
                    records_to_save.append(rec)
                    
                elif direction == "backward":
                    rec = base_record.copy()
                    rec["source_id"] = row_b['community_id']
                    rec["target_id"] = row_a['community_id']
                    records_to_save.append(rec)
                    
                else: # bidirectional or unknown
                    rec1 = base_record.copy()
                    rec1["source_id"] = row_a['community_id']
                    rec1["target_id"] = row_b['community_id']
                    
                    rec2 = base_record.copy()
                    rec2["source_id"] = row_b['community_id']
                    rec2["target_id"] = row_a['community_id']
                    
                    records_to_save.append(rec1)
                    records_to_save.append(rec2)
                
                # 写入输出文件
                for r in records_to_save:
                    output_file.write(json.dumps(r, ensure_ascii=False) + "\n")
                    output_file.flush()
            
            # 无论是否成功保留关系，都标记为已处理（避免重复尝试）
            checkpoint_record = {"src_idx": src_idx, "tgt_idx": tgt_idx}
            checkpoint_file.write(json.dumps(checkpoint_record, ensure_ascii=False) + "\n")
            checkpoint_file.flush()
                    
        except Exception as e:
            print(f"Error processing pair {src_idx}-{tgt_idx}: {e}")
        finally:
            await scheduler.release(cost)

    # 批量执行
    tasks = [worker(c) for c in filtered_candidates]
    
    # 使用 tqdm 显示进度
    try:
        from tqdm.asyncio import tqdm
        for f in tqdm.as_completed(tasks, desc="LLM Auditing"):
            await f
    except ImportError:
        await asyncio.gather(*tasks)
    
    output_file.close()
    checkpoint_file.close()

# --- 主流程 ---
def main():
    if not os.path.exists(INPUT_REPORTS) or not os.path.exists(INPUT_COMMUNITIES):
        print("错误：找不到输入文件，请先运行 Step 2。")
        return

    # 1. 加载数据
    df_comm = pd.read_parquet(INPUT_COMMUNITIES)
    df_reports = pd.read_parquet(INPUT_REPORTS)
    
    # 2. 初始化生成器 & 数学过滤
    generator = CandidateGenerator(df_comm, df_reports)
    candidates = generator.generate_candidates()
    
    if not candidates:
        print("未发现符合数学过滤条件的候选对。")
        return

    # 3. LLM 审计与生成
    print(f"\n>>> 开始 LLM 审计流程 (共 {len(candidates)} 个提案)...")
    asyncio.run(process_candidates(generator, candidates))
    
    # 4. 转存 Parquet
    print("\n>>> 正在转存最终关系表...")
    if os.path.exists(OUTPUT_RELATIONSHIPS_JSONL):
        data = []
        with open(OUTPUT_RELATIONSHIPS_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except: pass
        
        if data:
            df_rel = pd.DataFrame(data)
            # 确保列顺序
            cols = ["source_id", "target_id", "weight", "relation_type", "description", "vector_score"]
            # 如果有额外列也保留
            df_rel = df_rel[cols]
            df_rel.to_parquet(OUTPUT_RELATIONSHIPS_PARQUET, index=False)
            print(f"成功保存: {OUTPUT_RELATIONSHIPS_PARQUET} (共 {len(df_rel)} 条边)")
        else:
            print("警告：没有生成任何有效关系 (JSONL 为空)。")
    else:
        print("未找到 JSONL 输出文件。")

if __name__ == "__main__":
    main()