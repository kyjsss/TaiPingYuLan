import pandas as pd
import hashlib
import json
import numpy as np
import os
import config

# --- 配置区域 ---
INPUT_FILE = "taipingyulan.csv"
OUTPUT_COMMUNITIES = "create_final_communities.parquet"
OUTPUT_TEXT_UNITS = "create_final_text_units.parquet"
VECTOR_DIM = config.VECTOR_DIM      # Qwen Embedding 维度

# --- 切片策略配置 ---
LONG_TEXT_THRESHOLD = 500  # 文本长度阈值，超过此长度进行切片
CHUNK_SIZE = 300           # 切片长度
CHUNK_OVERLAP = 100        # 重叠长度
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP  # 步长

# --- 辅助函数 ---

def generate_id(content):
    """生成 MD5 ID"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def count_tokens(text):
    """计算 Token 数量"""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        return len(text)

def get_placeholder_vector(dim=config.VECTOR_DIM):
    """生成主向量占位符"""
    return [0.0] * dim

def get_chunk_vectors_placeholder(num_chunks, dim=config.VECTOR_DIM):
    """生成切片向量占位符"""
    return [[0.0] * dim for _ in range(num_chunks)]

def split_text_chunks_smart(text):
    """
    文本切片逻辑:
    1. 从第0个字符开始切分。
    2. 提取 Head (前200字) 和 Tail (后100字) 作为元数据。
    3. 末尾回退: 若最后一段不足 300 字，则向前回退，取最后 300 字。
    """
    # 1. 提取 Head/Tail
    head = text[:200]
    tail = text[-100:]
    
    # 2. 判断是否需要切分
    if len(text) <= LONG_TEXT_THRESHOLD:
        return [], head, tail
        
    chunks = []
    text_len = len(text)
    curr = 0
    
    while curr < text_len:
        # 检查是否到达末尾部分
        if curr + CHUNK_SIZE >= text_len:
            # 末尾回退策略
            final_chunk = text[-CHUNK_SIZE:]
            chunks.append(final_chunk)
            break
        else:
            # 正常切分
            chunk = text[curr : curr + CHUNK_SIZE]
            chunks.append(chunk)
            # 滑动
            curr += CHUNK_STEP
            
    return chunks, head, tail

def normalize_source_name(name):
    """
    处理文献来源名称:
    若包含点号 '·'，只取点号前的部分。
    """
    name = str(name).strip()
    if '·' in name:
        return name.split('·')[0].strip()
    return name

# --- 主逻辑 ---

def main():
    print(f"读取文件: {INPUT_FILE} ...")
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 文件 {INPUT_FILE} 不存在")
        return

    df = pd.read_csv(INPUT_FILE)
    df.fillna("", inplace=True)
    
    print("处理文献来源并统计频次...")
    df['文献来源'] = df['文献来源'].apply(normalize_source_name)
    source_counts = df['文献来源'].value_counts().to_dict()
    
    communities = {}
    text_units = []
    
    print("构建层级结构与处理正文...")
    
    for idx, row in df.iterrows():
        # --- A. 构建层级 (Communities) ---
        l0_name = row['部'].strip()
        if not l0_name: continue
        
        l0_path = l0_name
        l0_id = generate_id(f"L0_{l0_path}")
        
        # Level 0
        if l0_id not in communities:
            communities[l0_id] = {
                "id": l0_id, "title": l0_name, "level": 0,
                "parent_id": None, "child_ids": set()
            }
        current_community_id = l0_id
        current_path = l0_path
        
        # Level 1
        l1_name = row['子目'].strip()
        if l1_name:
            l1_path = f"{l0_path}/{l1_name}"
            l1_id = generate_id(f"L1_{l1_path}")
            communities[l0_id]["child_ids"].add(l1_id)
            
            if l1_id not in communities:
                communities[l1_id] = {
                    "id": l1_id, "title": l1_name, "level": 1,
                    "parent_id": l0_id, "child_ids": set()
                }
            current_community_id = l1_id
            current_path = l1_path
            
            # Level 2
            l2_name = str(row['小类']).strip()
            if l2_name:
                l2_path = f"{l1_path}/{l2_name}"
                l2_id = generate_id(f"L2_{l2_path}")
                communities[l1_id]["child_ids"].add(l2_id)
                
                if l2_id not in communities:
                    communities[l2_id] = {
                        "id": l2_id, "title": l2_name, "level": 2,
                        "parent_id": l1_id, "child_ids": set()
                    }
                current_community_id = l2_id
                current_path = l2_path

        # --- B. 处理正文 (TextUnit) ---
        # 保持原文，不进行简繁转换
        text_content = row['正文'].strip()
        if not text_content: continue
        
        source_name = row['文献来源']
        should_prepend = True if source_counts.get(source_name, 0) > 1 else False
        
        tu_id = generate_id(f"{current_path}::{text_content}")
        
        # 调用切片逻辑
        chunks, head, tail = split_text_chunks_smart(text_content)
        
        metadata = {"source": source_name, "note": row['来源备注']}
        
        text_units.append({
            "id": tu_id,
            "text": text_content,
            "n_tokens": count_tokens(text_content),
            "vector": get_placeholder_vector(VECTOR_DIM), 
            "community_id": current_community_id,
            
            # Chunk 信息
            "text_chunks": chunks,
            "chunk_vectors": get_chunk_vectors_placeholder(len(chunks), VECTOR_DIM),
            
            "head": head,
            "tail": tail,
            "source_metadata": json.dumps(metadata, ensure_ascii=False),
            "hierarchy_path": current_path,
            "prepend_source": should_prepend
        })

    # --- 后处理与导出 ---
    print("保存 Parquet 文件...")
    
    # 1. Communities
    community_list = []
    for cid, data in communities.items():
        is_leaf = len(data["child_ids"]) == 0
        community_list.append({
            "id": data["id"], "title": data["title"], "level": data["level"],
            "parent_id": data["parent_id"], "child_ids": list(data["child_ids"]),
            "is_leaf": is_leaf
        })
    
    df_communities = pd.DataFrame(community_list)
    cols_comm = ["id", "title", "level", "parent_id", "child_ids", "is_leaf"]
    df_communities[cols_comm].to_parquet(OUTPUT_COMMUNITIES, index=False)
    print(f"保存成功: {OUTPUT_COMMUNITIES} (共 {len(df_communities)} 条)")
    
    # 2. TextUnits
    df_text_units = pd.DataFrame(text_units)
    cols_tu = [
        "id", "text", "n_tokens", "vector", "community_id", 
        "text_chunks", "chunk_vectors", 
        "head", "tail", "source_metadata", 
        "hierarchy_path", "prepend_source"
    ]
    df_text_units = df_text_units[cols_tu]
    df_text_units.to_parquet(OUTPUT_TEXT_UNITS, index=False)
    print(f"保存成功: {OUTPUT_TEXT_UNITS} (共 {len(df_text_units)} 条)")
    
    print("处理完成。")

if __name__ == "__main__":
    main()