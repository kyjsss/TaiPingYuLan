import pandas as pd
import os

# --- 配置输入 ---
FILE_COMMUNITIES = "create_final_communities.parquet"
FILE_RELATIONSHIPS = "create_final_relationships_refined.parquet"

def export_vos_files(df_comm, df_rel, target_levels, file_prefix):
    """
    target_levels: list, 要包含的层级 ID
    """
    print(f"\n>>> 正在生成 {file_prefix} 图谱 (Code Levels: {target_levels})...")
    
    # 1. 筛选节点：只保留指定层级的节点
    valid_nodes_df = df_comm[df_comm['level'].isin(target_levels)]
    valid_node_ids = set(valid_nodes_df['id'])
    
    if len(valid_node_ids) == 0:
        print(f"  警告：层级 {target_levels} 下没有找到任何节点，跳过。")
        return

    # 2. 筛选边：两端都必须在指定层级内
    # 例如：Macro 图只保留 "子目-子目" 的连线，过滤掉 "子目-小类" 的连线
    mask = (df_rel['source_id'].isin(valid_node_ids)) & (df_rel['target_id'].isin(valid_node_ids))
    df_rel_filtered = df_rel[mask].copy()
    
    # 3. 提取活跃节点 (有连线的节点)
    active_node_ids = set(df_rel_filtered['source_id']) | set(df_rel_filtered['target_id'])
    
    print(f"  - 指定层级节点总数: {len(valid_node_ids)}")
    print(f"  - 最终活跃节点数 (有连线): {len(active_node_ids)}")
    print(f"  - 关联边数: {len(df_rel_filtered)}")

    if len(active_node_ids) == 0:
        print("  没有活跃节点，不生成文件。")
        return

    # ==========================
    # 准备数据映射 (ID -> Int, Root -> Cluster)
    # ==========================
    comm_dict = df_comm.set_index("id").to_dict(orient="index")
    uuid_to_int = {uuid: i+1 for i, uuid in enumerate(active_node_ids)}
    
    # 缓存 Level 0 (部) 标题，用于染色
    root_cache = {}
    def get_root_title(node_id):
        if node_id not in comm_dict: return "Unknown"
        if node_id in root_cache: return root_cache[node_id]
        
        node = comm_dict[node_id]
        # 如果自己是 Level 0
        if node['level'] == 0:
            res = node['title']
        # 如果有父节点
        elif node['parent_id']:
            res = get_root_title(node['parent_id'])
        else:
            res = "Unknown"
        root_cache[node_id] = res
        return res

    # 生成 Cluster ID
    all_roots = set()
    node_roots = {}
    for uuid in active_node_ids:
        r = get_root_title(uuid)
        node_roots[uuid] = r
        all_roots.add(r)
    root_to_cluster = {t: i+1 for i, t in enumerate(sorted(list(all_roots)))}

    # ==========================
    # 写入文件
    # ==========================
    node_file = f"{file_prefix}_map.txt"
    edge_file = f"{file_prefix}_network.txt"
    
    # 1. Map File
    with open(node_file, "w", encoding="utf-8") as f:
        f.write("id\tlabel\tdescription\tcluster\tweight\n")
        for uuid in active_node_ids:
            info = comm_dict[uuid]
            vos_id = uuid_to_int[uuid]
            label = info['title']
            
            root = node_roots[uuid]
            cluster_id = root_to_cluster.get(root, 0)
            
            # 描述显示层级：[Lv1] 天部 > 日
            desc = f"[Lv{info['level']}] {root} > {label}"
            
            # 权重设为 1.0 (让 VOSviewer 自己算密度)
            f.write(f"{vos_id}\t{label}\t{desc}\t{cluster_id}\t1.0\n")
            
    # 2. Network File
    with open(edge_file, "w", encoding="utf-8") as f:
        for _, row in df_rel_filtered.iterrows():
            src = uuid_to_int[row['source_id']]
            tgt = uuid_to_int[row['target_id']]
            w = row['weight']
            f.write(f"{src}\t{tgt}\t{w}\n")
            
    print(f"  成功! 请在 VOSviewer 加载: {node_file} 和 {edge_file}")

def main():
    if not os.path.exists(FILE_COMMUNITIES) or not os.path.exists(FILE_RELATIONSHIPS):
        print("错误：找不到 Parquet 文件。")
        return

    df_comm = pd.read_parquet(FILE_COMMUNITIES)
    df_rel = pd.read_parquet(FILE_RELATIONSHIPS)
    
    # === 方案 1: 宏观骨架图 (Macro) ===
    # 只包含 Code Level 1 (即您的 "子目" 层，如“日”、“月”)
    # 这张图会非常清晰，展示核心概念的连接
    export_vos_files(df_comm, df_rel, target_levels=[1], file_prefix="vos_macro")
    
    # === 方案 2: 全景细节图 (Full) ===
    # 包含 Code Level 1 + 2 (即 "子目" + "小类")
    # 展示所有层级的节点关系细节
    export_vos_files(df_comm, df_rel, target_levels=[1, 2], file_prefix="vos_full")

if __name__ == "__main__":
    main()