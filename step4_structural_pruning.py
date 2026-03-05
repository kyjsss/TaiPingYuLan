import pandas as pd
import numpy as np
import networkx as nx
import os

# --- 配置 ---
INPUT_RELATIONSHIPS = "create_final_relationships.parquet"
OUTPUT_FINAL_RELATIONSHIPS = "create_final_relationships_refined.parquet"

# 参数
KEEP_WEIGHT_THRESHOLD = 0.8  # 权重 >= 0.8 的强制保留
STD_MULTIPLIER = 3.0         # 异常值检测阈值 (Mean + 3 * Std)
MAX_PATH_LENGTH = 3          # 【新增】最大替代路径长度。超过此长度不算"冗余"，必须保留直连。

def main():
    print(">>> 开始图谱拓扑结构精炼 (Topology-Aware Pruning)...")
    
    if not os.path.exists(INPUT_RELATIONSHIPS):
        print("错误：找不到输入文件。")
        return

    df = pd.read_parquet(INPUT_RELATIONSHIPS)
    original_count = len(df)
    print(f"原始边数: {original_count}")
    
    # 1. 构建 NetworkX 图对象 (无向图)
    G = nx.from_pandas_edgelist(df, 'source_id', 'target_id', edge_attr=['weight'])
    
    print(f"图构建完成。节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    # 2. 识别 Super Hubs (Mean + 3 * Std)
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    mean_deg = np.mean(degree_values)
    std_deg = np.std(degree_values)
    cap_limit = mean_deg + (STD_MULTIPLIER * std_deg)
    
    print(f"\n[Hub 检测标准]")
    print(f"平均度数: {mean_deg:.2f}")
    print(f"标准差: {std_deg:.2f}")
    print(f"封顶阈值 (Cap Limit): {cap_limit:.2f}")
    
    super_hubs = [n for n, d in degrees.items() if d > cap_limit]
    print(f"检测到 {len(super_hubs)} 个超级枢纽 (Super Hubs):")
    # 打印前 5 个及其度数
    hub_details = sorted([(n, degrees[n]) for n in super_hubs], key=lambda x: x[1], reverse=True)
    for n, d in hub_details[:5]:
        print(f"  - Node ID: {n[:8]}... Degree: {d}")

    if not super_hubs:
        print("未发现异常 Hub，无需剪枝。")
        df.to_parquet(OUTPUT_FINAL_RELATIONSHIPS, index=False)
        return

    # 3. 执行智能剪枝逻辑
    edges_to_remove = set()
    pruned_stats = {"safe_kept": 0, "bridge_kept": 0, "redundant_removed": 0}

    print(f"\n开始对 {len(super_hubs)} 个 Hub 进行结构化瘦身...")
    
    for hub in super_hubs:
        # 获取该 Hub 的所有邻居
        neighbors = list(G.neighbors(hub))
        
        # 获取边数据并按权重升序排序 (优先审查低分边)
        hub_edges = []
        for n in neighbors:
            weight = G[hub][n]['weight']
            hub_edges.append((n, weight))
        
        hub_edges.sort(key=lambda x: x[1])
        
        for neighbor, weight in hub_edges:
            # 规则 1: 强制保留高分边 (尊重 LLM 的绝对判断)
            if weight >= KEEP_WEIGHT_THRESHOLD:
                pruned_stats["safe_kept"] += 1
                continue
            
            # 规则 2: 拓扑检测
            # A. 暂时移除这条直连边
            G.remove_edge(hub, neighbor)
            
            # B. 检查是否存在"较短"的替代路径
            try:
                # 只有当 3 跳以内能连通时，我们才认为这条直连是"冗余"的
                # 如果要绕太远 (比如 > 3)，说明直连很有必要，不能删
                shortest_len = nx.shortest_path_length(G, source=neighbor, target=hub)
                
                if shortest_len <= MAX_PATH_LENGTH:
                    # 存在短路径替代，说明是冗余边 -> 确认删除
                    # 记录到删除列表 (排序 tuple 以保证 frozenset 一致性)
                    edges_to_remove.add(frozenset([hub, neighbor]))
                    pruned_stats["redundant_removed"] += 1
                    # 【注意】这里不加回去了，因为我们要基于"已删除"的状态继续判断后续的边
                else:
                    # 路径太远，恢复直连
                    G.add_edge(hub, neighbor, weight=weight)
                    pruned_stats["bridge_kept"] += 1
                    
            except nx.NetworkXNoPath:
                # 根本不连通，恢复直连 (桥梁)
                G.add_edge(hub, neighbor, weight=weight)
                pruned_stats["bridge_kept"] += 1

    print("\n[剪枝统计]")
    print(f"  - 高分强制保留 (Score >= {KEEP_WEIGHT_THRESHOLD}): {pruned_stats['safe_kept']}")
    print(f"  - 结构性桥梁保留 (无短路径替代): {pruned_stats['bridge_kept']}")
    print(f"  - 冗余捷径移除 (存在替代路径): {pruned_stats['redundant_removed']}")

    # 4. 同步回 DataFrame 并保存
    if edges_to_remove:
        print("正在应用剪枝结果...")
        # 定义过滤逻辑
        def should_keep(row):
            edge_set = frozenset([row['source_id'], row['target_id']])
            return edge_set not in edges_to_remove

        df_refined = df[df.apply(should_keep, axis=1)]
    else:
        df_refined = df.copy()
        
    # 5. 导出
    final_cols = ["source_id", "target_id", "weight", "relation_type", "description", "vector_score"]
    # 过滤掉不存在的列
    available_cols = [c for c in final_cols if c in df_refined.columns]
    df_refined = df_refined[available_cols]
    
    df_refined.to_parquet(OUTPUT_FINAL_RELATIONSHIPS, index=False)
    
    print(f"\n>>> 治理完成！")
    print(f"最终边数: {len(df_refined)} (减少了 {original_count - len(df_refined)})")
    
    # 简单回检 Top 1 节点的新度数
    if len(df_refined) > 0 and super_hubs:
        top_hub = hub_details[0][0]
        # 计算该 Hub 在新表中的出现次数
        new_degree = len(df_refined[(df_refined['source_id'] == top_hub) | (df_refined['target_id'] == top_hub)])
        print(f"\n[Hub 治理效果]")
        print(f"Top 1 Node ({top_hub[:8]}...) 度数变化: {hub_details[0][1]} -> {new_degree}")

if __name__ == "__main__":
    main()