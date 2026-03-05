import pandas as pd
import numpy as np
import os

# --- 配置 ---
FILE_RELATIONSHIPS = "create_final_relationships.parquet"
FILE_COMMUNITIES = "create_final_communities.parquet"

def main():
    print(">>> 正在加载数据进行图谱体检...")
    
    if not os.path.exists(FILE_RELATIONSHIPS) or not os.path.exists(FILE_COMMUNITIES):
        print("错误：找不到 Parquet 文件。")
        return

    df_rel = pd.read_parquet(FILE_RELATIONSHIPS)
    df_comm = pd.read_parquet(FILE_COMMUNITIES)
    
    # 建立 ID -> Title 映射，方便看人话
    id2title = df_comm.set_index("id")["title"].to_dict()

    # =======================================
    # 1. 基础规模 (Basic Stats)
    # =======================================
    total_edges = len(df_rel)
    # 参与连接的唯一节点数
    unique_nodes_in_edges = pd.unique(df_rel[['source_id', 'target_id']].values.ravel('K'))
    total_nodes_in_comm = len(df_comm)
    
    print(f"\n{'='*20} 基础规模 {'='*20}")
    print(f"总边数 (Relationships): {total_edges}")
    print(f"活跃节点数 (Connected Nodes): {len(unique_nodes_in_edges)} / {total_nodes_in_comm}")
    print(f"孤岛节点数 (Islands): {total_nodes_in_comm - len(unique_nodes_in_edges)} (占比: {(total_nodes_in_comm - len(unique_nodes_in_edges))/total_nodes_in_comm:.1%})")
    
    # =======================================
    # 2. 连通性分析 (Connectivity)
    # =======================================
    # 计算每个节点的度 (Degree)
    all_nodes = df_rel['source_id'].tolist() + df_rel['target_id'].tolist()
    node_counts = pd.Series(all_nodes).value_counts()
    
    avg_degree = total_edges / total_nodes_in_comm # 注意：这里分母用总节点数更客观
    
    print(f"\n{'='*20} 拓扑连通性 {'='*20}")
    print(f"平均度数 (Avg Degree): {avg_degree:.2f} (理想值 > 1.5)")
    print(f"最大度数 (Max Degree): {node_counts.max()}")
    print(f"中位数度数: {node_counts.median()}")
    
    print("\n[Top 10 枢纽节点 (Hubs)]")
    for nid, count in node_counts.head(10).items():
        title = id2title.get(nid, "未知节点")
        print(f"  - {title}: {count} 条连接")

    # =======================================
    # 3. 分数分布 (Score Distribution)
    # =======================================
    print(f"\n{'='*20} 权重/置信度分布 {'='*20}")
    print(df_rel['weight'].describe().round(2))
    
    # 分桶统计
    bins = [0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0.6-0.7 (低)', '0.7-0.8 (中)', '0.8-0.9 (高)', '0.9-1.0 (极高)']
    try:
        score_dist = pd.cut(df_rel['weight'], bins=bins, labels=labels, include_lowest=True).value_counts().sort_index()
        for label, count in score_dist.items():
            print(f"  {label}: {count}")
    except:
        pass

    # =======================================
    # 4. 语义类型 (Relation Types)
    # =======================================
    print(f"\n{'='*20} 关系类型分布 {'='*20}")
    type_counts = df_rel['relation_type'].value_counts()
    for rtype, count in type_counts.items():
        print(f"  - {rtype}: {count}")

    # =======================================
    # 5. 双向性分析 (Bidirectionality)
    # =======================================
    # 逻辑：检查 A->B 和 B->A 是否同时存在
    # 创建一个排序后的元组集合来识别“对”
    df_rel['sorted_pair'] = df_rel.apply(lambda x: tuple(sorted([x['source_id'], x['target_id']])), axis=1)
    pair_counts = df_rel['sorted_pair'].value_counts()
    
    bidirectional_pairs = pair_counts[pair_counts == 2].count()
    unidirectional_pairs = pair_counts[pair_counts == 1].count()
    
    print(f"\n{'='*20} 方向性分析 {'='*20}")
    print(f"双向互通 (Bidirectional): {bidirectional_pairs} 对 (共 {bidirectional_pairs*2} 条边)")
    print(f"单向影响 (Unidirectional): {unidirectional_pairs} 对 (共 {unidirectional_pairs} 条边)")
    print(f"双向比例: {bidirectional_pairs / (bidirectional_pairs + unidirectional_pairs):.1%} (双向对 / 总关系对)")

if __name__ == "__main__":
    main()