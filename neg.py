import random

def load_positive_samples(filename):
    """
    从 .txt 文件加载正样本。
    
    参数:
        filename (str): 文件路径。
    
    返回:
        positive_pairs (set): 正样本的 (RNA_index, Drug_index) 元组集合。
        rows (list): RNA 节点索引列表。
        cols (list): 药物节点索引列表。
    """
    positive_pairs = set()
    rows, cols = [], []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:  # 确保每行有三列
                row, col, _ = map(int, parts)
                positive_pairs.add((row, col))
                rows.append(row)
                cols.append(col)

    return positive_pairs, rows, cols


def generate_negative_samples(positive_pairs, num_RNA, num_drug, num_negatives):
    """
    随机生成负样本，确保不与正样本冲突。
    
    参数:
        positive_pairs (set): 正样本的 (RNA_index, Drug_index) 元组集合。
        num_RNA (int): RNA 节点数量。
        num_drug (int): 药物节点数量。
        num_negatives (int): 负样本的数量。
    
    返回:
        negative_rows (list): 负样本的 RNA 节点索引列表。
        negative_cols (list): 负样本的药物节点索引列表。
    """
    negative_pairs = set()
    max_attempts = 10 * num_negatives  # 最大尝试次数，防止死循环

    while len(negative_pairs) < num_negatives and max_attempts > 0:
        rna_idx = random.randint(0, num_RNA - 1)
        drug_idx = random.randint(0, num_drug - 1)
        pair = (rna_idx, drug_idx)
        if pair not in positive_pairs:
            negative_pairs.add(pair)
        max_attempts -= 1

    if len(negative_pairs) < num_negatives:
        raise ValueError("无法生成足够的负样本，请检查输入数据或参数。")

    negative_rows, negative_cols = zip(*negative_pairs) if negative_pairs else ([], [])
    return list(negative_rows), list(negative_cols)


def save_samples_to_file(rows, cols, labels, filename):
    """
    将正负样本保存为 .txt 文件。
    
    参数:
        rows (list): RNA 节点索引列表。
        cols (list): 药物节点索引列表。
        labels (list): 样本标签（1 表示正样本，0 表示负样本）。
        filename (str): 输出文件路径。
    """
    with open(filename, 'w') as f:
        for row, col, label in zip(rows, cols, labels):
            f.write(f"{row} {col} {label}\n")


# 示例：生成包含正负样本的文件
if __name__ == "__main__":
    # 输入文件路径
    input_filename = "dataset/rrtest_x.txt"
    output_filename = "dataset/rrtest_x_n.txt"

    # 假设 RNA 节点数量为 1927，药物节点数量为 216（从文件最后一行的形状信息推断）
    num_RNA = 1927
    num_drug = 216

    # 加载正样本
    positive_pairs, rows, cols = load_positive_samples(input_filename)

    # 生成负样本（数量与正样本相同）
    num_negatives = len(rows)
    negative_rows, negative_cols = generate_negative_samples(positive_pairs, num_RNA, num_drug, num_negatives)

    # 合并正负样本
    all_rows = rows + negative_rows
    all_cols = cols + negative_cols
    all_labels = [1] * len(rows) + [0] * len(negative_rows)

    # 打印结果
    print(f"Number of positive samples: {len(rows)}")
    print(f"Number of negative samples: {len(negative_rows)}")

    # 保存正负样本到文件
    save_samples_to_file(all_rows, all_cols, all_labels, output_filename)

    print(f"Processed file saved to {output_filename}")