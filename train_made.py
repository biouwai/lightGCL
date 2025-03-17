    df1 = pd.read_csv("newSet/drugSimMat.csv",header=None)
    df2 = pd.read_csv("newSet/LncDrug_edge.csv")
    df3 = pd.read_csv("newSet/MiDrug_edge.csv")
    df = pd.concat([df3, df2], ignore_index=True)

    ncRNA_list = sorted(df['ncRNA_Name'].unique())
    drug_list = sorted(df['Drug_Name'].unique())

    ncRNA_id_map = {name:i for i,name in enumerate(ncRNA_list)}
    drug_id_map = {name:i for i,name in enumerate(drug_list)}
    # 生成交互三元组
    rows = df['ncRNA_Name'].map(ncRNA_id_map).values
    cols = df['Drug_Name'].map(drug_id_map).values
    data1 = np.ones(len(df))  # 所有关联标记为1
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

    # 创建稀疏矩阵函数
    def create_sparse_matrix(indices, rows, cols, data1):
        return sp.coo_matrix(
            (data1[indices], (rows[indices], cols[indices])),
            shape=(len(ncRNA_list), len(drug_list)),
            dtype=np.float32
        )

    # 生成训练/测试矩阵
    train = create_sparse_matrix(train_idx, rows, cols, data1)
    test = create_sparse_matrix(test_idx, rows, cols, data1)

    train1 = train.copy()
    print('train:',train.sum())

    # 转换为CSR格式
    train_csr = train.tocsr()
    test_csr = test.tocsr()

    # 保存为txt文件
    def save_sparse(matrix, filename):
        with open(filename, 'w') as f:
            for row, col in zip(matrix.row, matrix.col):
                f.write(f"{row} {col} 1\n")

    save_sparse(train, 'dataset/rrtrain_x.txt')
    save_sparse(test, 'dataset/rrtest_x.txt')