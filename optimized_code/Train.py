
# 9.训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = AML_to_Graph("/kaggle/working")

data_train = dataset[0]
data_val = dataset[1]

# 【新增】：手动从磁盘把刚才独立保存的 mask 贴回到 data_val 上
mask_path = os.path.join(dataset.processed_dir, 'val_mask.pt')
data_val.val_mask = torch.load(mask_path, weights_only=False)

epoch = 121

each_batch = 256
criterion = torch.nn.BCEWithLogitsLoss()

# data.num_features节点特征数量  in_channels=29               原作没有加edge_dim=data.num_edge_features！！！！！！
model = ImprovedGAT(in_channels=data_train.num_features, hidden_channels=16, out_channels=1, heads=8,
            edge_dim=data_train.num_edge_features)
model = model.to(device)

# 改进三：优化器从SGD换成Adam！  SGD随机梯度下降
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)  # 优化器从SGD换成Adam！！！lr从0.0001改成0.001
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)  # 学习率衰减：每20轮缩小学习率，后期收敛更平稳

train_all_y = data_train.y  # 取出训练集所有节点的标签
dev = train_all_y.device
sample_weight_list = torch.where(train_all_y == 1,
                                 torch.tensor(5.0, device=dev),
                                 torch.tensor(1.0, device=dev))       # [1,20,1,1,1,20,1,1,1,1]比如10个权重，标签1的权重20
train_sampler = WeightedRandomSampler(weights=sample_weight_list, num_samples=data_train.num_nodes,   #或者num_samples=len(train_all_y)都ok
                                      replacement=True)  # replacement=False表示无放回，WeightedRandomSampler可设置洗钱客户的权重大小，以及可以设置无放回。但是

train_loader = NeighborLoader(data_train,
                              num_neighbors=[30] * 2,
                              batch_size=256,
                              sampler=train_sampler)  # 注意：指定 sampler 时不能设 shuffle=True
val_loader = NeighborLoader(data_val,
                            num_neighbors=[30] * 2,
                            batch_size=256,
                            input_nodes=data_val.val_mask)


total_batch = len(train_all_y) // each_batch  # each_batch=256，total_batch是一共多少个batch

best_val_auc = 0.0  # 初始化AUC
for ep in range(epoch):
    total_loss = 0
    model.train()

    for t in train_loader:  # total_batch：大约有1800个batch
        train_batch = t.to(device)
        optimizer.zero_grad()  # 梯度清零

        logits = model(train_batch.x, train_batch.edge_index,train_batch.edge_attr)  # def forward(self, x, edge_index, edge_attr):
        ground_truth = train_batch.y  # ground_truth一维张量
        train_seed_num = train_batch.batch_size  # 256
        seed_y = ground_truth[:train_seed_num]  # train_batch包含256种子+邻居，loss只计算256种子
        seed_logits = logits[:train_seed_num]

        loss = criterion(seed_logits, seed_y)  # FocalLoss要求输入两个张量维度一样的
        loss.backward()  # 算出所有w对应梯度，存入w.grad。      重要补充：有x和w，经过GAT卷积，得到p预测概率，用过FocalLoss和真实y，得到每一个节点的loss。要算每个节点梯度，只需要3个要素：x，真实y，w权重。

        # 新增梯度剪裁，防止梯度爆炸震荡
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()          # 用梯度w.grad优化一整套w权重（新）
        total_loss += loss.item()  # loss是GPU上的张量，.item()把它变成了浮点，便于计算。

    if ep % 10 == 0:
        print(f"Epoch:{ep:01d},本轮epoch每个训练集的batch平均loss:{total_loss / total_batch},每个epoch总批次数量:{total_batch}")
        model.eval()
        y_true_all = []
        y_prob_all = []
        with torch.no_grad():        # 关闭梯度计算
            for val in val_loader:  # 这里变量名是val_loader注意。
                val = val.to(device)
                logits = model(val.x, val.edge_index,val.edge_attr)  # 二维张量，pred=[[0.8],[0.65],[0.7],[0.1],[0.4]]类似，shape是（256，1）
                pred = F.sigmoid(logits)  # 压成概率
                ground_truth = val.y                                  # 一维张量，只有1和0的值，形状是[256,]   #val.y是包括这256个节点的邻居节点的真实标签！！！
                seed_num = val.batch_size  # 取到256
                gt_seed = ground_truth[:seed_num]
                pred_seed = pred[:seed_num]
                # 优化：找到最佳阈值
                # pred_label=(pred_seed>0.5).float()                    #先是布尔值，True或False然后转成1或0数字，得到二维张量[[1.0],[1.0],[1.0],[0.0],[1.0]]

                y_true_all.extend(gt_seed.cpu().numpy())               # gt_seed是一维张量[256,]    y_true_all所有客户的标签
                y_prob_all.extend(pred_seed.cpu().numpy().flatten())

        # 核心：利用验证集的预测概率动态寻找最优截断阈值
        y_true_arr = np.array(y_true_all)
        y_prob_arr = np.array(y_prob_all)
        if len(np.unique(y_true_arr)) > 1:
            precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
            # 优化：画出PR-AUC图
            pr_auc = auc(recalls, precisions)
            AP = average_precision_score(y_true_arr, y_prob_arr)  # 计算 PR-AUC（Area Under Precision-Recall Curve）
            roc_auc = roc_auc_score(y_true_all, y_prob_all)  # 这里是预测概率

            pos_ratio = np.sum(y_true_arr == 1) / len(y_true_arr)

            plt.figure(figsize=(8, 6))
            plt.plot(recalls, precisions, color="#1f77b4", lw=2, label=f"PR curve")
            plt.hlines(y=pos_ratio, xmin=0, xmax=1, color="red", linestyle="--", label="Random Baseline")
            plt.xlabel("Recall", fontsize=12)
            plt.ylabel("Precision", fontsize=12)
            plt.title("AML Model Precision-Recall cure", fontsize=14)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.legend(loc="best")
            plt.grid(alpha=0.3)
            plt.savefig(f"pr_curve_epoch{ep}.png", dpi=150)
            plt.close()

            beta = 4
            f_beta_scores = (1 + beta ** 2) * (precisions * recalls) / (beta ** 2 * precisions + recalls + 1e-10)
            best_idx = np.argmax(f_beta_scores)
            best_threshold = thresholds[best_idx]
            best_f_score = f_beta_scores[best_idx]  # F1

            # 3. 用找到的最佳阈值生成最终的 0/1 预测标签,此步骤的时候已经确定了阈值。
            y_pred_arr = (y_prob_arr >= best_threshold).astype(float)  # 这里要注意：一定是>=,因为precision_recall_curve内部是大于等于阈值才是洗钱，才计算的recall 和precision，所以要统一。

            acc = accuracy_score(y_true_arr, y_pred_arr)  # sklearn的这些指标必须输入一维数组
            precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
            recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)

            print(f"验证集Acc：{acc:.4f}|PR-AUC（AP): {AP:.4f}|PR-AUC: {pr_auc:.4f}|ROC-AUC:{roc_auc:.4f}")
            print(f"Best Threshold: {best_threshold:.4f}|Precision:{precision:.4f}|recall:{recall:.4f},F4 score:{best_f_score:.4f}")

            if roc_auc > best_val_auc:
                best_val_auc = roc_auc
                torch.save(model.state_dict(), 'best_aml_model.pth')
                print(f"🌟 发现更好的模型！已保存。当前最高 ROC-AUC: {best_val_auc:.4f}")

    scheduler.step()  # 每轮更新学习率，逐步缩小步长

