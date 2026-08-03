import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch.utils.data import WeightedRandomSampler   #ImbalancedSampler
from torch_geometric.nn import GATConv, Linear
from typing import Optional,Callable
from sklearn import preprocessing
import pandas as pd
from torch_geometric.data import InMemoryDataset,Data
import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,precision_recall_curve,auc)
import os
import random
import matplotlib.pyplot as plt
from torch_geometric.nn import GATv2Conv

#全量有515088个账户，508731个正常客户，6357个洗钱客户。
#如果按照交易时间排序后，前80%有513888个账户，除以256得到2007.375个batch。

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 换成 GATv2
class ImprovedGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, edge_dim=None):
        super().__init__()

        # 第一层：多头注意力，捕捉多维度拓扑与交易特征
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,  # 拼接多头输出，输出维度是 hidden_channels * heads
            edge_dim=edge_dim,
            dropout=0.3  # 适当加入 dropout 防止过拟合
        )

        # 验证第一层输出后的维度计算：hidden_channels * heads = 16 * 8 = 128
        conv1_output_dim = hidden_channels * heads

        # 第二层：收敛多头，转化为最终的预测 logit
        self.conv2 = GATv2Conv(
            in_channels=conv1_output_dim,
            out_channels=out_channels,
            heads=1,
            concat=False,  # 第二层通常 concat=False（求平均），直接压维
            edge_dim=edge_dim,
            dropout=0.3
        )

    def forward(self, x, edge_index, edge_attr=None,return_embedding=False):#“默认情况下，这个开关是关着的（False）。调用函数的人如果不专门传这个参数，我就当作 False 来处理。”
        # 当你需要抽特征去给 XGBoost 做模型融合时，你手动传入 True：# 手动把开关打开！
# embeddings = model(data.x, data.edge_index, data.edge_attr, return_embedding=True)
        # 第一层卷积 + 激活 + Dropout
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        if return_embedding:
            return x

        # 第二层卷积（输出未激活的 Logits，交由 BCEWithLogitsLoss 处理）
        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        return x


# 8.模型构建
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, edge_dim):
        super().__init__()
        self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=heads, dropout=0.3,
                             add_self_loops=True,
                             edge_dim=edge_dim)  # 卷积加入自生原始特征！！！#dropout从0.6改成0.3 ,原作者没有加入边特征！！！！！！！！！！！！1
        self.conv2 = GATConv(hidden_channels * heads, int(hidden_channels / 4), heads=1,
                             dropout=0.3,
                             edge_dim=edge_dim)  # concat=False，这句没加，因为concat只有在heads>1时才有区别。concat=True是默认的，横向拼接所有头的特征，如果是False，则是取平均，不拼接。
        self.lin = Linear(int(hidden_channels / 4), out_channels)  # 从4到1
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):  # edge_index是二维张量，第一行是from的账户index，第二行是to的账户index。   输入x：是节点特征
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        logits = self.lin(x)
        # x=self.sigmoid(x)                          #删除
        return logits.squeeze(dim=1)


class AML_to_Graph(InMemoryDataset):
    def __init__(self, root, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):  # 原有edge_window_size=10,
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)  # 这里加weights_only=False
        # self.edge_window_size = edge_window_size

    @property
    def raw_file_names(self):
        return ["HI-Small_Trans.csv"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def df_label_encoder(self, df, columns):  # 先定义，之后用
        le = preprocessing.LabelEncoder()
        for i in columns:
            df[i] = le.fit_transform(df[i])
        return df

    # 3.数据处理:时间列归一化，收款货币、付款货币、付款方式这三列进行编码，同时拆分成2张表，收款表，付款表。
    def preprocess(self, df):
        # 收付货币统一编码，同一套mapping
        df = df.copy()
        le_curr = preprocessing.LabelEncoder()
        all_curr = pd.concat([df["Payment Currency"], df["Receiving Currency"]])  # 默认上下堆叠axis=0
        le_curr.fit(all_curr)
        df["Payment Currency"] = le_curr.transform(df["Payment Currency"])
        df["Receiving Currency"] = le_curr.transform(df["Receiving Currency"])
        mapping_curr = {cls: idx for idx, cls in
                        enumerate(le_curr.classes_)}  # le_curr.classes_ 得到类似array(['USD','RMB',,,,])一维数组
        currency_ls = [idx for idx, cls in enumerate(le_curr.classes_)]

        # 支付渠道单独编码
        le_form = preprocessing.LabelEncoder()
        df["Payment Format"] = le_form.fit_transform(df["Payment Format"])
        mapping_form = {cls: idx for idx, cls in enumerate(le_form.classes_)}
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Timestamp"] = df["Timestamp"].astype(int)  # 备注：不用apply(lambda x:x.value)开销太大。另，张量不支持datetime时间类型，所以要转成底层数字
        df["Timestamp"] = (df["Timestamp"] - df["Timestamp"].min()) / (
                df["Timestamp"].max() - df["Timestamp"].min())  # 归一化
        df["Account"] = df["From Bank"].astype(str) + "_" + df["Account"]
        df["Account.1"] = df["To Bank"].astype(str) + "_" + df["Account.1"]
        df = df.sort_values(by=["Account"])  # 注意：这里按照account排序了！！！
        receiving_df = df[["Account.1", "Amount Received", "Receiving Currency"]]
        paying_df = df[["Account", "Amount Paid", "Payment Currency"]]
        receiving_df = receiving_df.rename(
            columns={"Account.1": "Account"})  # 备注：不写columns=的话，就写axis=1，如果是行，要么写index=，要么写axis=0

        return df, paying_df, receiving_df, currency_ls  # currency_ls=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # 4.列出所有账户，去重，打上洗钱标签。（思路很清楚）
    def get_all_accounts(self, df):
        ldf = df[["Account", "From Bank"]]
        rdf = df[["Account.1", "To Bank"]]

        suspicious = df[df["Is Laundering"] == 1]
        s1 = suspicious[["Account", "Is Laundering"]]
        s2 = suspicious[["Account.1", "Is Laundering"]]
        s2 = s2.rename(columns={"Account.1": "Account"})
        suspicious = pd.concat([s1, s2])  # 默认上下堆叠axis=0
        suspicious = suspicious.drop_duplicates()

        ldf = ldf.rename(columns={"From Bank": "Bank"})
        rdf = rdf.rename(columns={"To Bank": "Bank", "Account.1": "Account"})
        df = pd.concat([ldf, rdf])
        df = df.drop_duplicates()
        df["Is Laundering"] = 0
        df.set_index("Account", inplace=True)  # 默认drop=True，原来account列会被删除，只留作行索引。
        df.update(suspicious.set_index("Account"))  # 这句很重要！！！！！！！！！！！
        df = df.reset_index()
        return df

    # 5.站点特征 Node Features:首先从accounts所有账户的表【Account、Bank、IS Laundering列】准备node_df表【account、bank（编译后）、avg paid 0、avg paid 1、avg paid 2、……、avg received 0、avg received 1、avg received 2……】
    def paid_currency_aggregate(self, currency_ls, paying_df, accounts):
        acc_df = accounts.copy()
        for i in currency_ls:
            col_name = f"avg_paid_{i}"
            temp = paying_df[paying_df["Payment Currency"] == i]
            if temp.empty:  # 额外加固，防止如果全量的currency_ls全货币中，有货币是payment Currency没有的
                acc_df[col_name] = 0
                continue
            account_mean = temp.groupby("Account")["Amount Paid"].mean().reset_index()
            account_mean = account_mean.rename(columns={"Amount Paid": col_name})  # 改变步骤：先重命名后再merge，防止bug
            acc_df = acc_df.merge(account_mean, on="Account", how="left")
        # 改：节点特征每个货币的均值进行取对数压缩
        paid_cols = [f"avg_paid_{i}" for i in currency_ls]
        acc_df[paid_cols] = np.log1p(acc_df[paid_cols].fillna(0))
        return acc_df

    def received_currency_aggregate(self, currency_ls, receiving_df, accounts):
        acc_df = accounts.copy()
        for i in currency_ls:
            col_name = f"avg_received_{i}"
            temp = receiving_df[receiving_df["Receiving Currency"] == i]
            if temp.empty:
                acc_df[col_name] = pd.NA
                continue
            account_mean = temp.groupby("Account")["Amount Received"].mean().reset_index()
            account_mean = account_mean.rename(columns={"Amount Received": col_name})  # 改变步骤：先重命名后再merge，防止bug
            acc_df = acc_df.merge(account_mean, on="Account", how="left")
        acc_df = acc_df.fillna(0)
        # 改：节点特征每个货币的均值进行取对数压缩
        received_cols = [f"avg_received_{i}" for i in currency_ls]
        acc_df[received_cols] = np.log1p(acc_df[received_cols])
        return acc_df

    # 改进：在特征节点新增3列，账户的资金来源账户个数，账户的出钱账户个数，总交易笔数
    # 改进：在特征节点新增宏观拓扑统计特征
    def topological_aggregate(self, df, node_df):
        acc_df = node_df.copy()

        out_deg = df.groupby("Account")["Account.1"].nunique().reset_index(name="unique_out_accounts")
        in_deg = df.groupby("Account.1")["Account"].nunique().reset_index(name="unique_in_accounts")
        in_deg = in_deg.rename(columns={"Account.1": "Account"})

        out_cnt = df.groupby("Account").size().reset_index(name="total_out_count")
        in_cnt = df.groupby("Account.1").size().reset_index(name="total_in_count")
        in_cnt = in_cnt.rename(columns={"Account.1": "Account"})

        acc_df = acc_df.merge(out_deg, on="Account", how="left")
        acc_df = acc_df.merge(in_deg, on="Account", how="left")
        acc_df = acc_df.merge(out_cnt, on="Account", how="left")
        acc_df = acc_df.merge(in_cnt, on="Account", how="left")

        topo_cols = ["unique_out_accounts", "unique_in_accounts", "total_out_count", "total_in_count"]
        acc_df[topo_cols] = acc_df[topo_cols].fillna(0)

        acc_df["avg_T_out"] = (acc_df["total_out_count"] / acc_df["unique_out_accounts"]).fillna(0)
        acc_df["avg_T_in"] = (acc_df["total_in_count"] / acc_df["unique_in_accounts"]).fillna(0)

        density_cols = ["avg_T_out", "avg_T_in"]
        acc_df[density_cols] = np.log1p(acc_df[density_cols])
        acc_df[topo_cols] = np.log1p(acc_df[topo_cols])  # 对数压缩：防止极端大户或洗钱中转站的数值过大导致梯度爆炸

        return acc_df

    # 6.node features
    def get_node_attr(self, currency_ls, paying_df, receiving_df, accounts, df):  # 这里加了df参数
        node_df = self.paid_currency_aggregate(currency_ls, paying_df, accounts)
        node_df = self.received_currency_aggregate(currency_ls, receiving_df, node_df)

        # 改进：在特征节点新增4列，账户的资金来源账户个数，账户的出钱账户个数，总交易笔数
        node_df = self.topological_aggregate(df, node_df)

        ll = node_df['Is Laundering'].to_numpy()  # 开始提取
        node_label = torch.tensor(ll, dtype=torch.float)
        del ll
        node_df = node_df.drop(["Account", "Is Laundering"], axis=1)
        node_df = self.df_label_encoder(node_df, ["Bank"])
        arr = node_df.to_numpy()
        node_x = torch.tensor(arr, dtype=torch.float)
        del arr
        return node_x, node_label  # accounts_node  # accounts_node是宽表，这里就是作者的accounts宽表。额外输出accounts_node

    # 7.Edge features
    def get_edge_attr(self, accounts, df):
        accounts = accounts.reset_index(drop=True)  # 确保从0开始编号，把所有唯一账户重新编码，丢弃原索引。谨慎。
        accounts["ID"] = accounts.index
        mapping_dict = dict(zip(accounts["Account"], accounts["ID"]))  ## 这里account代入的是account_node宽表，这注释不要看了。
        df["From"] = df["Account"].map(mapping_dict)
        df["To"] = df["Account.1"].map(mapping_dict)

        # 备注：边特征['Timestamp', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format'],边特征是6列。
        # 改进：把Account Paid和Account Received 进行取对数压缩np.log1p()
        amount_cols = ["Amount Paid", "Amount Received"]
        for col in amount_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        df = df.drop(["Account", "Account.1", "From Bank", "To Bank"], axis=1)
        from_np = df["From"].to_numpy()
        from_arr = torch.tensor(from_np, dtype=torch.long)
        del from_np
        to_np = df["To"].to_numpy()
        to_arr = torch.tensor(to_np, dtype=torch.long)  # 交易边标签需要到张量的时候是整数。
        del to_np
        edge_index = torch.stack([from_arr, to_arr],
                                 dim=0)  # dim=0上下堆叠    #注意，PyTorch Geometric硬性要求二维张量形状[2,交易条数]]，结果就是第一行都是from的账户index，第二行都是to的账号index
        df = df.drop(["From", "To"], axis=1)
        edge_np = df.to_numpy()
        edge_attr = torch.tensor(edge_np, dtype=torch.float)
        del edge_np
        return edge_attr, edge_index  # edge_attr的账户和银行的信息都去掉了，edge_index是“账户+银行”唯一编码的序号，从0开始。

    def process(self):
        # df=pd.read_csv(self.raw_paths[0],nrows=10000)                      #替换!!!!!!分块取df
        chunk_list = []
        batch_size = 100000
        reader = pd.read_csv(self.raw_paths[0], chunksize=batch_size)
        for chunk in reader:
            chunk_list.append(chunk)
        df = pd.concat(chunk_list, axis=0, ignore_index=True)

        df, paying_df, receiving_df, currency_ls = self.preprocess(df)

        df = df.sort_values(by="Timestamp").reset_index(drop=True)    #新增,把所有交易数据按照交易时间排序，然后再划分80%训练，20%测试集
        split_idx = int(len(df) * 0.8)                                #新增
        df_train = df.iloc[:split_idx].copy()

        df_test = df.iloc[split_idx:].copy()
        test_accounts_set = set(df_test["Account"]).union(set(df_test["Account.1"])) #这里set，union都有去重效果，并集。

        accounts_train = self.get_all_accounts(df_train)
        node_attr_train, node_label_train = self.get_node_attr(currency_ls,        #修改，防止泄露给验证集
                                                   df_train[["Account", "Amount Paid", "Payment Currency"]],
                                                   df_train[["Account.1", "Amount Received", "Receiving Currency"]].rename(columns={"Account.1":"Account"}),
                                                   accounts_train,
                                                   df_train)
        edge_attr_train, edge_index_train = self.get_edge_attr(accounts_train, df_train)

        data_train = Data(
            x=node_attr_train,
            edge_index=edge_index_train,
            y=node_label_train,
            edge_attr=edge_attr_train
        )

        #构建全量表
        accounts = self.get_all_accounts(df)
        node_attr_val, node_label_val = self.get_node_attr(currency_ls,paying_df, receiving_df,accounts,df)
        edge_attr_val, edge_index_val = self.get_edge_attr(accounts, df)

        # 根据全局 accounts 表的顺序，把在后 20% 交易中出现过的账户设为 True
        eval_mask = torch.tensor(
            accounts["Account"].isin(test_accounts_set).values,
            dtype=torch.bool
        )
        data_val = Data(
            x=node_attr_val,
            edge_index=edge_index_val,
            y=node_label_val,
            edge_attr=edge_attr_val,
        )

        data_list = [data_train, data_val]   #把2张图打包

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)  # 如果多张图用，这里只有1张图
        torch.save((data, slices), self.processed_paths[0])

        # 【新增】：把 eval_mask 单独存为一个独立的文件在 processed 目录下
        torch.save(eval_mask, os.path.join(self.processed_dir, 'val_mask.pt'))
