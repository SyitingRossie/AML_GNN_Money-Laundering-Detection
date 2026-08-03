
#  优化:用GATv2，最后实例化的是ImprovedGAT
from torch_geometric.nn import GATv2Conv
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


# 8.模型构建[ORIGINAL]
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

