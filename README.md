# Anti-Money Laundering Detection with GNN (Account-Level)
基于图神经网络的反洗钱账户识别项目，由开源基线迭代优化而来。本项目针对Kaggle开源基线进行全方位的重构，重点解决了传统GNN在金融风控场景中的时序数据泄露、特征工程疏漏、样本极度不平衡等核心缺陷。


# 一、基本信息
1、初次获取渠道：Kaggle Notebook 

2、原作Kaggle链接：https://www.kaggle.com/code/issacchanjj/anti-money-laundering-detection-with-gnn/notebook

3、原作GitHub仓库链接：https://github.com/issacchan26/AntiMoneyLaunderingDetectionWithGNN/tree/main

4、项目结构：dataset.py、model.py、train.py

5、使用IBM数据集：HI-Small_Trans.csv  

6、原始数据集字段：【'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1','Amount Received', 'Receiving Currency', 'Amount Paid','Payment Currency', 'Payment Format', 'Is Laundering'】  其中'Is Laundering'是标签，1=洗钱客户，0=正常客户

7、文件结构：
代码分为两个版本
- baseline_source: 第一版基线GNN代码（原demo）
- optimized_code: 本人优化版本



# 二、核心优化
**1.严格时序防泄露设计**

**基线痛点**：原demo是基于所有交易统计节点标签和节点特征后，再进行随机切分训练集和验证集，忽略了金融风控实际场景中的时间先后顺序。

(1)、特征泄露：训练集和验证集均采用全局的节点特征，训练集的节点已经融合了验证集的交易数据，容易出现验证集指标虚高，模型在真实线下业务场景推理时效果大幅滑坡，泛化能力差。

(2)、标签泄露：训练集和验证集的标签都是全局标签。例如：a节点全局有过洗钱交易标签，那么该节点标签始终是1。反之，正常客户也是如此，导致训练阶段提前知道了未来的洗钱结果。

(3)、拓扑泄露：由于交易边用的是一张全局的图，训练集卷积了未来时序生成的交易边，致使训练节点利用了未来资金流向的网络结构。

**优化方案**：采用时序窗口切分（交易按时间排序后再切分）

(1)、严格时序切分：训练图基于0-70%（观测窗口）交易构建，预测70-85%（预测窗口）交易时段的账户洗钱标签；验证图基于0-85%交易构建，预测85-100%交易时段的账户标签。

(2)、独立构图：训练、验证阶段独立构图，独立统计节点特征，交易边特征，交易标签。

(3)、主节点筛选：限定训练目标节点为在预测窗口内有活动且在观察期有拓扑历史节点，暂不考虑冷启动节点。



**2.特征工程重构：**

**基线痛点**：原demo节点特征仅由15种货币的进款均值、出款均值，以及bank列（LabelEncoder编码）构成（31维稀疏特征），存在严重的特征表达缺陷。

(1)、未量纲化且分布偏态：交易金额绝对值跨度大，直接使用原始金额的均值，会引发梯度震荡。

(2)、缺乏风控聚合意义：反洗钱的核心识别对象是拓扑层面的资金异常流动模式（如“多对一汇聚”、“快进快出”、“大额分拆”等），并非单一币种的静态金额。

(3)、类别编码缺陷：bank列使用LabelEncoder编码转化为连续整数,无意义地引入了顺序关系和数值大小关系。

**优化方案**：

摒弃原31维稀疏特征，将节点特征重构为 **4 大风控业务维度、共 12 维稠密特征**，并在各时序窗口内独立进行 $\log1p$ 极值平滑与 `StandardScaler` 标准化处理：

(1)、资金规模与体量（平滑长尾分布，消除大额量纲干扰）

total_amount_paid：总支出金额   
total_amount_received：总接收金额

avg_amount_paid： 笔均付出金额  
avg_amount_received：笔均接收金额

(2)、资金流动与留存（捕捉“快进快出”与高复杂度洗钱）

net_flow_ratio：资金净流转率，公式=(Rec−Paid)/(Rec+Paid+1e−5)
unique_currency_count：涉及交易货币种类去重总数

(3)、图拓扑与交互度（识别“归集”与“分发”网络角色）

unique_out_accounts：出度去重对手数 (Out-Degree)  
unique_in_accounts：入度去重对手数 (In-Degree)

(4)、行为频次与时序（捕捉高频拆单与自动化定时归集）

total_out_count：总付款笔数
total_in_count：总收款笔数

avg_T_out：平均转出时间间隔频次
avg_T_in：平均转入时间间隔频次

1. total_amount_paid      : 总付出金额 (对数压缩)
2. total_amount_received  : 总接收金额 (对数压缩)
3. avg_amount_paid        : 笔均付出金额 (对数压缩)
4. avg_amount_received    : 笔均接收金额 (对数压缩)
5. net_flow_ratio         : 资金净流转率 (Received - Paid) / (Received + Paid + 1e-5)
6. unique_currency_count  : 涉及货币种类总数 (衡量交易复杂度)
7. unique_out_accounts    : 出度去重数
8. unique_in_accounts     : 入度去重数
9. total_out_count        : 总付款笔数
10. total_in_count        : 总进款笔数
11. avg_T_out             : 平均转出频次
12. avg_T_in              : 平均转入频次
### 2. 特征工程重构：12 维高稠密风控特征矩阵

彻底废弃原基线 31 维稀疏且带数值偏差的特征，将节点特征重构为 **4 大风控业务维度、共 12 维稠密特征**，并在各时序窗口内独立进行 $\log1p$ 极值平滑与 `StandardScaler` 标准化处理：

(3)、交易边的时间戳归一化（MinMaxScaler）处理中，max依赖于未来的测试集，除了数据泄露外，将时间归一化，这种处理方式在实际模型中几乎没有直接的预测意义。上线后面对未来的新交易，[0,1]映射会直接溢出失效。

(4)、边特征金额缺乏汇率折算，不同货币的交易金额实际差异大。

3、缺乏对样本数据不平衡的处理

(1)、原是在epoch循环之前一次性随机采样，那每轮epoch样本顺序都一样，多轮后容易过拟合。

4、GAT模型构建：
原设计中应有边特征，但在模型里未加edge_dim参数，致使模型没有学习到交易边特征。
原设计没有使用自还边，会导致卷积过程中没有融合节点自身的特征信息。

5、优化器选择：原用SGD随机梯度下降。

6、缺乏模型验证指标：在极度不平衡的数据集中仅使用Accuracy是无意义的(全预测为0就能达到99%的准确率）。

# 四、本人优化方案（核心任务：节点分类（账户级））


2、特征工程优化：

(1)、在节点特征中，对各类货币进款均值、和出款均值，进行取对数压缩log1p，防止极端大户或洗钱中转站的数值过大导致梯度爆炸。

(2)、在节点特征新增宏观拓扑统计特征，新增6列，并取对数压缩：

A.结构拓扑特征（出度与入度）：（该账户转给多少个不重复的目标账户/多少个不重复的来源账户向该账户转账）

B.交易频次特征（总笔数）：（该账户发起的总转出笔数/接收的总转入笔数）

C.关联密度特征（平均单对单交易频次）：（该账户发起的总转出笔数/接收的总转入笔数）

(3)、时间戳处理：仅在训练集（0-70%）上计算时间戳的起始基准点(min),通过平移操作将其转换为”距离历史窗口起点的相对秒数“，避免了未来信息泄露，且缩小了数值尺度。

(4)、交易边金额进行汇率折算，统一维度表。

3、不平衡样本加权采样：

(1)、采用WeightedRandomSampler加权有放回采样：洗钱节点设置采样权重=5，正常账户权重=1。WeightedRandomSampler为迭代器，每轮Epoch重新执行一次随机采样，避免模型过拟合到固定的采样结构。

1. 损失函数补偿：计算了训练集中的正负样本比例，并传入 BCEWithLogitsLoss 的 pos_weight（设置了最大截断值 15 防止梯度剧烈震荡）。
2. 评估指标选择：放弃了容易受负样本主导的 ROC-AUC，转而以 PR-AUC (Average Precision) 为核心监控指标。
3. 动态阈值选取：由于反洗钱业务更倾向于‘宁可错杀、不能放过’，我在验证集上通过最大化 $F_4$-Score（给 Recall 赋以 4 倍于 Precision 的权重）来动态搜索最佳决策阈值。”

4、GAT卷积网络搭建：

(1)、GATConv中加入edge_dim参数，使模型没有学习交易边特征；加入自环边add_self_loops=True，加入自环边的交易边信息fill_value="mean"。

(2）、用GATv2Conv替代GATConv。


5、优化器：优化器从SGD换成Adam。

6、完善验证指标：增加PR-AUC（AP)、ROC-AUC、recall、precision、F4 score 


五、其他优化设想

目前ImprovedGAT.forward 中预留了return_embedding=True 开关，下一步可以写一段代码：抽取 GNN 的节点 Embedding，拼上原始节点的 tabular 特征，喂给 XGBoost / LightGBM 做二阶段分类。

六、附注

原Demo的图：

(1）、所有节点表（共2列）：Account（”银行_账号“）、Bank（LabelEncoder编码）

(2)、节点特征表(31列)：Bank（LabelEncoder编码）、avg_paid_0 ~ avg_paid_14、avg_received_0 ~ avg_received_14

(3)、交易边特征表(6列）：Timestamp（归一化）、Amount Received（对数压缩）、Receiving Currency（LabelEncoder编码）、Amount Paid（对数压缩）、Payment Currency（LabelEncoder编码）、Payment Format（LabelEncoder编码）

(4)、边的连接关系(2列）：from_np、to_np（节点账号排序后映射为0~N-1的整数节点ID）

改进：

(1）、节点表：维持原样

(2)、节点特征：

将原始31列稀疏节点特征，改进为12维度高信息密度特征（同时去掉Bank列），并用standardscale归一化。



(3)、交易边特征：

1. Payment Currency       : 热独编码
2. is_cross_currency      : 是否跨币种交易（1/0)
3. Amount_USD_log         : 交易金额（统一成美元并取对数）
4. hour_sin,hour_cos      : 小时周期性编码（将 0-23 点映射至单位圆坐标，消除跨日数值断层）
5. dow_sin,dow_cos        : 星期周期性编码（将 周一~周日映射至单位圆坐标，消除跨周数值断层）
6. delta_last_in_log      : 账户本笔出款距离上一次出款的秒数，取对数，看连续分拆转账频率
7. delta_last_out_log     : 账户本笔出款距离上一次进款的秒数，取对数，看过桥转走的时间



----训练训练集的标签采用交易时间在70%-85%之间的标签，验证集标签用。训练集的节点特征只在训练集里统计，验证集的节点特征取自全部交易数据，为简化训练模型，这里验证集并未采取严格按照时序滚动回测（每天往前推，每天重新算特征，每天验证一次),而是基于时间戳的固定窗口切分（即后20%交易量打包为验证窗口，用于）。

六、局限

(1)、当前是静态基线，工业级要用滚动时间窗口动态构图。
(2)、后期需要再建一个边分类的模型，作为另一个任务。
