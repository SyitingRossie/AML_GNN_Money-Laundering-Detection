# 一、原基线代码来源
该文件夹为开源基线模型源码，用于反洗钱GNN项目迭代的基础版本。

# 二、基本信息
1.初次获取渠道：Kaggle Notebook  
2.Kaggle链接：https://www.kaggle.com/code/issacchanjj/anti-money-laundering-detection-with-gnn/notebook

3.原作GitHub仓库链接：https://github.com/issacchan26/AntiMoneyLaunderingDetectionWithGNN/tree/main
4.项目结构：dataset.py、model.py、train.py
5.使用IBM数据集：HI-Small_Trans.csv  
6.原始数据集字段：【'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1','Amount Received', 'Receiving Currency', 'Amount Paid','Payment Currency', 'Payment Format', 'Is Laundering'】  其中'Is Laundering'是标签，1=洗钱客户，0=正常客户

# 三、原基线代码存在的核心缺陷
1.数据泄露问题：原作先将所有账户特征一并整理后，再进行随机切分，切90%训练集和10%验证集，因此账户节点特征已经包含了10%的测试集数据特征，例如特征列有每个账户的不同货币进款（或出款）的平均金额。如果训练集就统计了测试集在内的平均收付款金额，容易出现验证集指标虚高，模型在真实线下业务场景推理时效果大幅滑坡，泛化能力差。
2.

# 四、本人优化方案
1.修复数据泄露：按交易数据的时间排序，然后先切分前80%作为训练集，后20%为验证集。训练集的账户节点特征只在训练集里统计，验证集的节点取自验证集，验证集的节点特征取自全部交易数据，这里验证集没有严格按照时序滚动回测（每天往前推，每天重新算特征，每天验证一次),而是基于时间戳的固定窗口切分（即后20%交易量打包一起，不按天再细分）
2.
