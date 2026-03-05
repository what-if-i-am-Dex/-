【关于Credit Scoring数据集的说明】

这个数据集代表云际联盟实际运行时收集到的数据，其目标是训练模型来预测 ServiceCompletionStatus，即本次交易能否成功完成。

数据集中，每一行代表一笔云际服务交易记录，字段如下：
-TransactionAmount
表示本次云际服务中请求的资源总量，例如所需算力、存储或带宽等，原对应金融数据中的贷款金额。
-RemainingDebt
表示当前这笔服务中尚未交付的资源量，也可以理解为服务未履约部分的度量。
-TotalPaymentReceived
实际收到或完成的服务资源量，表示该节点已经完成交付的部分。
-PenaltyRecoveries
因服务延迟、失约等原因被追偿的资源量，用于衡量违约后的追回能力或赔偿情况。
-RecoveryCost
联盟为了追回违约资源而付出的额外成本，反映节点违约所带来的负面影响程度。
-LastTransactionValue
上一次服务交易的资源量或价值，可用于分析节点服务行为的连续性和稳定性。
-LastReputationScore
上一次交易前，该节点的信誉评分（例如 0~100 分），由系统基于历史行为计算得出。这个字段是一个连续型特征，代表服务方在系统中的历史信任水平。
-LateServicePenalty
因服务交付延迟或未达SLA标准所产生的罚款值，表示其时效性表现。
-ServiceCompletionStatus（label）
【标签字段】表示本次服务是否顺利完成。
	-Fully Paid：服务完成，资源交付符合预期
	-Charged Off：服务失败，可能因中断、违约、性能不达标等原因被终止
	-Current：收集数据时本次交易还在进行中


【关于 ServiceCompletionStatus 与 LastReputationScore 的说明】
	这个数据集中的目标是训练模型来预测 ServiceCompletionStatus，即服务是否成功完成。因此，它被设置为主要的监督学习标签。
	而 LastReputationScore 是一种反映服务方历史行为的数值型特征，它由系统在上一笔交易之前计算得出。它虽然是“信誉分数”，但它反映的是模型输入时已有的信息，并非这次交易的结果。
	也就是说，我们并不是直接去预测信誉分数本身，而是利用包括“上一笔信誉分”在内的各种特征，来判断本次服务是否会完成，最终可以通过模型反复迭代后构建新的信誉评分机制