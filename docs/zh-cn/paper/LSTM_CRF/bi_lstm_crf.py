# Author: Robert Guthrie

# 导入必要的模块
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


# -------------辅助函数的功能是使代码更具可读性。--------------------
def argmax(vec):
    # 将argmax作为python int返回
    # 返回vec的dim为1维度的最大索引值
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    # 将句子转化为ID
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# 以正向算法的数值稳定方式计算log sum exp
# 前向算法是不断累积之前的结果，这样就会有个缺点
# 指数和累积到一定的程度后，会超过计算机浮点值得最大值，变为inf,这样取log后也是inf
# 为了避免这种情况，用一个合适的值clip去提指数和的公因子，这样就不会使其某项变得过大而无法计算
# SUM = log(exp(s1)+exp(s2)+...+exp(s100))
#     = log{(clip)*[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]}
#     = clip + log[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]
# where clip=max
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))



# ------------创建模型-------------------------
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # word embedding dim
        self.hidden_dim = hidden_dim        # Bi-LSTM hidden dim
        self.vocab_size = vocab_size        # 字典的大小
        self.tag_to_ix = tag_to_ix          # NER类别转成ID
        self.tagset_size = len(tag_to_ix)   # NER类别的个数

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)  # 维度是 vocab_size*embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,  # 单向的凭借在一起，所以hidden_dim//2
                            num_layers=1, bidirectional=True)  # 1层 bilstm

        # 将LSTM的输出映射到标记空间。
        # 将BILSTM提取的特征向量映射到特征空间，即经过全连接得到 “发射分数”
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 转换参数矩阵。 输入i，j是得分从j转换到i。
        # 转移矩阵的参数初始化，transitions[i,j]表示从第j个tag转移到第i个tag的 “转移分数”
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))   # 维度是tag_size*tag_size的方阵

        # 这两个语句强制执行我们从不转移到开始标记的约束
        # 并且我们永远不会从停止标记转移
        # 初始化所有其他tag转移到START_TAG的分数非常小，即不可能由其他tag转移到START_TAG
        # 初始化STOP_TAG转移到所有其他tag的分数非常小，即不可能由STOP_TAG转移到其他tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 初始化LSTM的参数
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        # 通过Bi-LSTM提权特征
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)  # 转化成embedding的序列
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)  # 得到 “发射分数”
        return lstm_feats

    def _score_sentence(self, feats, tags):  # feats: 发射分数， tags：路径的tag
        # Gives the score of a provided tag sequence
        # 计算给定tag序列的分数，即1条路径的分数
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # 递推计算路径分数： 转移分数 + 发射分数
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]  # 考虑最后一个tag转移到STOP_TAG的分数
        return score

    def _forward_alg(self, feats):
        # 使用前向算法来计算分区函数
        # 通过前向算法递推计算 所有可能路径的分数总和
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG包含所有得分.
        # 初始化step 0即START位置的发射分数，START_TAG取0其他取-10000
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 包装一个变量，以便我们获得自动反向提升
        # 将初始化的STRAT位置为0的发射分数赋值给previous(forward_var)
        forward_var = init_alphas

        # 通过句子迭代
        # 迭代整个句子，feats就是发射分数,feat(obs)
        for feat in feats:  # 对每个时间步进行for循环
            # 当前时间步的前向tensor,存放着当前时间步的每个tag的分数
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size): # 当前时间步可以转移到各个tag，分别计算转移到各个tag的分数
                # 广播发射得分：无论以前的标记是怎样的都是相同的
                # 取出当前tag的发射分数，与之前时间步的tag无关
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # trans_score的第i个条目是从i转换到next_tag的分数
                # 取出当前tag由之前tag转移过来的转移分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # next_tag_var的第i个条目是我们执行log-sum-exp之前的边（i -> next_tag）的值
                # 当前路径的分数： 之前时间步的分数 + 转移分数 + 发射分数
                next_tag_var = forward_var + trans_score + emit_score
                # 此标记的转发变量是所有分数的log-sum-exp。
                # 对当前的分数取log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 更新previous,递推计算下一个时间步
            forward_var = torch.cat(alphas_t).view(1, -1)
        # 考虑最终转移带哦STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # 计算最终扥分
        alpha = log_sum_exp(terminal_var)
        return alpha


    def _viterbi_decode(self, feats):  # 发射分数，转移分数
        # 每个时间步存放了当前tag对应之前的路径
        backpointers = []

        # Initialize the viterbi variables in log space
        # 初始化Viterbi的previous变脸
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:  # 对时间步进行循环
            bptrs_t = []  # holds the backpointers for this step 保存当前时间步的回溯指针
            viterbivars_t = []  # holds the viterbi variables for this step 保存当前时间步的viterbi变量

            for next_tag in range(self.tagset_size):  # 对当前时间步各个tag进行循环
                # next_tag_var [i]保存上一步的标签i的维特比变量
                # 加上从标签i转换到next_tag的分数。
                # 我们这里不包括emission分数，因为最大值不依赖于它们（我们在下面添加它们）

                # Viterbi算法记录最优路径时只考虑上一步的分数以及上一步tag转移到当前tag的转移分数
                # 并不取决于当前tag的发射分数
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)  #当前tag是有哪一个previous tag转移过来的
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))  # 存的路径
            # 现在添加emission分数，并将forward_var分配给我们刚刚计算的维特比变量集
            # 更新previous，加上当前tag的发射分数obs(feat)
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # 回溯指针记录当前时间步各个tag来源前一步的tag
            backpointers.append(bptrs_t)

        # 过渡到STOP_TAG
        # 转移到STOP_TAG的转移分数
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]  # 最优路径所对应的path_score

        # 按照后退指针解码最佳路径。
        # 通过回溯指针解码最优路径
        best_path = [best_tag_id]
        # best_tag_id作为线头，反向遍历backpointers找到最优路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始标记（我们不想将其返回给调用者）
        # 去除START_TAG
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()  # 将best_path 反过来，从第一步到最后一步
        return path_score, best_path  # 返回最优路径对应的分数和最优路径

    def neg_log_likelihood(self, sentence, tags):
        # CRF损失函数由两部分组成，真实路径的分数和所有路径的总分数
        # 真实路径的分数应该是所有路径中分数最高的
        # log真实路径的分数/log所有可能得路径的分数，越大越好，构造crf loss 函数取反，loss越小越好
        feats = self._get_lstm_features(sentence) # 发射分数
        forward_score = self._forward_alg(feats)  # 所有可能得路径的分数
        gold_score = self._score_sentence(feats, tags)  # 真实路径的分数
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # 模型的forward
        # 获取BiLSTM的emission分数
        lstm_feats = self._get_lstm_features(sentence)

        # 根据功能，找到最佳路径。
        # 根据发射分数以及转移分数，通过viterbi解码找到一条最优路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


#  ------------------------进行训练------------------------------
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# 构造一些训练数据
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 在训练前检查模型的预测效果
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# return: 
# (tensor(2.6907), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1])


# 模型的训练
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # 步骤1. 请记住，Pytorch积累了梯度，需要清零梯度
        # We need to clear them out before each instance
        model.zero_grad()

        # 步骤2. 为网络准备输入，即将它们转换为单词索引的张量.
        # 将输入转化为tensors
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # 步骤3. 进行前向计算，取出crf loss
        loss = model.neg_log_likelihood(sentence_in, targets)

        # 步骤4.通过调用optimizer.step（）来计算损失，梯度和更新参数
        loss.backward()
        optimizer.step()

# 训练后检查预测
# 训练结束看模型的预测结果，对比观察模型是否学到
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# 得到结果

# return： 
# (tensor(20.4906), [0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])