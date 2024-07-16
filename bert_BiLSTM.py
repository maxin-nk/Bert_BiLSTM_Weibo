import torch
import pandas as pd
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertModel
from torch import nn
from tqdm import tqdm  # 终端显示进度条
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from torch.nn.functional import one_hot
from sklearn.metrics import accuracy_score


# ----------------------------------------------------------------------------------------------------------------------Step1.自定义数据集
class MydataSet(Dataset):
    def __init__(self, path, split):
        """
        初始化数据类
        :param path: 数据路径
        :param split: 分割文档
        """
        print("MydataSet.__init__()......")
        self.dataset = load_dataset("csv", data_files=path, split=split, )

    def __getitem__(self, item):
        # Column:需根据数据集调整
        print("MydataSet.__getitem__()......")
        text = self.dataset[item]['Column1']
        label = self.dataset[item]['Column2']
        return text, label

    def __len__(self):
        print("MydataSet.__len__()......")
        return len(self.dataset)


# ----------------------------------------------------------------------------------------------------------------------Step1.切分数据集
def My_train_Test(split, path):
    """
    划分训练集\验证集\测试集
    :param path: 数据集路径
    :return: 训练集，测试集
    """
    print("Mytrain_test()......")
    split = 0.6

    data = pd.read_csv(path, encoding='utf-8', sep=",", header=0, index_col=None)
    line = int(data.shape[0]*split)
    off = int(0.2*data.shape[0])
    data.iloc[:line, :].to_csv("./file/comment_for_bilstm_train.csv", encoding='utf-8', index=False)
    data.iloc[line:line+off, :].to_csv("./file/comment_for_bilstm_val.csv", encoding='utf-8', index=False)
    data.iloc[line+off:, :].to_csv("./file/comment_for_bilstm_test.csv", encoding='utf-8', index=False)

    # # 测试数据集构建
    # data.iloc[:600, :].to_csv("./file/comment_for_bilstm_train.csv", encoding='utf-8', index=False)
    # data.iloc[600:1000, :].to_csv("./file/comment_for_bilstm_val.csv", encoding='utf-8', index=False)
    # data.iloc[1000:2000, :].to_csv("./file/comment_for_bilstm_test.csv", encoding='utf-8', index=False)


# ----------------------------------------------------------------------------------------------------------------------Step2.定义批处理函数
def collate_fn(data):
    print("collate_fn()......")

    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 分词并编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 单个句子参与编码
        truncation=True,  # 当句子长度大于max_length时,截断
        padding='max_length',  # 一律补pad到max_length长度
        max_length=500,
        return_tensors='pt',  # 以pytorch的形式返回，可取值tf,pt,np,默认为返回list
        return_length=True,
    )

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']  # input_ids 就是编码后的词
    attention_mask = data['attention_mask']  # pad的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']  # (如果是一对句子)第一个句子和特殊符号的位置是0,第二个句子的位置是1
    labels = torch.LongTensor(labels)  # 该批次的labels

    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels


# ----------------------------------------------------------------------------------------------------------------------Step3.定义Bert-BiLSTM，上游Bert，下游BiLSTM,最后再加一个全连接层
class BiLSTM(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        print("BiLSTM.__init__()......")
        super(BiLSTM, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 加载bert中文模型,生成embedding层
        self.embedding = BertModel.from_pretrained('./pretrained_bert_chinese/')

        # 冻结上游模型参数(不进行预训练模型参数学习)
        for param in self.embedding.parameters():
            param.requires_grad_(False)

        # 生成下游RNN层以及全连接层
        # input_size:输入x中预期特征的个数
        # hidden_size:处于隐藏状态h的特征数(隐藏层维数)
        # num_layers:循环层数。例如，设置num_layers=2意味着将两个LSTM堆叠在一起形成一个堆叠的LSTM，第二个LSTM接收第一个LSTM的输出并计算最终结果。
        # batch_first:如果为True，则输入和输出张量以(batch, seq, feature)形式提供。默认为False（将LSTM批量数据和dataloader批量数据统一）
        # dropout:如果非零，则在除最后一层以外的每一层LSTM输出上引入一个Dropout层，Dropout概率等于Dropout。默认值:0
        # bidirectional:如果为True，则成为双向LSTM。默认值:假
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=self.drop)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)

        # 使用CrossEntropyLoss作为损失函数时，不需要激活。因为实际上CrossEntropyLoss将softmax-log-NLLLoss一并实现的。但是使用其他损失函数时还是需要加入softmax层的。

    # def forward(self, input_ids, attention_mask, token_type_ids):
    #     embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #     embedded = embedded.last_hidden_state  # 第0维才是我们需要的embedding,embedding.last_hidden_state = embedding[0]
    #     out, (h_n, c_n) = self.lstm(embedded)
    #     print(out.shape)  # [128, 200 ,1536]  因为是双向的，所以out是两个hidden_dim拼接而成。768*2 = 1536
    #     # h_n[-2, :, :] 为正向lstm最后一个隐藏状态。
    #     # h_n[-1, :, :] 为反向lstm最后一个隐藏状态
    #     print(out[:, -1, :768] == h_n[-2, :, :])  # 正向lstm最后时刻的输出在output最后一层
    #     print(out[:, 0, 768:] == h_n[-1, :, :])  # 反向lstm最后时刻的输出在output第一层

    def forward(self, input_ids, attention_mask, token_type_ids):
        print("BiLSTM.forward()......")

        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedded = embedded.last_hidden_state  # 第0维才是我们需要的embedding,embedding.last_hidden_state = embedding[0]
        # 3维张量
        # out:(batch_size,序列-句子长度,单词向量维数)->(8, 500, 768*2)
        # (h_n, c_n):((4, 8, 768), (4, 8, 768))
        out, (h_n, c_n) = self.lstm(embedded)

        # print(out.shape)  # [128, 200 ,1536]  因为是双向的，所以out是两个hidden_dim拼接而成。768*2 = 1536

        # num_layers变，相应的切片下标也要跟着变
        # h_n[-2, :, :] 为正向lstm最后一个隐藏状态。
        # h_n[-1, :, :] 为反向lstm最后一个隐藏状态
        # print(out[:, -1, :768] == h_n[-2, :, :])  # 正向lstm最后时刻的输出在output最后一层
        # print(out[:, 0, 768:] == h_n[-1, :, :])  # 反向lstm最后时刻的输出在output第一层
        # print(out[:, -1, :].shape)  # [128, 1536]  128句话，每句话的最后一个状态
        output = torch.cat((h_n[-3, :, :], h_n[-1, :, :]), dim=1)

        # print(output.shape)
        # # 检查是否拼接成功
        # print(out[:, -1, :768] == output[:, :768])
        # print(out[:, 0, 768:] == output[:, 768:])

        output = self.fc(output)
        return output


if __name__ == '__main__':

    # ----加载数据集
    split = 0.7
    # My_train_Test(split, 'D:/Comment_for_bilstm.csv')
    # train_dataset = MydataSet('D:/Comment_for_bilstm.csv', "train")
    train_dataset = MydataSet('D:/Comment_for_bilstm_train.csv', "train")
    test_dataset = MydataSet('D:/Comment_for_bilstm_test.csv', "train")
    # print(train_dataset.__len__())
    # print(train_dataset[5])

    # ----数据预处理+训练
    batch_size = 8
    epochs = 5
    dropout = 0.4
    rnn_hidden_size = 768
    rnn_layers = 1
    class_num = 3
    lr = 1e-4

    # 数据预处理
    token = BertTokenizer.from_pretrained('./pretrained_bert_chinese/')  # 加载字典和分词工具（预训练的bert模型）
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)  # 装载训练集
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)  # 装载训练集

    # 模型训练
    model = BiLSTM(dropout, rnn_hidden_size, class_num)

    # 选择损失和优化
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        t = -1
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):  # 检查一个批次是否编码成功，相关变量参见collate_fn解释

            t = i+8
            # print(len(train_loader))
            # print(input_ids[0])  # 第一句话分词后在bert-base-chinese字典中的word_to_id
            # print(token.decode(input_ids[0]))  # （解码）检查第一句话的id_to_word
            # print(input_ids.shape)  # 一个批次8句话，每句话被word_to_id成200维
            # print(attention_mask.shape)  # 对于使用者而言，不是重要的。含义上面有说明，感兴趣可以做实验测试
            # print(token_type_ids.shape)  # 对于使用者而言，不是重要的。含义上面有说明，感兴趣可以做实验测试
            # print(labels)  # 该批次的labels
            # break

            one_hot_labels = one_hot(labels-1, num_classes=class_num).to(dtype=torch.float)

            optimizer.zero_grad()  # 梯度置零

            out = model.forward(input_ids, attention_mask, token_type_ids)
            # print(out.shape)

            ll = loss(out, one_hot_labels)

            # TODO 预测+输出精度
            # 输出损失
            # print('\n--epoch', epoch+1, '\t--loss:', sum_loss / (len(train_comments) / 8), '\t--train_acc:', train_acc, '\t--test_acc', test_acc)
            print('\n--epoch:', epoch+1, '\t--data_line:[', t+1, ',', t+1+8, ']\t--loss:', ll)

            ll.backward()
            optimizer.step()

