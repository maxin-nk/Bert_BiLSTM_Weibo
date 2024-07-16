import csv
import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
# from d2l import torch as d2l
from tqdm import tqdm  # 终端显示进度条
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# ----------------------------------------------------------------------------------------------------------------------读取评论文件的评论信息
def read_file(file_name):
    comments_data = None

    # 读取评论信息
    with open(file_name, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        # 读取评论数据和对应的标签信息
        comments_data = [[line[0], int(line[1])] for line in reader if len(line[0]) > 0]

    # 打乱数据集
    random.shuffle(comments_data)
    data = pd.DataFrame(comments_data)
    same_sentence_num = data.duplicated().sum()  # 统计重复的评论内容个数

    if same_sentence_num > 0:
        data = data.drop_duplicates()  # 删除重复的样本信息

    f.close()

    return data


# ----------------------------------------------------------------------------------------------------------------------Step2-1: 生成层-预训练模块-定义BERTClassifier分类器模型（直接分类）
class BERTClassifier(nn.Module):

    def __init__(self, output_dim, pretrained_name='pretrained_bert_chinese'):
        """
        初始化加载 pretrained_bert_chinese 原型，即Bert中的Bert-Base-chinese模型
        :param output_dim: 分类的标签维度
        :param pretrained_name: 预训练模型名称
        """
        super(BERTClassifier, self).__init__()

        # 定义 Bert 模型
        self.bert = BertModel.from_pretrained('./pretrained_bert_chinese/')

        # 外接全连接层
        self.mlp = nn.Linear(768, output_dim)

        # 冻结模型参数（减少训练时间和计算资源消耗/防止过拟合/模型压缩和加速）
        for param in self.bert.parameters():
            param.requires_grad_(False)

    def forward(self, tokens_X):
        # 得到最后一层的 '<cls>' 信息， 其标志全部上下文信息
        res = self.bert(**tokens_X)

        # res[1]代表序列的上下文信息'<cls>'，外接全连接层，进行情感分析
        return self.mlp(res[1])


# ----------------------------------------------------------------------------------------------------------------------评估函数
def evaluate(net, comments_data, labels_data):
    """
    数据太多，内存不足，无法直接使用sklearn.metrics.accuracy_score
    :param net:
    :param comments_data:
    :param labels_data:
    :return:
    """
    print("开始执行evaluate(net, comments_data, labels_data)......")
    sum_correct, i = 0, 0

    while i <= len(comments_data):
        print(i+8)
        comments = comments_data[i: min(i + 8, len(comments_data))]

        tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt')

        res = net(tokens_X)  # 获得到预测结果

        y = torch.tensor(labels_data[i: min(i + 8, len(comments_data))]).reshape(-1)

        sum_correct += (res.argmax(axis=1) == y).sum()  # 累加预测正确的结果
        i += 8

    print("evaluate()执行完毕......")
    return sum_correct / len(comments_data)  # 返回(总正确结果/所有样本)，精确率


# d2l.train_ch13()


# ----------------------------------------------------------------------------------------------------------------------训练Bert-BiLSTM分类器
def train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels, epochs):
    max_acc = 0.5  # 初始化模型最大精度为0.5

    # 先测试未训练前的模型精确度
    train_acc = evaluate(net, train_comments, train_labels)
    test_acc = evaluate(net, test_comments, test_labels)

    # 输出精度
    print('--epoch', 0, '\t--train_acc:', train_acc, '\t--test_acc', test_acc)

    # 累计训练18万条数据 epochs 次，优化模型
    for epoch in tqdm(range(epochs)):

        # 每次开始训练时， i 为 0 表示从第一条数据开始训练
        i, sum_loss = 0, 0

        # 开始训练模型
        while i < len(train_comments):
            print("第%s次epoch，第[%d, %d]条数据训练" % (epoch, i, i + 8))

            # 批量训练，每次训练8条样本数据
            comments = train_comments[i: min(i + 8, len(train_comments))]

            # 通过 tokenizer 数据化输入的评论语句信息，准备输入bert分类器
            tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt')

            # 将数据输入到bert分类器模型中，获得结果
            res = net(tokens_X)

            # 批量获取实际结果信息
            y = torch.tensor(train_labels[i: min(i + 8, len(train_comments))]).reshape(-1)

            optimizer.zero_grad()  # 清空梯度
            l = loss(res, y)  # 计算损失
            l.backward()  # 后向传播
            optimizer.step()  # 更新梯度

            sum_loss += l.detach()  # 累加损失
            i += 8  # 样本下标累加

            break

        # 计算训练集与测试集的精度
        train_acc = evaluate(net, train_comments, train_labels)
        test_acc = evaluate(net, test_comments, test_labels)

        # 输出精度
        print('\n--epoch', epoch+1, '\t--loss:', sum_loss / (len(train_comments) / 8), '\t--train_acc:', train_acc,
              '\t--test_acc', test_acc)

        torch.save(net.state_dict(), 'bert.parameters')

        # 如果测试集精度 大于 之前保存的最大精度，保存模型参数，并重设最大值
        if test_acc > max_acc:
            # 更新历史最大精确度
            max_acc = test_acc

            # 保存模型
            torch.save(net.state_dict(), 'bert.parameters')


if __name__ == "__main__":

    # 划分训练集与测试集，并将pandas数据类型转化为列表类型
    # comments_data = read_file('./file/comments.csv')
    comments_data = read_file('./file/comments.csv')[:250]

    # split = 0.7
    # split_line = int(len(comments_data) * split)
    # train_comments, train_labels = list(comments_data[: split_line][0]), list(comments_data[: split_line][1])
    # test_comments, test_labels = list(comments_data[split_line:][0]), list(comments_data[split_line:][1])
    train_comments, test_comments, train_labels, test_labels = [list(i) for i in train_test_split(comments_data.iloc[:, 0], comments_data.iloc[:, 1], test_size=0.3, random_state=42)]
    print("数据集划分完成！")

    # ------------------------------------------------------------------------------------------------------------------(直接利用Bert分类)BERTClassifier分类器，因为最终结果为3分类，所以输出维度为3，代表概率分布
    net = BERTClassifier(output_dim=3)
    print("bert类实例化完成！")
    # net = net.to(device)
    # 定义tokenizer对象，用于将评论语句转化为BertModel的输入信息
    tokenizer = BertTokenizer.from_pretrained('./pretrained_bert_chinese/')
    print("tokenizer化完成！")

    # 损失函数+Adam优化
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels, 5)
