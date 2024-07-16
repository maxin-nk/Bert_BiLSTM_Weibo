import torch
from datasets import load_dataset  # hugging-face dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.nn.functional import one_hot
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics.functional import accuracy, recall, precision, f1_score  # lightning中的评估
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# 存在下述报错时添加：RuntimeError: could not create a primitive descriptor for an LSTM forward propagation primitive
import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX512"
os.environ["MKLDNN"] = "0"
os.environ["MKLDNN_VERBOSE"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"


# 自定义数据集
class MydataSet(Dataset):
    def __init__(self, path, split):
        print("MydataSet.__init__()......")
        self.dataset = load_dataset('csv', data_files=path, split=split)

    def __getitem__(self, item):
        print("MydataSet.__getitem__()......")
        text = self.dataset[item]['Column1']
        label = self.dataset[item]['Column2']
        return text, label

    def __len__(self):
        print("MydataSet.__len__()......")
        return len(self.dataset)


# 定义批处理函数
def collate_fn(data):
    print("Mcollate()......")
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 分词并编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 单个句子参与编码
        truncation=True,  # 当句子长度大于max_length时,截断
        padding='max_length',  # 一律补pad到max_length长度
        max_length=200,
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


# 定义模型，上游使用bert预训练，下游任务选择双向LSTM模型，最后加一个全连接层
class BiLSTMClassifier(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        print("BiLSTMClassifier.__init__()......")
        super(BiLSTMClassifier, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 加载bert中文模型,生成embedding层
        self.embedding = BertModel.from_pretrained('./pretrained_bert_chinese/')

        # 冻结上游模型参数(不进行预训练模型参数学习)
        for param in self.embedding.parameters():
            param.requires_grad_(False)

        # 生成下游RNN层以及全连接层
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=self.drop)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        # 使用CrossEntropyLoss作为损失函数时，不需要激活。因为实际上CrossEntropyLoss将softmax-log-NLLLoss一并实现的。

    def forward(self, input_ids, attention_mask, token_type_ids):
        print("BiLSTMClassifier.forward()......")
        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedded = embedded.last_hidden_state  # 第0维才是我们需要的embedding,embedding.last_hidden_state = embedding[0]
        out, (h_n, c_n) = self.lstm(embedded)
        output = torch.cat((h_n[-3, :, :], h_n[-1, :, :]), dim=1)
        output = self.fc(output)
        return output


# 定义pytorch lightning
class BiLSTMLighting(pl.LightningModule):
    def __init__(self, drop, hidden_dim, output_dim):
        print("BiLSTMLightning.__init__()......")
        super(BiLSTMLighting, self).__init__()
        self.model = BiLSTMClassifier(drop, hidden_dim, output_dim)  # 设置model
        self.criterion = nn.CrossEntropyLoss()  # 设置损失函数
        self.train_dataset = MydataSet('D:/Comment_for_bilstm_train.csv', 'train')
        self.val_dataset = MydataSet('D:/Comment_for_bilstm_val.csv', 'train')
        self.test_dataset = MydataSet('D:/Comment_for_bilstm_test.csv', 'train')

        print("数据集读取完成")

    def configure_optimizers(self):
        print("BiLSTMLightning.configure_optimizers()......")
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    def forward(self, input_ids, attention_mask, token_type_ids):
        print("BiLSTMLightning.forward()......")  # forward(self,x)
        return self.model(input_ids, attention_mask, token_type_ids)

    # 训练步骤
    def training_step(self, batch, batch_idx):
        print("BiLSTMLightning.training_step()......")
        input_ids, attention_mask, token_type_ids, labels = batch  # x, y = batch
        y = one_hot(labels - 1, num_classes=3)
        # 将one_hot_labels类型转换成float
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()  # 将[128, 1, 3]挤压为[128,3]
        loss = self.criterion(y_hat, y)  # criterion(input, target)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  # 将loss输出在控制台
        return loss

    # 装载：训练集
    def train_dataloader(self):
        print("BiLSTMLightning.train_dataloader()......")
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        return train_loader

    # 装载：验证集
    def val_dataloader(self):
        print("BiLSTMLightning.val_dataloader()......")
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return val_loader

    # 验证步骤
    def validation_step(self, batch, batch_idx):
        print("BiLSTMLightning.validation_step()......")
        input_ids, attention_mask, token_type_ids, labels = batch
        y = one_hot(labels - 1, num_classes=3)
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    # 装载：测试集
    def test_dataloader(self):
        print("BiLSTMLightning.test_dataloader()......")
        test_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return test_loader

    # 测试步骤
    def test_step(self, batch, batch_idx):
        print("BiLSTMLightning.test_step()......")
        input_ids, attention_mask, token_type_ids, labels = batch

        target = labels - 1  # 用于待会儿计算acc和f1-score
        y = one_hot(target, num_classes=3)
        y = y.to(dtype=torch.float)

        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()
        pred = torch.argmax(y_hat, dim=1)
        acc = (pred == target).float().mean()

        loss = self.criterion(y_hat, y)
        self.log('loss', loss)
        # task: Literal["binary", "multiclass", "multilabel"],对应[二分类，多分类，多标签]
        #  average=None分别输出各个类别, 不加默认算平均
        re = recall(pred, target, task="multiclass", num_classes=class_num, average=None)
        pre = precision(pred, target, task="multiclass", num_classes=class_num, average=None)
        f1 = f1_score(pred, target, task="multiclass", num_classes=class_num, average=None)

        def log_score(name, scores):
            for i, score_class in enumerate(scores):
                self.log(f"{name}_class{i}", score_class)

        log_score("recall", re)
        log_score("precision", pre)
        log_score("f1", f1)
        self.log('acc', accuracy(pred, target, task="multiclass", num_classes=class_num))
        self.log('avg_recall', recall(pred, target, task="multiclass", num_classes=class_num, average="weighted"))
        self.log('avg_precision', precision(pred, target, task="multiclass", num_classes=class_num, average="weighted"))
        self.log('avg_f1', f1_score(pred, target, task="multiclass", num_classes=class_num, average="weighted"))


def train():
    print("train()......")
    # # 增加过拟合回调函数,提前停止,经过测试发现不太好用，因为可能会停止在局部最优值
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',  # 监控对象为'val_loss'
    #     patience=4,  # 耐心观察4个epoch
    #     min_delta=0.0,  # 默认为0.0，指模型性能最小变化量
    #     verbose=True,  # 在输出中显示一些关于early stopping的信息，如为何停止等
    # )

    # 增加回调最优模型，这个比较好用
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 监控对象为'val_loss'
        dirpath='checkpoints/',  # 保存模型的路径
        filename='model-{epoch:02d}-{val_loss:.2f}',  # 最优模型的名称
        save_top_k=1,  # 只保存最好的那个
        mode='min'  # 当监控对象指标最小时
    )

    # Trainer可以帮助调试，比如快速运行、只使用一小部分数据进行测试、完整性检查等，
    # max_epochs:一旦达到这个次数，就停止训练。
    # log_every_n_step: epochs内log的频率
    # accelerator:支持传递不同的加速类型(“cpu”，“gpu”，“tpu”，“hpu”，“mps”，“auto”)以及自定义加速实例。
    # devices:要使用的设备。可以设置为正数(int或str)，设备索引序列(list或str)，
    # 详情请见官方文档https://lightning.ai/docs/pytorch/latest/debug/debugging_basic.html值“-1”表示应该使用所有可用的设备，或“auto”表示根据所选加速器进行自动选择。
    # fast_device_run:如果设置为“n”(int)则运行n，如果设置为“True”则运行1，批处理train, val和test以查找任何错误(即:一种单元测试)。默认值:' '假' '。
    # precision:双精度(64，'64'或'64-true')，全精度(32，'32'或'32-true')， 16位混合精度(16，'16'，'16-mixed')或bfloat16混合精度('bf16'， 'bf16-mixed')。支持CPU、GPU、tpu、hpu。
    # callbacks:添加回调函数或回调函数列表。
    trainer = Trainer(max_epochs=epochs, log_every_n_steps=10, accelerator='gpu', devices="auto", fast_dev_run=False, precision=16, callbacks=[checkpoint_callback])
    model = BiLSTMLighting(drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    print("trainer.fit(model)......")
    trainer.fit(model)


def test(model_path):
    print("test()......")
    # 加载之前训练好的最优模型参数
    model = BiLSTMLighting.load_from_checkpoint(checkpoint_path=model_path, drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    trainer = Trainer(fast_dev_run=False)
    result = trainer.test(model)
    print(result)


if __name__ == '__main__':

    batch_size = 128
    epochs = 30
    dropout = 0.5
    rnn_hidden = 768
    rnn_layer = 1
    class_num = 3
    lr = 0.001
    model_path = './lightning_logs/checkpoints/version1_model.ckpt'

    token = BertTokenizer.from_pretrained('./pretrained_bert_chinese/')

    train()
    test(model_path)
