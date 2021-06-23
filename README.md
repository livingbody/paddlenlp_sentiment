# 〇、『NLP直播课』Day 5：情感分析预训练模型SKEP


aistudio地址：[https://aistudio.baidu.com/aistudio/projectdetail/2082926](https://aistudio.baidu.com/aistudio/projectdetail/2082926)

本项目将详细全面介绍情感分析任务的两种子任务，句子级情感分析和目标级情感分析。

同时演示如何使用情感分析预训练模型SKEP完成以上两种任务，详细介绍预训练模型SKEP及其在 PaddleNLP 的使用方式。

本项目主要包括“任务介绍”、“情感分析预训练模型SKEP”、“句子级情感分析”、“目标级情感分析”等四个部分。





```python
!pip install --upgrade paddlenlp -i https://pypi.org/simple 
```

##  1.Part A. 情感分析任务

众所周知，人类自然语言中包含了丰富的情感色彩：表达人的情绪（如悲伤、快乐）、表达人的心情（如倦怠、忧郁）、表达人的喜好（如喜欢、讨厌）、表达人的个性特征和表达人的立场等等。情感分析在商品喜好、消费决策、舆情分析等场景中均有应用。利用机器自动分析这些情感倾向，不但有助于帮助企业了解消费者对其产品的感受，为产品改进提供依据；同时还有助于企业分析商业伙伴们的态度，以便更好地进行商业决策。

被人们所熟知的情感分析任务是将一段文本分类，如分为情感极性为**正向**、**负向**、**其他**的三分类问题：
<br></br>
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/b630901b397e4e7a8e78ab1d306dfa1fc070d91015a64ef0b8d590aaa8cfde14" width="600" ></center>
<br><center>情感分析任务</center></br>

- **正向：** 表示正面积极的情感，如高兴，幸福，惊喜，期待等。
- **负向：** 表示负面消极的情感，如难过，伤心，愤怒，惊恐等。
- **其他：** 其他类型的情感。

实际上，以上熟悉的情感分析任务是**句子级情感分析任务**。


情感分析任务还可以进一步分为**句子级情感分析**、**目标级情感分析**等任务。在下面章节将会详细介绍两种任务及其应用场景。


## 2.Part B. 情感分析预训练模型SKEP

近年来，大量的研究表明基于大型语料库的预训练模型（Pretrained Models, PTM）可以学习通用的语言表示，有利于下游NLP任务，同时能够避免从零开始训练模型。随着计算能力的发展，深度模型的出现（即 Transformer）和训练技巧的增强使得 PTM 不断发展，由浅变深。

情感预训练模型SKEP（Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis）。SKEP利用情感知识增强预训练模型， 在14项中英情感分析典型任务上全面超越SOTA，此工作已经被ACL 2020录用。SKEP是百度研究团队提出的基于情感知识增强的情感预训练算法，此算法采用无监督方法自动挖掘情感知识，然后利用情感知识构建预训练目标，从而让机器学会理解情感语义。SKEP为各类情感分析任务提供统一且强大的情感语义表示。

**论文地址**：https://arxiv.org/abs/2005.05635

<p align="center">
<img src="https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep.png" width="80%" height="60%"> <br />
</p>

百度研究团队在三个典型情感分析任务，句子级情感分类（Sentence-level Sentiment Classification），评价目标级情感分类（Aspect-level Sentiment Classification）、观点抽取（Opinion Role Labeling），共计14个中英文数据上进一步验证了情感预训练模型SKEP的效果。

具体实验效果参考：https://github.com/baidu/Senta#skep




## 3.Part C 句子级情感分析 & 目标级情感分析

### 3.1Part C.1 句子级情感分析


对给定的一段文本进行情感极性分类，常用于影评分析、网络论坛舆情分析等场景。如:

```text
选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般	1
15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错	1
房间太小。其他的都一般。。。。。。。。。	0
```

其中`1`表示正向情感，`0`表示负向情感。


<br></br>
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/4aae00a800ae4831b6811b669f7461d8482344b183454d8fb7d37c83defb9567" width="550" ></center>
<br><center>句子级情感分析任务</center></br>


#### 3.1.1常用数据集

ChnSenticorp数据集是公开中文情感分析常用数据集， 其为2分类数据集。PaddleNLP已经内置该数据集，一键即可加载。



# 一、ChnSenticorp数据集【分数0.9542】


```python
from paddlenlp.datasets import load_dataset

train_ds, dev_ds, test_ds = load_dataset("chnsenticorp", splits=["train", "dev", "test"])

print(train_ds[0])
print(train_ds[1])
print(train_ds[2])
```

    {'text': '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般', 'label': 1, 'qid': ''}
    {'text': '15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错', 'label': 1, 'qid': ''}
    {'text': '房间太小。其他的都一般。。。。。。。。。', 'label': 0, 'qid': ''}



## 1. SKEP模型加载

PaddleNLP已经实现了SKEP预训练模型，可以通过一行代码实现SKEP加载。

句子级情感分析模型是SKEP fine-tune 文本分类常用模型`SkepForSequenceClassification`。其首先通过SKEP提取句子语义特征，之后将语义特征进行分类。


![](https://ai-studio-static-online.cdn.bcebos.com/fc21e1201154451a80f32e0daa5fa84386c1b12e4b3244e387ae0b177c1dc963)





```python
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

# 指定模型名称，一键加载模型
model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch", num_classes=len(train_ds.label_list))
# 同样地，通过指定模型名称一键加载对应的Tokenizer，用于处理文本数据，如切分token，转token_id等。
tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch")
```

    [2021-06-16 00:17:41,649] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    [2021-06-16 00:17:51,743] [    INFO] - Found /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.vocab.txt



```python
# 加入日志显示
from visualdl import LogWriter

writer = LogWriter("./log")
```

`SkepForSequenceClassification`可用于句子级情感分析和目标级情感分析任务。其通过预训练模型SKEP获取输入文本的表示，之后将文本表示进行分类。

* `pretrained_model_name_or_path`：模型名称。支持"skep_ernie_1.0_large_ch"，"skep_ernie_2.0_large_en"。
	- "skep_ernie_1.0_large_ch"：是SKEP模型在预训练ernie_1.0_large_ch基础之上在海量中文数据上继续预训练得到的中文预训练模型；
    - "skep_ernie_2.0_large_en"：是SKEP模型在预训练ernie_2.0_large_en基础之上在海量英文数据上继续预训练得到的英文预训练模型；
  
* `num_classes`: 数据集分类类别数。


关于SKEP模型实现详细信息参考：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/skep
    

## 2. 数据处理

同样地，我们需要将原始ChnSentiCorp数据处理成模型可以读入的数据格式。

SKEP模型对中文文本处理按照字粒度进行处理，我们可以使用PaddleNLP内置的`SkepTokenizer`完成一键式处理。


```python
import os
from functools import partial


import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad

from utils import create_dataloader

def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence has the following format:
    ::
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence pair mask has the following format:
    ::

        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).


    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`int`, optional): The input label if not is_test.
    """
    # 将原数据处理成model可读入的格式，enocded_inputs是一个dict，包含input_ids、token_type_ids等字段
    encoded_inputs = tokenizer(
        text=example["text"], max_seq_len=max_seq_length)

    # input_ids：对文本切分token后，在词汇表中对应的token id
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids：当前token属于句子1还是句子2，即上述图中表达的segment ids
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        # label：情感极性类别
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        # qid：每条数据的编号
        qid = np.array([example["qid"]], dtype="int64")
        return input_ids, token_type_ids, qid
```


```python
# 批量数据大小
batch_size = 20
# 文本序列最大长度
max_seq_length = 128

# 将数据处理成模型可读入的数据格式
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

# 将数据组成批量式数据，如
# 将不同长度的文本序列padding到批量式数据中最大长度
# 将每条数据label堆叠在一起
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack()  # labels
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

## 3.模型训练和评估


定义损失函数、优化器以及评价指标后，即可开始训练。


**推荐超参设置：**

* `max_seq_length=256`
* `batch_size=48`
* `learning_rate=2e-5`
* `epochs=10`

实际运行时可以根据显存大小调整batch_size和max_seq_length大小。




```python
import time

from utils import evaluate

# 训练轮次
epochs = 10
# 训练过程中保存模型参数的文件夹
ckpt_dir = "skep_ckpt"
# len(train_data_loader)一轮训练所需要的step数
num_training_steps = len(train_data_loader) * epochs

# Adam优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=2e-5,
    parameters=model.parameters())
# 交叉熵损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()
# accuracy评价指标
metric = paddle.metric.Accuracy()
```


```python
# 开启训练
global_step = 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        # 喂数据给model
        logits = model(input_ids, token_type_ids)
        # 计算损失函数值
        loss = criterion(logits, labels)
        # 预测分类概率值
        probs = F.softmax(logits, axis=1)
        # 计算acc
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            save_dir = os.path.join(ckpt_dir, "model_%d" % global_step)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 评估当前训练的模型
            evaluate(model, criterion, metric, dev_data_loader)
            # 保存当前模型参数等
            model.save_pretrained(save_dir)
            # 保存tokenizer的词表等
            tokenizer.save_pretrained(save_dir)
```

    global step 10, epoch: 1, batch: 10, loss: 0.66369, accu: 0.61000, speed: 2.16 step/s
    global step 20, epoch: 1, batch: 20, loss: 0.42656, accu: 0.66250, speed: 2.15 step/s
    global step 30, epoch: 1, batch: 30, loss: 0.39585, accu: 0.73667, speed: 2.13 step/s
    global step 40, epoch: 1, batch: 40, loss: 0.36039, accu: 0.76875, speed: 2.14 step/s
    global step 50, epoch: 1, batch: 50, loss: 0.43957, accu: 0.79800, speed: 2.12 step/s
    global step 60, epoch: 1, batch: 60, loss: 0.18666, accu: 0.81750, speed: 2.12 step/s
    global step 70, epoch: 1, batch: 70, loss: 0.34755, accu: 0.83357, speed: 2.13 step/s
    global step 80, epoch: 1, batch: 80, loss: 0.06662, accu: 0.84688, speed: 2.12 step/s
    global step 90, epoch: 1, batch: 90, loss: 0.21058, accu: 0.85389, speed: 2.12 step/s
    global step 100, epoch: 1, batch: 100, loss: 0.18824, accu: 0.86250, speed: 2.12 step/s
    eval loss: 0.27623, accu: 0.90083
    global step 110, epoch: 1, batch: 110, loss: 0.34183, accu: 0.91500, speed: 0.34 step/s
    global step 120, epoch: 1, batch: 120, loss: 0.37227, accu: 0.91000, speed: 2.13 step/s
    global step 130, epoch: 1, batch: 130, loss: 0.09790, accu: 0.90333, speed: 2.12 step/s
    global step 140, epoch: 1, batch: 140, loss: 0.28844, accu: 0.91250, speed: 2.11 step/s
    global step 150, epoch: 1, batch: 150, loss: 0.29506, accu: 0.90700, speed: 2.11 step/s
    global step 160, epoch: 1, batch: 160, loss: 0.08923, accu: 0.90500, speed: 2.12 step/s
    global step 170, epoch: 1, batch: 170, loss: 0.21318, accu: 0.90429, speed: 2.11 step/s
    global step 180, epoch: 1, batch: 180, loss: 0.15742, accu: 0.90563, speed: 2.11 step/s
    global step 190, epoch: 1, batch: 190, loss: 0.07610, accu: 0.90833, speed: 2.11 step/s
    global step 200, epoch: 1, batch: 200, loss: 0.30590, accu: 0.90800, speed: 2.10 step/s
    eval loss: 0.28047, accu: 0.89333
    global step 210, epoch: 1, batch: 210, loss: 0.05899, accu: 0.92500, speed: 0.34 step/s
    global step 220, epoch: 1, batch: 220, loss: 0.33334, accu: 0.91750, speed: 2.11 step/s
    global step 230, epoch: 1, batch: 230, loss: 0.05420, accu: 0.92000, speed: 2.13 step/s
    global step 240, epoch: 1, batch: 240, loss: 0.11374, accu: 0.92000, speed: 2.11 step/s
    global step 250, epoch: 1, batch: 250, loss: 0.11836, accu: 0.92000, speed: 2.12 step/s
    global step 260, epoch: 1, batch: 260, loss: 0.26100, accu: 0.91833, speed: 2.13 step/s
    global step 270, epoch: 1, batch: 270, loss: 0.11742, accu: 0.91643, speed: 2.10 step/s
    global step 280, epoch: 1, batch: 280, loss: 0.09372, accu: 0.91812, speed: 2.12 step/s
    global step 290, epoch: 1, batch: 290, loss: 0.15071, accu: 0.92000, speed: 2.11 step/s
    global step 300, epoch: 1, batch: 300, loss: 0.32532, accu: 0.91600, speed: 2.09 step/s
    eval loss: 0.22735, accu: 0.91000
    global step 310, epoch: 1, batch: 310, loss: 0.19757, accu: 0.92500, speed: 0.34 step/s
    global step 320, epoch: 1, batch: 320, loss: 0.50677, accu: 0.91250, speed: 2.10 step/s
    global step 330, epoch: 1, batch: 330, loss: 0.10222, accu: 0.91000, speed: 2.09 step/s
    global step 340, epoch: 1, batch: 340, loss: 0.04122, accu: 0.92000, speed: 2.12 step/s
    global step 350, epoch: 1, batch: 350, loss: 0.36210, accu: 0.92100, speed: 2.10 step/s
    global step 360, epoch: 1, batch: 360, loss: 0.09653, accu: 0.92000, speed: 2.12 step/s
    global step 370, epoch: 1, batch: 370, loss: 0.10296, accu: 0.92214, speed: 2.10 step/s
    global step 380, epoch: 1, batch: 380, loss: 0.15785, accu: 0.92063, speed: 2.12 step/s
    global step 390, epoch: 1, batch: 390, loss: 0.24529, accu: 0.92056, speed: 2.11 step/s
    global step 400, epoch: 1, batch: 400, loss: 0.37184, accu: 0.91750, speed: 2.12 step/s
    eval loss: 0.19365, accu: 0.93167
    global step 410, epoch: 1, batch: 410, loss: 0.27297, accu: 0.92500, speed: 0.34 step/s
    global step 420, epoch: 1, batch: 420, loss: 0.02970, accu: 0.94000, speed: 2.11 step/s
    global step 430, epoch: 1, batch: 430, loss: 0.20546, accu: 0.93500, speed: 2.10 step/s
    global step 440, epoch: 1, batch: 440, loss: 0.25177, accu: 0.93375, speed: 2.11 step/s
    global step 450, epoch: 1, batch: 450, loss: 0.10112, accu: 0.93300, speed: 2.13 step/s
    global step 460, epoch: 1, batch: 460, loss: 0.05097, accu: 0.93417, speed: 2.11 step/s
    global step 470, epoch: 1, batch: 470, loss: 0.36673, accu: 0.93786, speed: 2.11 step/s
    global step 480, epoch: 1, batch: 480, loss: 0.13309, accu: 0.93500, speed: 2.15 step/s
    global step 490, epoch: 2, batch: 10, loss: 0.05503, accu: 0.94056, speed: 2.09 step/s
    global step 500, epoch: 2, batch: 20, loss: 0.03811, accu: 0.94300, speed: 2.12 step/s
    eval loss: 0.17977, accu: 0.94083
    global step 510, epoch: 2, batch: 30, loss: 0.02016, accu: 0.96500, speed: 0.34 step/s
    global step 520, epoch: 2, batch: 40, loss: 0.08749, accu: 0.97250, speed: 2.12 step/s
    global step 530, epoch: 2, batch: 50, loss: 0.01431, accu: 0.96833, speed: 2.12 step/s
    global step 540, epoch: 2, batch: 60, loss: 0.06466, accu: 0.96500, speed: 2.12 step/s
    global step 550, epoch: 2, batch: 70, loss: 0.00825, accu: 0.96700, speed: 2.09 step/s
    global step 560, epoch: 2, batch: 80, loss: 0.27605, accu: 0.97000, speed: 2.07 step/s
    global step 570, epoch: 2, batch: 90, loss: 0.00833, accu: 0.97143, speed: 2.08 step/s
    global step 580, epoch: 2, batch: 100, loss: 0.07668, accu: 0.97500, speed: 2.08 step/s
    global step 590, epoch: 2, batch: 110, loss: 0.02110, accu: 0.97500, speed: 2.08 step/s
    global step 600, epoch: 2, batch: 120, loss: 0.01693, accu: 0.97350, speed: 2.08 step/s
    eval loss: 0.20099, accu: 0.93333
    global step 610, epoch: 2, batch: 130, loss: 0.07885, accu: 0.98000, speed: 0.34 step/s
    global step 620, epoch: 2, batch: 140, loss: 0.02133, accu: 0.97250, speed: 2.12 step/s
    global step 630, epoch: 2, batch: 150, loss: 0.11444, accu: 0.97167, speed: 2.12 step/s
    global step 640, epoch: 2, batch: 160, loss: 0.09162, accu: 0.96375, speed: 2.12 step/s
    global step 650, epoch: 2, batch: 170, loss: 0.03649, accu: 0.96000, speed: 2.11 step/s
    global step 660, epoch: 2, batch: 180, loss: 0.16949, accu: 0.96167, speed: 2.09 step/s
    global step 670, epoch: 2, batch: 190, loss: 0.07496, accu: 0.96071, speed: 2.08 step/s
    global step 680, epoch: 2, batch: 200, loss: 0.05775, accu: 0.96125, speed: 2.09 step/s
    global step 690, epoch: 2, batch: 210, loss: 0.02518, accu: 0.96389, speed: 2.08 step/s
    global step 700, epoch: 2, batch: 220, loss: 0.29069, accu: 0.96400, speed: 2.09 step/s
    eval loss: 0.19786, accu: 0.93500
    global step 710, epoch: 2, batch: 230, loss: 0.01367, accu: 0.98000, speed: 0.34 step/s
    global step 720, epoch: 2, batch: 240, loss: 0.27000, accu: 0.96250, speed: 2.11 step/s
    global step 730, epoch: 2, batch: 250, loss: 0.22897, accu: 0.96333, speed: 2.11 step/s
    global step 740, epoch: 2, batch: 260, loss: 0.35368, accu: 0.96375, speed: 2.12 step/s
    global step 750, epoch: 2, batch: 270, loss: 0.01384, accu: 0.96400, speed: 2.11 step/s
    global step 760, epoch: 2, batch: 280, loss: 0.02931, accu: 0.96583, speed: 2.11 step/s
    global step 770, epoch: 2, batch: 290, loss: 0.10045, accu: 0.96429, speed: 2.12 step/s
    global step 780, epoch: 2, batch: 300, loss: 0.23759, accu: 0.96250, speed: 2.11 step/s
    global step 790, epoch: 2, batch: 310, loss: 0.16488, accu: 0.96222, speed: 2.10 step/s
    global step 800, epoch: 2, batch: 320, loss: 0.04031, accu: 0.96350, speed: 2.10 step/s
    eval loss: 0.18547, accu: 0.93583
    global step 810, epoch: 2, batch: 330, loss: 0.01714, accu: 0.98500, speed: 0.34 step/s
    global step 820, epoch: 2, batch: 340, loss: 0.12877, accu: 0.97250, speed: 2.11 step/s
    global step 830, epoch: 2, batch: 350, loss: 0.09537, accu: 0.96333, speed: 2.13 step/s
    global step 840, epoch: 2, batch: 360, loss: 0.02554, accu: 0.96375, speed: 2.11 step/s
    global step 850, epoch: 2, batch: 370, loss: 0.02596, accu: 0.96700, speed: 2.11 step/s
    global step 860, epoch: 2, batch: 380, loss: 0.06234, accu: 0.96667, speed: 2.12 step/s
    global step 870, epoch: 2, batch: 390, loss: 0.03390, accu: 0.96286, speed: 2.12 step/s
    global step 880, epoch: 2, batch: 400, loss: 0.13883, accu: 0.96188, speed: 2.11 step/s
    global step 890, epoch: 2, batch: 410, loss: 0.03865, accu: 0.96389, speed: 2.12 step/s
    global step 900, epoch: 2, batch: 420, loss: 0.07129, accu: 0.96500, speed: 2.12 step/s
    eval loss: 0.18045, accu: 0.94500
    global step 910, epoch: 2, batch: 430, loss: 0.06873, accu: 0.96500, speed: 0.34 step/s
    global step 920, epoch: 2, batch: 440, loss: 0.42489, accu: 0.94750, speed: 2.09 step/s
    global step 930, epoch: 2, batch: 450, loss: 0.07079, accu: 0.94667, speed: 2.09 step/s
    global step 940, epoch: 2, batch: 460, loss: 0.20789, accu: 0.95125, speed: 2.09 step/s
    global step 950, epoch: 2, batch: 470, loss: 0.01841, accu: 0.95300, speed: 2.09 step/s
    global step 960, epoch: 2, batch: 480, loss: 0.16434, accu: 0.95500, speed: 2.12 step/s
    global step 970, epoch: 3, batch: 10, loss: 0.01268, accu: 0.95857, speed: 2.09 step/s
    global step 980, epoch: 3, batch: 20, loss: 0.01741, accu: 0.96313, speed: 2.13 step/s
    global step 990, epoch: 3, batch: 30, loss: 0.00250, accu: 0.96556, speed: 2.10 step/s
    global step 1000, epoch: 3, batch: 40, loss: 0.04604, accu: 0.96650, speed: 2.10 step/s
    eval loss: 0.17832, accu: 0.94250
    global step 1010, epoch: 3, batch: 50, loss: 0.01056, accu: 0.97000, speed: 0.34 step/s
    global step 1020, epoch: 3, batch: 60, loss: 0.17508, accu: 0.97500, speed: 2.12 step/s
    global step 1030, epoch: 3, batch: 70, loss: 0.03105, accu: 0.97667, speed: 2.12 step/s
    global step 1040, epoch: 3, batch: 80, loss: 0.00211, accu: 0.97875, speed: 2.10 step/s
    global step 1050, epoch: 3, batch: 90, loss: 0.02796, accu: 0.98000, speed: 2.12 step/s
    global step 1060, epoch: 3, batch: 100, loss: 0.02163, accu: 0.98167, speed: 2.11 step/s
    global step 1070, epoch: 3, batch: 110, loss: 0.02049, accu: 0.98214, speed: 2.11 step/s
    global step 1080, epoch: 3, batch: 120, loss: 0.00498, accu: 0.98375, speed: 2.11 step/s
    global step 1090, epoch: 3, batch: 130, loss: 0.11775, accu: 0.98333, speed: 2.11 step/s
    global step 1100, epoch: 3, batch: 140, loss: 0.00286, accu: 0.98350, speed: 2.11 step/s
    eval loss: 0.20801, accu: 0.94000
    global step 1110, epoch: 3, batch: 150, loss: 0.05442, accu: 0.99000, speed: 0.35 step/s
    global step 1120, epoch: 3, batch: 160, loss: 0.22692, accu: 0.99000, speed: 2.11 step/s
    global step 1130, epoch: 3, batch: 170, loss: 0.23527, accu: 0.98833, speed: 2.10 step/s
    global step 1140, epoch: 3, batch: 180, loss: 0.00406, accu: 0.98875, speed: 2.12 step/s
    global step 1150, epoch: 3, batch: 190, loss: 0.01034, accu: 0.98800, speed: 2.12 step/s
    global step 1160, epoch: 3, batch: 200, loss: 0.02243, accu: 0.98917, speed: 2.09 step/s
    global step 1170, epoch: 3, batch: 210, loss: 0.18565, accu: 0.99000, speed: 2.10 step/s
    global step 1180, epoch: 3, batch: 220, loss: 0.12593, accu: 0.98875, speed: 2.09 step/s
    global step 1190, epoch: 3, batch: 230, loss: 0.02952, accu: 0.98722, speed: 2.12 step/s
    global step 1200, epoch: 3, batch: 240, loss: 0.02360, accu: 0.98550, speed: 2.11 step/s
    eval loss: 0.20882, accu: 0.93667
    global step 1210, epoch: 3, batch: 250, loss: 0.00685, accu: 0.98500, speed: 0.35 step/s
    global step 1220, epoch: 3, batch: 260, loss: 0.06420, accu: 0.98250, speed: 2.11 step/s
    global step 1230, epoch: 3, batch: 270, loss: 0.01404, accu: 0.97833, speed: 2.11 step/s
    global step 1240, epoch: 3, batch: 280, loss: 0.04239, accu: 0.98000, speed: 2.12 step/s
    global step 1250, epoch: 3, batch: 290, loss: 0.17109, accu: 0.98000, speed: 2.11 step/s
    global step 1260, epoch: 3, batch: 300, loss: 0.00354, accu: 0.98167, speed: 2.11 step/s
    global step 1270, epoch: 3, batch: 310, loss: 0.01427, accu: 0.98143, speed: 2.11 step/s
    global step 1280, epoch: 3, batch: 320, loss: 0.02518, accu: 0.97937, speed: 2.12 step/s
    global step 1290, epoch: 3, batch: 330, loss: 0.02603, accu: 0.97944, speed: 2.11 step/s
    global step 1300, epoch: 3, batch: 340, loss: 0.06825, accu: 0.97900, speed: 2.11 step/s
    eval loss: 0.22403, accu: 0.94417
    global step 1310, epoch: 3, batch: 350, loss: 0.00817, accu: 0.97500, speed: 0.34 step/s
    global step 1320, epoch: 3, batch: 360, loss: 0.10467, accu: 0.98000, speed: 2.10 step/s
    global step 1330, epoch: 3, batch: 370, loss: 0.01421, accu: 0.98000, speed: 2.12 step/s
    global step 1340, epoch: 3, batch: 380, loss: 0.11193, accu: 0.98375, speed: 2.12 step/s
    global step 1350, epoch: 3, batch: 390, loss: 0.00187, accu: 0.98200, speed: 2.12 step/s
    global step 1360, epoch: 3, batch: 400, loss: 0.12342, accu: 0.98083, speed: 2.12 step/s
    global step 1370, epoch: 3, batch: 410, loss: 0.00354, accu: 0.98071, speed: 2.10 step/s
    global step 1380, epoch: 3, batch: 420, loss: 0.05424, accu: 0.97937, speed: 2.11 step/s
    global step 1390, epoch: 3, batch: 430, loss: 0.19338, accu: 0.98000, speed: 2.10 step/s
    global step 1400, epoch: 3, batch: 440, loss: 0.09014, accu: 0.98050, speed: 2.11 step/s
    eval loss: 0.20496, accu: 0.94250
    global step 1410, epoch: 3, batch: 450, loss: 0.03076, accu: 0.96500, speed: 0.34 step/s
    global step 1420, epoch: 3, batch: 460, loss: 0.06543, accu: 0.96500, speed: 2.11 step/s
    global step 1430, epoch: 3, batch: 470, loss: 0.02071, accu: 0.96833, speed: 2.12 step/s
    global step 1440, epoch: 3, batch: 480, loss: 0.10331, accu: 0.97000, speed: 2.15 step/s
    global step 1450, epoch: 4, batch: 10, loss: 0.03837, accu: 0.97300, speed: 2.07 step/s
    global step 1460, epoch: 4, batch: 20, loss: 0.03014, accu: 0.97750, speed: 2.11 step/s
    global step 1470, epoch: 4, batch: 30, loss: 0.00427, accu: 0.97857, speed: 2.12 step/s
    global step 1480, epoch: 4, batch: 40, loss: 0.00315, accu: 0.98062, speed: 2.12 step/s
    global step 1490, epoch: 4, batch: 50, loss: 0.00463, accu: 0.98167, speed: 2.10 step/s
    global step 1500, epoch: 4, batch: 60, loss: 0.11253, accu: 0.98250, speed: 2.10 step/s
    eval loss: 0.20801, accu: 0.95000
    global step 1510, epoch: 4, batch: 70, loss: 0.00583, accu: 1.00000, speed: 0.34 step/s
    global step 1520, epoch: 4, batch: 80, loss: 0.00116, accu: 1.00000, speed: 2.12 step/s
    global step 1530, epoch: 4, batch: 90, loss: 0.12053, accu: 0.99500, speed: 2.12 step/s
    global step 1540, epoch: 4, batch: 100, loss: 0.04390, accu: 0.99125, speed: 2.12 step/s
    global step 1550, epoch: 4, batch: 110, loss: 0.00650, accu: 0.99100, speed: 2.13 step/s
    global step 1560, epoch: 4, batch: 120, loss: 0.01139, accu: 0.99250, speed: 2.12 step/s
    global step 1570, epoch: 4, batch: 130, loss: 0.00491, accu: 0.99214, speed: 2.11 step/s
    global step 1580, epoch: 4, batch: 140, loss: 0.00918, accu: 0.99187, speed: 2.10 step/s
    global step 1590, epoch: 4, batch: 150, loss: 0.00107, accu: 0.99167, speed: 2.09 step/s
    global step 1600, epoch: 4, batch: 160, loss: 0.17874, accu: 0.99200, speed: 2.10 step/s
    eval loss: 0.23476, accu: 0.94500
    global step 1610, epoch: 4, batch: 170, loss: 0.02435, accu: 0.98500, speed: 0.34 step/s
    global step 1620, epoch: 4, batch: 180, loss: 0.21216, accu: 0.98500, speed: 2.11 step/s
    global step 1630, epoch: 4, batch: 190, loss: 0.00713, accu: 0.98500, speed: 2.11 step/s
    global step 1640, epoch: 4, batch: 200, loss: 0.04036, accu: 0.98500, speed: 2.10 step/s
    global step 1650, epoch: 4, batch: 210, loss: 0.08170, accu: 0.98600, speed: 2.10 step/s
    global step 1660, epoch: 4, batch: 220, loss: 0.00496, accu: 0.98500, speed: 2.11 step/s
    global step 1670, epoch: 4, batch: 230, loss: 0.00629, accu: 0.98571, speed: 2.08 step/s
    global step 1680, epoch: 4, batch: 240, loss: 0.01479, accu: 0.98687, speed: 2.07 step/s
    global step 1690, epoch: 4, batch: 250, loss: 0.01058, accu: 0.98611, speed: 2.08 step/s
    global step 1700, epoch: 4, batch: 260, loss: 0.00524, accu: 0.98750, speed: 2.07 step/s
    eval loss: 0.26514, accu: 0.94333
    global step 1710, epoch: 4, batch: 270, loss: 0.00180, accu: 0.99000, speed: 0.34 step/s
    global step 1720, epoch: 4, batch: 280, loss: 0.03001, accu: 0.99500, speed: 2.12 step/s
    global step 1730, epoch: 4, batch: 290, loss: 0.19720, accu: 0.98833, speed: 2.11 step/s
    global step 1740, epoch: 4, batch: 300, loss: 0.00364, accu: 0.98875, speed: 2.12 step/s
    global step 1750, epoch: 4, batch: 310, loss: 0.00480, accu: 0.98800, speed: 2.10 step/s
    global step 1760, epoch: 4, batch: 320, loss: 0.00561, accu: 0.98667, speed: 2.11 step/s
    global step 1770, epoch: 4, batch: 330, loss: 0.08540, accu: 0.98643, speed: 2.11 step/s
    global step 1780, epoch: 4, batch: 340, loss: 0.11045, accu: 0.98562, speed: 2.11 step/s
    global step 1790, epoch: 4, batch: 350, loss: 0.01310, accu: 0.98556, speed: 2.12 step/s
    global step 1800, epoch: 4, batch: 360, loss: 0.11485, accu: 0.98450, speed: 2.11 step/s
    eval loss: 0.19862, accu: 0.94917
    global step 1810, epoch: 4, batch: 370, loss: 0.01272, accu: 0.98500, speed: 0.34 step/s
    global step 1820, epoch: 4, batch: 380, loss: 0.00728, accu: 0.98500, speed: 2.12 step/s
    global step 1830, epoch: 4, batch: 390, loss: 0.01882, accu: 0.98167, speed: 2.11 step/s
    global step 1840, epoch: 4, batch: 400, loss: 0.03454, accu: 0.98250, speed: 2.11 step/s
    global step 1850, epoch: 4, batch: 410, loss: 0.00542, accu: 0.98500, speed: 2.11 step/s
    global step 1860, epoch: 4, batch: 420, loss: 0.08424, accu: 0.98250, speed: 2.11 step/s
    global step 1870, epoch: 4, batch: 430, loss: 0.04708, accu: 0.98357, speed: 2.11 step/s
    global step 1880, epoch: 4, batch: 440, loss: 0.13142, accu: 0.98375, speed: 2.11 step/s
    global step 1890, epoch: 4, batch: 450, loss: 0.00463, accu: 0.98389, speed: 2.11 step/s
    global step 1900, epoch: 4, batch: 460, loss: 0.06545, accu: 0.98300, speed: 2.11 step/s
    eval loss: 0.22056, accu: 0.95000
    global step 1910, epoch: 4, batch: 470, loss: 0.00246, accu: 1.00000, speed: 0.34 step/s
    global step 1920, epoch: 4, batch: 480, loss: 0.00599, accu: 0.99750, speed: 2.15 step/s
    global step 1930, epoch: 5, batch: 10, loss: 0.00468, accu: 0.99833, speed: 2.08 step/s
    global step 1940, epoch: 5, batch: 20, loss: 0.06147, accu: 0.99750, speed: 2.10 step/s
    global step 1950, epoch: 5, batch: 30, loss: 0.00439, accu: 0.99600, speed: 2.11 step/s
    global step 1960, epoch: 5, batch: 40, loss: 0.00056, accu: 0.99500, speed: 2.11 step/s
    global step 1970, epoch: 5, batch: 50, loss: 0.00207, accu: 0.99429, speed: 2.12 step/s
    global step 1980, epoch: 5, batch: 60, loss: 0.03551, accu: 0.99438, speed: 2.10 step/s
    global step 1990, epoch: 5, batch: 70, loss: 0.02504, accu: 0.99389, speed: 2.09 step/s
    global step 2000, epoch: 5, batch: 80, loss: 0.00138, accu: 0.99400, speed: 2.10 step/s
    eval loss: 0.27716, accu: 0.93917
    global step 2010, epoch: 5, batch: 90, loss: 0.02371, accu: 1.00000, speed: 0.34 step/s
    global step 2020, epoch: 5, batch: 100, loss: 0.00390, accu: 0.99250, speed: 2.11 step/s
    global step 2030, epoch: 5, batch: 110, loss: 0.01003, accu: 0.99333, speed: 2.11 step/s
    global step 2040, epoch: 5, batch: 120, loss: 0.17414, accu: 0.99250, speed: 2.11 step/s
    global step 2050, epoch: 5, batch: 130, loss: 0.00887, accu: 0.98900, speed: 2.10 step/s
    global step 2060, epoch: 5, batch: 140, loss: 0.00043, accu: 0.98917, speed: 2.10 step/s
    global step 2070, epoch: 5, batch: 150, loss: 0.08947, accu: 0.98857, speed: 2.10 step/s
    global step 2080, epoch: 5, batch: 160, loss: 0.00083, accu: 0.98938, speed: 2.10 step/s
    global step 2090, epoch: 5, batch: 170, loss: 0.01438, accu: 0.98944, speed: 2.10 step/s
    global step 2100, epoch: 5, batch: 180, loss: 0.00039, accu: 0.99000, speed: 2.10 step/s
    eval loss: 0.29346, accu: 0.94083
    global step 2110, epoch: 5, batch: 190, loss: 0.00070, accu: 0.98500, speed: 0.34 step/s
    global step 2120, epoch: 5, batch: 200, loss: 0.01716, accu: 0.98750, speed: 2.13 step/s
    global step 2130, epoch: 5, batch: 210, loss: 0.00368, accu: 0.98667, speed: 2.12 step/s
    global step 2140, epoch: 5, batch: 220, loss: 0.01363, accu: 0.98500, speed: 2.11 step/s
    global step 2150, epoch: 5, batch: 230, loss: 0.00149, accu: 0.98500, speed: 2.12 step/s
    global step 2160, epoch: 5, batch: 240, loss: 0.00389, accu: 0.98583, speed: 2.11 step/s
    global step 2170, epoch: 5, batch: 250, loss: 0.02237, accu: 0.98786, speed: 2.11 step/s
    global step 2180, epoch: 5, batch: 260, loss: 0.03758, accu: 0.98625, speed: 2.10 step/s
    global step 2190, epoch: 5, batch: 270, loss: 0.00395, accu: 0.98667, speed: 2.13 step/s
    global step 2200, epoch: 5, batch: 280, loss: 0.00281, accu: 0.98700, speed: 2.12 step/s
    eval loss: 0.24953, accu: 0.94750
    global step 2210, epoch: 5, batch: 290, loss: 0.06479, accu: 0.99500, speed: 0.34 step/s
    global step 2220, epoch: 5, batch: 300, loss: 0.00362, accu: 0.99000, speed: 2.12 step/s
    global step 2230, epoch: 5, batch: 310, loss: 0.00133, accu: 0.99167, speed: 2.12 step/s
    global step 2240, epoch: 5, batch: 320, loss: 0.01455, accu: 0.99250, speed: 2.10 step/s
    global step 2250, epoch: 5, batch: 330, loss: 0.00616, accu: 0.99400, speed: 2.11 step/s
    global step 2260, epoch: 5, batch: 340, loss: 0.02826, accu: 0.99333, speed: 2.12 step/s
    global step 2270, epoch: 5, batch: 350, loss: 0.00314, accu: 0.99357, speed: 2.11 step/s
    global step 2280, epoch: 5, batch: 360, loss: 0.00606, accu: 0.99313, speed: 2.10 step/s
    global step 2290, epoch: 5, batch: 370, loss: 0.13500, accu: 0.99222, speed: 2.12 step/s
    global step 2300, epoch: 5, batch: 380, loss: 0.01494, accu: 0.99100, speed: 2.10 step/s
    eval loss: 0.22059, accu: 0.94833
    global step 2310, epoch: 5, batch: 390, loss: 0.00501, accu: 0.99500, speed: 0.34 step/s
    global step 2320, epoch: 5, batch: 400, loss: 0.00110, accu: 0.99750, speed: 2.11 step/s
    global step 2330, epoch: 5, batch: 410, loss: 0.07260, accu: 0.99500, speed: 2.11 step/s
    global step 2340, epoch: 5, batch: 420, loss: 0.00071, accu: 0.99625, speed: 2.10 step/s
    global step 2350, epoch: 5, batch: 430, loss: 0.02240, accu: 0.99600, speed: 2.12 step/s
    global step 2360, epoch: 5, batch: 440, loss: 0.00094, accu: 0.99583, speed: 2.11 step/s
    global step 2370, epoch: 5, batch: 450, loss: 0.00046, accu: 0.99571, speed: 2.11 step/s
    global step 2380, epoch: 5, batch: 460, loss: 0.01053, accu: 0.99625, speed: 2.10 step/s
    global step 2390, epoch: 5, batch: 470, loss: 0.00535, accu: 0.99611, speed: 2.11 step/s
    global step 2400, epoch: 5, batch: 480, loss: 0.00095, accu: 0.99600, speed: 2.14 step/s
    eval loss: 0.25939, accu: 0.95583
    global step 2410, epoch: 6, batch: 10, loss: 0.05473, accu: 0.99500, speed: 0.34 step/s
    global step 2420, epoch: 6, batch: 20, loss: 0.00022, accu: 0.99250, speed: 2.13 step/s
    global step 2430, epoch: 6, batch: 30, loss: 0.00084, accu: 0.99167, speed: 2.12 step/s
    global step 2440, epoch: 6, batch: 40, loss: 0.00049, accu: 0.99375, speed: 2.11 step/s
    global step 2450, epoch: 6, batch: 50, loss: 0.00121, accu: 0.99300, speed: 2.12 step/s
    global step 2460, epoch: 6, batch: 60, loss: 0.11661, accu: 0.99083, speed: 2.10 step/s
    global step 2470, epoch: 6, batch: 70, loss: 0.00209, accu: 0.99071, speed: 2.11 step/s
    global step 2480, epoch: 6, batch: 80, loss: 0.01499, accu: 0.99125, speed: 2.10 step/s
    global step 2490, epoch: 6, batch: 90, loss: 0.00098, accu: 0.99111, speed: 2.12 step/s
    global step 2500, epoch: 6, batch: 100, loss: 0.00047, accu: 0.99150, speed: 2.12 step/s
    eval loss: 0.26659, accu: 0.94000
    global step 2510, epoch: 6, batch: 110, loss: 0.00937, accu: 1.00000, speed: 0.34 step/s
    global step 2520, epoch: 6, batch: 120, loss: 0.00596, accu: 0.99250, speed: 2.10 step/s
    global step 2530, epoch: 6, batch: 130, loss: 0.00506, accu: 0.99000, speed: 2.08 step/s
    global step 2540, epoch: 6, batch: 140, loss: 0.00139, accu: 0.99000, speed: 2.08 step/s
    global step 2550, epoch: 6, batch: 150, loss: 0.01524, accu: 0.98800, speed: 2.07 step/s
    global step 2560, epoch: 6, batch: 160, loss: 0.00405, accu: 0.98833, speed: 2.09 step/s
    global step 2570, epoch: 6, batch: 170, loss: 0.00958, accu: 0.99000, speed: 2.08 step/s
    global step 2580, epoch: 6, batch: 180, loss: 0.00104, accu: 0.99000, speed: 2.09 step/s
    global step 2590, epoch: 6, batch: 190, loss: 0.02491, accu: 0.98944, speed: 2.08 step/s
    global step 2600, epoch: 6, batch: 200, loss: 0.00579, accu: 0.99000, speed: 2.09 step/s
    eval loss: 0.20523, accu: 0.94500
    global step 2610, epoch: 6, batch: 210, loss: 0.02198, accu: 0.99000, speed: 0.34 step/s
    global step 2620, epoch: 6, batch: 220, loss: 0.00218, accu: 0.99250, speed: 2.12 step/s
    global step 2630, epoch: 6, batch: 230, loss: 0.06024, accu: 0.99167, speed: 2.11 step/s
    global step 2640, epoch: 6, batch: 240, loss: 0.01196, accu: 0.99375, speed: 2.11 step/s
    global step 2650, epoch: 6, batch: 250, loss: 0.00681, accu: 0.99300, speed: 2.11 step/s
    global step 2660, epoch: 6, batch: 260, loss: 0.00567, accu: 0.99333, speed: 2.11 step/s
    global step 2670, epoch: 6, batch: 270, loss: 0.00048, accu: 0.99071, speed: 2.10 step/s
    global step 2680, epoch: 6, batch: 280, loss: 0.00303, accu: 0.99187, speed: 2.10 step/s
    global step 2690, epoch: 6, batch: 290, loss: 0.00093, accu: 0.99167, speed: 2.11 step/s
    global step 2700, epoch: 6, batch: 300, loss: 0.02021, accu: 0.99200, speed: 2.11 step/s
    eval loss: 0.26426, accu: 0.94500
    global step 2710, epoch: 6, batch: 310, loss: 0.00034, accu: 1.00000, speed: 0.34 step/s
    global step 2720, epoch: 6, batch: 320, loss: 0.01036, accu: 0.99250, speed: 2.11 step/s
    global step 2730, epoch: 6, batch: 330, loss: 0.00159, accu: 0.99000, speed: 2.13 step/s
    global step 2740, epoch: 6, batch: 340, loss: 0.00811, accu: 0.99125, speed: 2.14 step/s
    global step 2750, epoch: 6, batch: 350, loss: 0.00064, accu: 0.99100, speed: 2.12 step/s
    global step 2760, epoch: 6, batch: 360, loss: 0.00776, accu: 0.99000, speed: 2.12 step/s
    global step 2770, epoch: 6, batch: 370, loss: 0.00065, accu: 0.99000, speed: 2.12 step/s
    global step 2780, epoch: 6, batch: 380, loss: 0.02581, accu: 0.99000, speed: 2.11 step/s
    global step 2790, epoch: 6, batch: 390, loss: 0.00494, accu: 0.98944, speed: 2.11 step/s
    global step 2800, epoch: 6, batch: 400, loss: 0.00518, accu: 0.98850, speed: 2.09 step/s
    eval loss: 0.25173, accu: 0.95000
    global step 2810, epoch: 6, batch: 410, loss: 0.00219, accu: 0.98500, speed: 0.34 step/s
    global step 2820, epoch: 6, batch: 420, loss: 0.03799, accu: 0.98500, speed: 2.11 step/s
    global step 2830, epoch: 6, batch: 430, loss: 0.00116, accu: 0.98833, speed: 2.12 step/s
    global step 2840, epoch: 6, batch: 440, loss: 0.12505, accu: 0.98625, speed: 2.11 step/s
    global step 2850, epoch: 6, batch: 450, loss: 0.02803, accu: 0.98800, speed: 2.12 step/s
    global step 2860, epoch: 6, batch: 460, loss: 0.01766, accu: 0.99000, speed: 2.10 step/s
    global step 2870, epoch: 6, batch: 470, loss: 0.01021, accu: 0.98929, speed: 2.11 step/s
    global step 2880, epoch: 6, batch: 480, loss: 0.00318, accu: 0.99000, speed: 2.14 step/s
    global step 2890, epoch: 7, batch: 10, loss: 0.00104, accu: 0.99000, speed: 2.09 step/s
    global step 2900, epoch: 7, batch: 20, loss: 0.02120, accu: 0.98950, speed: 2.10 step/s
    eval loss: 0.20966, accu: 0.95000
    global step 2910, epoch: 7, batch: 30, loss: 0.00365, accu: 0.99500, speed: 0.34 step/s
    global step 2920, epoch: 7, batch: 40, loss: 0.00092, accu: 0.99000, speed: 2.12 step/s
    global step 2930, epoch: 7, batch: 50, loss: 0.00045, accu: 0.98833, speed: 2.11 step/s
    global step 2940, epoch: 7, batch: 60, loss: 0.00243, accu: 0.98750, speed: 2.10 step/s
    global step 2950, epoch: 7, batch: 70, loss: 0.01829, accu: 0.99000, speed: 2.11 step/s
    global step 2960, epoch: 7, batch: 80, loss: 0.06863, accu: 0.98833, speed: 2.12 step/s
    global step 2970, epoch: 7, batch: 90, loss: 0.00231, accu: 0.98786, speed: 2.11 step/s
    global step 2980, epoch: 7, batch: 100, loss: 0.00491, accu: 0.98875, speed: 2.10 step/s
    global step 2990, epoch: 7, batch: 110, loss: 0.02363, accu: 0.98944, speed: 2.11 step/s
    global step 3000, epoch: 7, batch: 120, loss: 0.00049, accu: 0.98900, speed: 2.11 step/s
    eval loss: 0.26964, accu: 0.94083
    global step 3010, epoch: 7, batch: 130, loss: 0.00222, accu: 1.00000, speed: 0.34 step/s
    global step 3020, epoch: 7, batch: 140, loss: 0.08524, accu: 0.99250, speed: 2.12 step/s
    global step 3030, epoch: 7, batch: 150, loss: 0.00022, accu: 0.99500, speed: 2.11 step/s
    global step 3040, epoch: 7, batch: 160, loss: 0.00173, accu: 0.99625, speed: 2.12 step/s
    global step 3050, epoch: 7, batch: 170, loss: 0.00021, accu: 0.99700, speed: 2.11 step/s
    global step 3060, epoch: 7, batch: 180, loss: 0.09083, accu: 0.99583, speed: 2.11 step/s
    global step 3070, epoch: 7, batch: 190, loss: 0.00046, accu: 0.99500, speed: 2.11 step/s
    global step 3080, epoch: 7, batch: 200, loss: 0.00132, accu: 0.99438, speed: 2.10 step/s
    global step 3090, epoch: 7, batch: 210, loss: 0.01907, accu: 0.99500, speed: 2.11 step/s
    global step 3100, epoch: 7, batch: 220, loss: 0.00194, accu: 0.99500, speed: 2.09 step/s
    eval loss: 0.30268, accu: 0.94500
    global step 3110, epoch: 7, batch: 230, loss: 0.00054, accu: 0.99000, speed: 0.35 step/s
    global step 3120, epoch: 7, batch: 240, loss: 0.00202, accu: 0.99250, speed: 2.09 step/s
    global step 3130, epoch: 7, batch: 250, loss: 0.00169, accu: 0.99167, speed: 2.08 step/s
    global step 3140, epoch: 7, batch: 260, loss: 0.04441, accu: 0.98875, speed: 2.08 step/s
    global step 3150, epoch: 7, batch: 270, loss: 0.00642, accu: 0.98800, speed: 2.09 step/s
    global step 3160, epoch: 7, batch: 280, loss: 0.00170, accu: 0.98917, speed: 2.12 step/s
    global step 3170, epoch: 7, batch: 290, loss: 0.00720, accu: 0.99071, speed: 2.11 step/s
    global step 3180, epoch: 7, batch: 300, loss: 0.00031, accu: 0.99125, speed: 2.11 step/s
    global step 3190, epoch: 7, batch: 310, loss: 0.00523, accu: 0.99167, speed: 2.12 step/s
    global step 3200, epoch: 7, batch: 320, loss: 0.00163, accu: 0.99150, speed: 2.12 step/s
    eval loss: 0.26829, accu: 0.95083
    global step 3210, epoch: 7, batch: 330, loss: 0.00114, accu: 0.99500, speed: 0.34 step/s



    ---------------------------------------------------------------------------
    
    KeyboardInterrupt                         Traceback (most recent call last)
    
    <ipython-input-8-d4cb7c86a209> in <module>
          8         logits = model(input_ids, token_type_ids)
          9         # 计算损失函数值
    ---> 10         loss = criterion(logits, labels)
         11         # 预测分类概率值
         12         probs = F.softmax(logits, axis=1)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        896                 self._built = True
        897 
    --> 898             outputs = self.forward(*inputs, **kwargs)
        899 
        900             for forward_post_hook in self._forward_post_hooks.values():


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/loss.py in forward(self, input, label)
        403             axis=self.axis,
        404             use_softmax=self.use_softmax,
    --> 405             name=self.name)
        406 
        407         return ret


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/functional/loss.py in cross_entropy(input, label, weight, ignore_index, reduction, soft_label, axis, use_softmax, name)
       1390             input, label, 'soft_label', soft_label, 'ignore_index',
       1391             ignore_index, 'numeric_stable_mode', True, 'axis', axis,
    -> 1392             'use_softmax', use_softmax)
       1393 
       1394         if weight is not None:


    KeyboardInterrupt: 


## 4.预测提交结果


使用训练得到的模型还可以对文本进行情感预测。



```python
import numpy as np
import paddle

# 处理测试集数据
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    is_test=True)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack() # qid
): [data for data in fn(samples)]
test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```


```python
# 根据实际运行情况，更换加载的参数路径
params_path = 'skep_ckp/model_3200/model_state.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
```


```python
label_map = {0: '0', 1: '1'}
results = []
# 切换model模型为评估模式，关闭dropout等随机因素
model.eval()
for batch in test_data_loader:
    input_ids, token_type_ids, qids = batch
    # 喂数据给模型
    logits = model(input_ids, token_type_ids)
    # 预测分类
    probs = F.softmax(logits, axis=-1)
    idx = paddle.argmax(probs, axis=1).numpy()
    idx = idx.tolist()
    labels = [label_map[i] for i in idx]
    qids = qids.numpy().tolist()
    results.extend(zip(qids, labels))
```


```python
res_dir = "./results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# 写入预测结果
with open(os.path.join(res_dir, "ChnSentiCorp.tsv"), 'w', encoding="utf8") as f:
    f.write("index\tprediction\n")
    for qid, label in results:
        f.write(str(qid[0])+"\t"+label+"\n")
```

# 二、SE-ABSA16_PHNS数据集【0.6181】
## 1.Part C.2 目标级情感分析

在电商产品分析场景下，除了分析整体商品的情感极性外，还细化到以商品具体的“方面”为分析主体进行情感分析（aspect-level），如下、：

* 这个薯片口味有点咸，太辣了，不过口感很脆。

关于薯片的**口味方面**是一个负向评价（咸，太辣），然而对于**口感方面**却是一个正向评价（很脆）。

* 我很喜欢夏威夷，就是这边的海鲜太贵了。

关于**夏威夷**是一个正向评价（喜欢），然而对于**夏威夷的海鲜**却是一个负向评价（价格太贵）。



<br></br>
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/052d46409ba3451693a718552b968d188fa4677235bc43ddbc15fe11ad3b57b1" width="600" ></center>
<br><center>目标级情感分析任务</center></br>


[千言数据集](https://www.luge.ai/)已提供了许多任务常用数据集。
其中情感分析数据集下载链接：https://aistudio.baidu.com/aistudio/competition/detail/50/?isFromLUGE=TRUE

SE-ABSA16_PHNS数据集是关于手机的目标级情感分析数据集。PaddleNLP已经内置了该数据集，加载方式，如下：



```python
from paddlenlp.datasets import load_dataset
train_ds, test_ds = load_dataset("seabsa16", "phns", splits=["train", "test"])

print(train_ds[0])
print(train_ds[1])
print(train_ds[2])
```

    {'text': 'phone#design_features', 'text_pair': '今天有幸拿到了港版白色iPhone 5真机，试玩了一下，说说感受吧：1. 真机尺寸宽度与4/4s保持一致没有变化，长度多了大概一厘米，也就是之前所说的多了一排的图标。2. 真机重量比上一代轻了很多，个人感觉跟i9100的重量差不多。（用惯上一代的朋友可能需要一段时间适应了）3. 由于目前还没有版的SIM卡，无法插卡使用，有购买的朋友要注意了，并非简单的剪卡就可以用，而是需要去运营商更换新一代的SIM卡。4. 屏幕显示效果确实比上一代有进步，不论是从清晰度还是不同角度的视角，iPhone 5绝对要更上一层，我想这也许是相对上一代最有意义的升级了。5. 新的数据接口更小，比上一代更好用更方便，使用的过程会有这样的体会。6. 从简单的几个操作来讲速度比4s要快，这个不用测试软件也能感受出来，比如程序的调用以及照片的拍摄和浏览。不过，目前水货市场上坑爹的价格，最好大家可以再观望一下，不要急着出手。', 'label': 1}
    {'text': 'display#quality', 'text_pair': '今天有幸拿到了港版白色iPhone 5真机，试玩了一下，说说感受吧：1. 真机尺寸宽度与4/4s保持一致没有变化，长度多了大概一厘米，也就是之前所说的多了一排的图标。2. 真机重量比上一代轻了很多，个人感觉跟i9100的重量差不多。（用惯上一代的朋友可能需要一段时间适应了）3. 由于目前还没有版的SIM卡，无法插卡使用，有购买的朋友要注意了，并非简单的剪卡就可以用，而是需要去运营商更换新一代的SIM卡。4. 屏幕显示效果确实比上一代有进步，不论是从清晰度还是不同角度的视角，iPhone 5绝对要更上一层，我想这也许是相对上一代最有意义的升级了。5. 新的数据接口更小，比上一代更好用更方便，使用的过程会有这样的体会。6. 从简单的几个操作来讲速度比4s要快，这个不用测试软件也能感受出来，比如程序的调用以及照片的拍摄和浏览。不过，目前水货市场上坑爹的价格，最好大家可以再观望一下，不要急着出手。', 'label': 1}
    {'text': 'ports#connectivity', 'text_pair': '今天有幸拿到了港版白色iPhone 5真机，试玩了一下，说说感受吧：1. 真机尺寸宽度与4/4s保持一致没有变化，长度多了大概一厘米，也就是之前所说的多了一排的图标。2. 真机重量比上一代轻了很多，个人感觉跟i9100的重量差不多。（用惯上一代的朋友可能需要一段时间适应了）3. 由于目前还没有版的SIM卡，无法插卡使用，有购买的朋友要注意了，并非简单的剪卡就可以用，而是需要去运营商更换新一代的SIM卡。4. 屏幕显示效果确实比上一代有进步，不论是从清晰度还是不同角度的视角，iPhone 5绝对要更上一层，我想这也许是相对上一代最有意义的升级了。5. 新的数据接口更小，比上一代更好用更方便，使用的过程会有这样的体会。6. 从简单的几个操作来讲速度比4s要快，这个不用测试软件也能感受出来，比如程序的调用以及照片的拍摄和浏览。不过，目前水货市场上坑爹的价格，最好大家可以再观望一下，不要急着出手。', 'label': 1}


## 2.SKEP模型加载

目标级情感分析模型同样使用`SkepForSequenceClassification`模型，但目标级情感分析模型的输入不单单是一个句子，而是句对。一个句子描述“评价对象方面（aspect）”，另一个句子描述"对该方面的评论"。如下图所示。


![](https://ai-studio-static-online.cdn.bcebos.com/1a4b76447dae404caa3bf123ea28e375179cb09a02de4bef8a2f172edc6e3c8f)




```python
# 指定模型名称一键加载模型
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

model = SkepForSequenceClassification.from_pretrained(
    'skep_ernie_1.0_large_ch', num_classes=len(train_ds.label_list))
# 指定模型名称一键加载tokenizer
tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')
```

    [2021-06-16 01:08:11,541] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    [2021-06-16 01:08:21,501] [    INFO] - Found /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.vocab.txt


## 3.数据处理

同样地，我们需要将原始SE_ABSA16_PHNS数据处理成模型可以读入的数据格式。

SKEP模型对中文文本处理按照字粒度进行处理，我们可以使用PaddleNLP内置的`SkepTokenizer`完成一键式处理。


```python
from functools import partial
import os
import time

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False,
                    dataset_name="chnsenticorp"):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence has the following format:
    ::
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence pair mask has the following format:
    ::

        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
    
    note: There is no need token type ids for skep_roberta_large_ch model.


    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.
        dataset_name((obj:`str`, defaults to "chnsenticorp"): The dataset name, "chnsenticorp" or "sst-2".

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    encoded_inputs = tokenizer(
        text=example["text"],
        text_pair=example["text_pair"],
        max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids
```


```python
from utils import create_dataloader
# 处理的最大文本序列长度
max_seq_length=256
# 批量数据大小
batch_size=20

# 将数据处理成model可读入的数据格式
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)
# 将数据组成批量式数据，如
# 将不同长度的文本序列padding到批量式数据中最大长度
# 将每条数据label堆叠在一起
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(dtype="int64")  # labels
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

## 4.模型训练

定义损失函数、优化器以及评价指标后，即可开始训练。


```python
# 训练轮次
epochs = 10
# 总共需要训练的step数
num_training_steps = len(train_data_loader) * epochs
# 优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=5e-5,
    parameters=model.parameters())
# 交叉熵损失
criterion = paddle.nn.loss.CrossEntropyLoss()
# Accuracy评价指标
metric = paddle.metric.Accuracy()
```


```python
# 开启训练
ckpt_dir = "skep_aspect"
global_step = 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        # 喂数据给model
        logits = model(input_ids, token_type_ids)
        # 计算损失函数值
        loss = criterion(logits, labels)
        # 预测分类概率
        probs = F.softmax(logits, axis=1)
        # 计算acc
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:

            save_dir = os.path.join(ckpt_dir, "model_%d" % global_step)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 保存模型参数
            model.save_pretrained(save_dir)
            # 保存tokenizer的词表等
            tokenizer.save_pretrained(save_dir)
```

    global step 10, epoch: 1, batch: 10, loss: 0.67361, acc: 0.58000, speed: 1.00 step/s
    global step 20, epoch: 1, batch: 20, loss: 0.70621, acc: 0.57000, speed: 1.00 step/s
    global step 30, epoch: 1, batch: 30, loss: 0.57455, acc: 0.59000, speed: 1.00 step/s
    global step 40, epoch: 1, batch: 40, loss: 0.74245, acc: 0.58000, speed: 0.99 step/s
    global step 50, epoch: 1, batch: 50, loss: 0.56560, acc: 0.60100, speed: 0.99 step/s
    global step 60, epoch: 1, batch: 60, loss: 0.76653, acc: 0.60500, speed: 0.99 step/s
    global step 70, epoch: 2, batch: 3, loss: 0.78028, acc: 0.60602, speed: 1.00 step/s
    global step 80, epoch: 2, batch: 13, loss: 0.53692, acc: 0.61466, speed: 0.98 step/s
    global step 90, epoch: 2, batch: 23, loss: 0.64045, acc: 0.61693, speed: 0.98 step/s
    global step 100, epoch: 2, batch: 33, loss: 0.96845, acc: 0.61824, speed: 0.99 step/s
    global step 110, epoch: 2, batch: 43, loss: 0.57729, acc: 0.62614, speed: 0.41 step/s
    global step 120, epoch: 2, batch: 53, loss: 0.46690, acc: 0.62813, speed: 0.99 step/s
    global step 130, epoch: 2, batch: 63, loss: 0.62161, acc: 0.62982, speed: 1.00 step/s
    global step 140, epoch: 3, batch: 6, loss: 0.52529, acc: 0.63324, speed: 0.99 step/s
    global step 150, epoch: 3, batch: 16, loss: 0.57512, acc: 0.63703, speed: 0.98 step/s
    global step 160, epoch: 3, batch: 26, loss: 0.53794, acc: 0.64066, speed: 0.99 step/s
    global step 170, epoch: 3, batch: 36, loss: 0.63488, acc: 0.64269, speed: 1.00 step/s
    global step 180, epoch: 3, batch: 46, loss: 0.69480, acc: 0.64310, speed: 0.99 step/s
    global step 190, epoch: 3, batch: 56, loss: 0.52648, acc: 0.64583, speed: 0.98 step/s
    global step 200, epoch: 3, batch: 66, loss: 0.61528, acc: 0.64755, speed: 1.01 step/s
    global step 210, epoch: 4, batch: 9, loss: 0.63117, acc: 0.64995, speed: 0.40 step/s
    global step 220, epoch: 4, batch: 19, loss: 0.46006, acc: 0.65132, speed: 0.99 step/s
    global step 230, epoch: 4, batch: 29, loss: 0.43491, acc: 0.65366, speed: 0.99 step/s
    global step 240, epoch: 4, batch: 39, loss: 0.61736, acc: 0.65560, speed: 0.98 step/s
    global step 250, epoch: 4, batch: 49, loss: 0.56354, acc: 0.65718, speed: 0.99 step/s
    global step 260, epoch: 4, batch: 59, loss: 0.65595, acc: 0.65690, speed: 0.99 step/s
    global step 270, epoch: 5, batch: 2, loss: 0.47317, acc: 0.65676, speed: 1.01 step/s
    global step 280, epoch: 5, batch: 12, loss: 0.67179, acc: 0.65903, speed: 0.99 step/s
    global step 290, epoch: 5, batch: 22, loss: 0.53909, acc: 0.66027, speed: 0.99 step/s
    global step 300, epoch: 5, batch: 32, loss: 0.63901, acc: 0.66160, speed: 0.99 step/s
    global step 310, epoch: 5, batch: 42, loss: 0.57467, acc: 0.66252, speed: 0.41 step/s
    global step 320, epoch: 5, batch: 52, loss: 0.53458, acc: 0.66322, speed: 0.98 step/s
    global step 330, epoch: 5, batch: 62, loss: 0.60461, acc: 0.66495, speed: 0.99 step/s
    global step 340, epoch: 6, batch: 5, loss: 0.45266, acc: 0.66563, speed: 1.01 step/s
    global step 350, epoch: 6, batch: 15, loss: 0.62391, acc: 0.66662, speed: 0.99 step/s
    global step 360, epoch: 6, batch: 25, loss: 0.54902, acc: 0.66880, speed: 0.99 step/s
    global step 370, epoch: 6, batch: 35, loss: 0.51146, acc: 0.66897, speed: 0.98 step/s
    global step 380, epoch: 6, batch: 45, loss: 0.60425, acc: 0.66887, speed: 0.98 step/s
    global step 390, epoch: 6, batch: 55, loss: 0.54138, acc: 0.67057, speed: 0.97 step/s
    global step 400, epoch: 6, batch: 65, loss: 0.38777, acc: 0.67018, speed: 1.01 step/s
    global step 410, epoch: 7, batch: 8, loss: 0.51965, acc: 0.67246, speed: 0.41 step/s
    global step 420, epoch: 7, batch: 18, loss: 0.54822, acc: 0.67323, speed: 0.99 step/s
    global step 430, epoch: 7, batch: 28, loss: 0.58745, acc: 0.67432, speed: 0.99 step/s
    global step 440, epoch: 7, batch: 38, loss: 0.38566, acc: 0.67479, speed: 0.99 step/s
    global step 450, epoch: 7, batch: 48, loss: 0.76449, acc: 0.67346, speed: 0.99 step/s
    global step 460, epoch: 7, batch: 58, loss: 0.56186, acc: 0.67208, speed: 0.99 step/s
    global step 470, epoch: 8, batch: 1, loss: 0.63861, acc: 0.67157, speed: 1.01 step/s
    global step 480, epoch: 8, batch: 11, loss: 0.57409, acc: 0.67248, speed: 0.99 step/s
    global step 490, epoch: 8, batch: 21, loss: 0.59991, acc: 0.67315, speed: 0.98 step/s
    global step 500, epoch: 8, batch: 31, loss: 0.65107, acc: 0.67389, speed: 0.99 step/s
    global step 510, epoch: 8, batch: 41, loss: 0.50752, acc: 0.67430, speed: 0.41 step/s
    global step 520, epoch: 8, batch: 51, loss: 0.51834, acc: 0.67538, speed: 0.99 step/s
    global step 530, epoch: 8, batch: 61, loss: 0.57100, acc: 0.67499, speed: 0.99 step/s
    global step 540, epoch: 9, batch: 4, loss: 0.40028, acc: 0.67571, speed: 1.00 step/s
    global step 550, epoch: 9, batch: 14, loss: 0.67644, acc: 0.67542, speed: 0.98 step/s
    global step 560, epoch: 9, batch: 24, loss: 0.47629, acc: 0.67541, speed: 0.98 step/s
    global step 570, epoch: 9, batch: 34, loss: 0.50073, acc: 0.67628, speed: 0.98 step/s
    global step 580, epoch: 9, batch: 44, loss: 0.37358, acc: 0.67687, speed: 0.98 step/s
    global step 590, epoch: 9, batch: 54, loss: 0.47544, acc: 0.67803, speed: 0.99 step/s
    global step 600, epoch: 9, batch: 64, loss: 0.41326, acc: 0.67914, speed: 1.01 step/s
    global step 610, epoch: 10, batch: 7, loss: 0.40692, acc: 0.67996, speed: 0.41 step/s
    global step 620, epoch: 10, batch: 17, loss: 0.37122, acc: 0.68077, speed: 0.98 step/s
    global step 630, epoch: 10, batch: 27, loss: 0.63428, acc: 0.68163, speed: 0.99 step/s
    global step 640, epoch: 10, batch: 37, loss: 0.53830, acc: 0.68168, speed: 0.98 step/s
    global step 650, epoch: 10, batch: 47, loss: 0.44218, acc: 0.68220, speed: 0.98 step/s
    global step 660, epoch: 10, batch: 57, loss: 0.54936, acc: 0.68277, speed: 0.99 step/s
    global step 670, epoch: 10, batch: 67, loss: 0.35732, acc: 0.68301, speed: 1.03 step/s


## 5.预测提交结果

使用训练得到的模型还可以对评价对象进行情感预测。


```python
@paddle.no_grad()
def predict(model, data_loader, label_map):
    """
    Given a prediction dataset, it gives the prediction results.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
    """
    model.eval()
    results = []
    for batch in data_loader:
        input_ids, token_type_ids = batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results
```


```python
# 处理测试集数据
label_map = {0: '0', 1: '1'}
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    is_test=True)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
): [data for data in fn(samples)]
test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```


```python
# 根据实际运行情况，更换加载的参数路径
params_path = 'skep_ckpt/model_600/model_state.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)

results = predict(model, test_data_loader, label_map)
```


```python
# 写入预测结果
with open(os.path.join("results", "SE-ABSA16_PHNS.tsv"), 'w', encoding="utf8") as f:
    f.write("index\tprediction\n")
    for idx, label in enumerate(results):
        f.write(str(idx)+"\t"+label+"\n")
```

将预测文件结果压缩至zip文件，提交[千言比赛网站](https://aistudio.baidu.com/aistudio/competition/detail/50/?isFromLUGE=TRUE)

**NOTE:** results文件夹中NLPCC14-SC.tsv、SE-ABSA16_CAME.tsv、COTE_BD.tsv、COTE_MFW.tsv、COTE_DP.tsv等文件是为了顺利提交，补齐的文件。
其结果还有待提高。


```python
#将预测文件结果压缩至zip文件，提交
!zip -r results.zip results
```

      adding: results/ (stored 0%)
      adding: results/SE-ABSA16_PHNS.tsv (deflated 64%)
      adding: results/ChnSentiCorp.tsv (deflated 63%)


# 三、NLPCC14-SC数据【0.8196】

## 1.数据处理


```python
!unzip data/data94266/nlp_dataset.zip
```


```python
!head NLPCC14-SC/train.tsv
```

    label	text_a
    1	请问这机不是有个遥控器的吗？
    1	发短信特别不方便！背后的屏幕很大用起来不舒服，是手触屏的！切换屏幕很麻烦！
    1	手感超好，而且黑色相比白色在转得时候不容易眼花，找童年的记忆啦。
    1	！！！！！
    1	先付款的   有信用
    1	价格质量售后都很满意
    1	书的质量和印刷都不错，字的大小也刚刚好，很清楚，喜欢
    1	超级值得看的一个电影
    1	今天突然看到卓越有卖这个的，可是韩国不是卖没有了吗。虽然是引进版的，可是之前也卖没有了。卓越从哪里找出来的啊



```python
from paddlenlp.datasets import load_dataset
from paddlenlp.datasets import load_dataset
train_ds, test_ds  = load_dataset("chnsenticorp", data_files={"train": "NLPCC14-SC/train.tsv", "test": "NLPCC14-SC/test.tsv"})
dev_ds  = load_dataset("chnsenticorp", data_files={ "test": "NLPCC14-SC/test.tsv"})

print(train_ds[0])
print(train_ds[1])
print(train_ds[2])
```

    {'text': '请问这机不是有个遥控器的吗？', 'label': 1, 'qid': ''}
    {'text': '发短信特别不方便！背后的屏幕很大用起来不舒服，是手触屏的！切换屏幕很麻烦！', 'label': 1, 'qid': ''}
    {'text': '手感超好，而且黑色相比白色在转得时候不容易眼花，找童年的记忆啦。', 'label': 1, 'qid': ''}



```python
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

# 指定模型名称，一键加载模型
model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch", num_classes=len(train_ds.label_list))
# 同样地，通过指定模型名称一键加载对应的Tokenizer，用于处理文本数据，如切分token，转token_id等。
tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch")
```


```python
import os
from functools import partial


import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad

from utils import create_dataloader

def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False):

    # 将原数据处理成model可读入的格式，enocded_inputs是一个dict，包含input_ids、token_type_ids等字段
    encoded_inputs = tokenizer(
        text=example["text"], max_seq_len=max_seq_length)

    # input_ids：对文本切分token后，在词汇表中对应的token id
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids：当前token属于句子1还是句子2，即上述图中表达的segment ids
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        # label：情感极性类别
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        # qid：每条数据的编号
        qid = np.array([example["qid"]], dtype="int64")
        return input_ids, token_type_ids, qid
```


```python
# 批量数据大小
batch_size = 30
# 文本序列最大长度
max_seq_length = 128

# 将数据处理成模型可读入的数据格式
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

# 将数据组成批量式数据，如
# 将不同长度的文本序列padding到批量式数据中最大长度
# 将每条数据label堆叠在一起
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack()  # labels
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```


```python

import time

from utils import evaluate

# 训练轮次
epochs = 10
# 训练过程中保存模型参数的文件夹
ckpt_dir = "nlpcc14"
# len(train_data_loader)一轮训练所需要的step数
num_training_steps = len(train_data_loader) * epochs

# Adam优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=2e-5,
    parameters=model.parameters())
# 交叉熵损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()
# accuracy评价指标
metric = paddle.metric.Accuracy()
```

## 2.训练&&评估


```python
# 开启训练
global_step = 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        # 喂数据给model
        logits = model(input_ids, token_type_ids)
        # 计算损失函数值
        loss = criterion(logits, labels)
        # 预测分类概率值
        probs = F.softmax(logits, axis=1)
        # 计算acc
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            save_dir = os.path.join(ckpt_dir, "model_%d" % global_step)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 评估当前训练的模型
            # evaluate(model, criterion, metric, dev_data_loader)
            # 保存当前模型参数等
            model.save_pretrained(save_dir)
            # 保存tokenizer的词表等
            tokenizer.save_pretrained(save_dir)
```


```python
import numpy as np
import paddle

# 处理测试集数据
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    is_test=True)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack() # qid
): [data for data in fn(samples)]
test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```


```python
# 根据实际运行情况，更换加载的参数路径
params_path = 'nlpcc14/model_3300/model_state.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
```

    Loaded parameters from nlpcc14/model_3300/model_state.pdparams


## 3.预测


```python
label_map = {0: '0', 1: '1'}
results = []
# 切换model模型为评估模式，关闭dropout等随机因素
model.eval()
for batch in test_data_loader:
    input_ids, token_type_ids, qids = batch
    # 喂数据给模型
    logits = model(input_ids, token_type_ids)
    # 预测分类
    probs = F.softmax(logits, axis=-1)
    idx = paddle.argmax(probs, axis=1).numpy()
    idx = idx.tolist()
    labels = [label_map[i] for i in idx]
    qids = qids.numpy().tolist()
    results.extend(zip(qids, labels))
```


```python
res_dir = "./results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# 写入预测结果
with open(os.path.join(res_dir, "NLPCC14-SC.tsv"), 'w', encoding="utf8") as f:
    f.write("index\tprediction\n")
    for qid, label in results:
        f.write(str(qid[0])+"\t"+label+"\n")
```


```python
!zip -r results.zip results
```

# 四、SE-ABSA16_CAME数据集
SE-ABSA16_phs内置了，SE-ABSA16_CAME没有内置？

## 1.数据查看


```python
!head -n5 SE-ABSA16_CAME/train.tsv
```

    label	text_a	text_b
    0	camera#design_features	千呼万唤始出来，尼康的APSC小相机终于发布了，COOLPIX A. 你怎么看呢？我看，尼康是挤牙膏挤惯了啊，1，外观既没有V1时尚，也没P7100专业，反而类似P系列。2，CMOS炒冷饭。3，OVF没有任何提示和显示。（除了框框)4，28MM镜头是不错，可是F2.8定焦也太小气了。5，电池坑爹，用D800和V1的电池很难吗？6，考虑到1100美元的定价，富士X100S表示很欢乐。***好处是，可以确定，尼康会继续大力发展1系列了***另外体积比X100S小也算是A的优势吧***。等2014年年中跌倒1900左右的时候就可以入手了。
    0	camera#operation_performance	千呼万唤始出来，尼康的APSC小相机终于发布了，COOLPIX A. 你怎么看呢？我看，尼康是挤牙膏挤惯了啊，1，外观既没有V1时尚，也没P7100专业，反而类似P系列。2，CMOS炒冷饭。3，OVF没有任何提示和显示。（除了框框)4，28MM镜头是不错，可是F2.8定焦也太小气了。5，电池坑爹，用D800和V1的电池很难吗？6，考虑到1100美元的定价，富士X100S表示很欢乐。***好处是，可以确定，尼康会继续大力发展1系列了***另外体积比X100S小也算是A的优势吧***。等2014年年中跌倒1900左右的时候就可以入手了。
    0	hardware#usability	千呼万唤始出来，尼康的APSC小相机终于发布了，COOLPIX A. 你怎么看呢？我看，尼康是挤牙膏挤惯了啊，1，外观既没有V1时尚，也没P7100专业，反而类似P系列。2，CMOS炒冷饭。3，OVF没有任何提示和显示。（除了框框)4，28MM镜头是不错，可是F2.8定焦也太小气了。5，电池坑爹，用D800和V1的电池很难吗？6，考虑到1100美元的定价，富士X100S表示很欢乐。***好处是，可以确定，尼康会继续大力发展1系列了***另外体积比X100S小也算是A的优势吧***。等2014年年中跌倒1900左右的时候就可以入手了。
    0	software#design_features	千呼万唤始出来，尼康的APSC小相机终于发布了，COOLPIX A. 你怎么看呢？我看，尼康是挤牙膏挤惯了啊，1，外观既没有V1时尚，也没P7100专业，反而类似P系列。2，CMOS炒冷饭。3，OVF没有任何提示和显示。（除了框框)4，28MM镜头是不错，可是F2.8定焦也太小气了。5，电池坑爹，用D800和V1的电池很难吗？6，考虑到1100美元的定价，富士X100S表示很欢乐。***好处是，可以确定，尼康会继续大力发展1系列了***另外体积比X100S小也算是A的优势吧***。等2014年年中跌倒1900左右的时候就可以入手了。



```python
!head -n5 SE-ABSA16_CAME/test.tsv
```

    qid	text_a	text_b
    0	camera#quality	一直潜水，昨天入d300s +35 1.8g，谈谈感受，dx说，标题一定要长！在我们这尼康一个代理商开的大型体验中心提的货，老板和销售mm都很热情，不欺诈，也没有店大欺客，mm很热情，从d300s到d800，d7000，到d3x配各种镜头，全部把玩了一番，感叹啊，真他妈好东西！尤其d3x，有钱了，一定要他妈买一个，还有，就是d800，一摸心中的神机，顿时凉了半截，可能摸她之前，摸了她们的头牌，d3x的缘故，这手感 真是差了点，样子嘛，之所以喜欢尼康，就是喜欢棱角分明的感觉，d3x方方正正 ，甚是讨喜，d800这丫头，变得圆滑了不少，不喜欢。都说电子产品，买新不买旧，我倒不认为这么看，中低端产品的确如此，但顶级的高端产品，真不是这么回事啊，d3x也是51点对焦，我的d300s也是51点，但明显感觉，对焦就是比d300s 快，准，暗部反差较小时，也很少拉风箱，我的d300s就不行，光线不好反差较小，拉回来拉过去，半天合不上焦，说真的，一分价钱一分货啊，d800电子性能 肯定是先进的，但机械性能 跟d3x还是没可比性，传感器固然先进，但三千多万 像素 和两千多万像素 对我们来说，真的差别这么大吗？d800e3万多，有这钱真的不如加点买 d3x啊，真要是d3x烂，为什么尼康不停产了？人说高像素 是给商业摄影师用，我们的音乐老师，是业余的音乐制作人，也拍摄一些商业广告，平时他玩的时候 都是数码什么的，nc 加起来十几个，大三元全都配齐，但干活的时候，还是120的机器，照他那话说，数码 像素太低，不够用啊！废话说太多了，谈谈感受吧，当初一直在纠结d7000和d300s，都说什么d7000画质超越d300s，我也信，但昨天拿到实机后，我瞬间就决定 d300s了，我的手算小的，握住d300s，我感觉，刚刚好，而且手柄凹槽 ，我觉得还不够深，握感不是十分的充盈，这点要像宾得k5学习，而且d7000小了一点，背部操作空间局促，大拇指没地放，果断d300s，而且试机的时候，我给d300s 换上了24-70，可能我练健身比较久了，没感觉有啥重量，蛮趁手的，现在配35 1.8 感觉轻飘飘的，哈哈，
    1	focus#operation_performance	一直潜水，昨天入d300s +35 1.8g，谈谈感受，dx说，标题一定要长！在我们这尼康一个代理商开的大型体验中心提的货，老板和销售mm都很热情，不欺诈，也没有店大欺客，mm很热情，从d300s到d800，d7000，到d3x配各种镜头，全部把玩了一番，感叹啊，真他妈好东西！尤其d3x，有钱了，一定要他妈买一个，还有，就是d800，一摸心中的神机，顿时凉了半截，可能摸她之前，摸了她们的头牌，d3x的缘故，这手感 真是差了点，样子嘛，之所以喜欢尼康，就是喜欢棱角分明的感觉，d3x方方正正 ，甚是讨喜，d800这丫头，变得圆滑了不少，不喜欢。都说电子产品，买新不买旧，我倒不认为这么看，中低端产品的确如此，但顶级的高端产品，真不是这么回事啊，d3x也是51点对焦，我的d300s也是51点，但明显感觉，对焦就是比d300s 快，准，暗部反差较小时，也很少拉风箱，我的d300s就不行，光线不好反差较小，拉回来拉过去，半天合不上焦，说真的，一分价钱一分货啊，d800电子性能 肯定是先进的，但机械性能 跟d3x还是没可比性，传感器固然先进，但三千多万 像素 和两千多万像素 对我们来说，真的差别这么大吗？d800e3万多，有这钱真的不如加点买 d3x啊，真要是d3x烂，为什么尼康不停产了？人说高像素 是给商业摄影师用，我们的音乐老师，是业余的音乐制作人，也拍摄一些商业广告，平时他玩的时候 都是数码什么的，nc 加起来十几个，大三元全都配齐，但干活的时候，还是120的机器，照他那话说，数码 像素太低，不够用啊！废话说太多了，谈谈感受吧，当初一直在纠结d7000和d300s，都说什么d7000画质超越d300s，我也信，但昨天拿到实机后，我瞬间就决定 d300s了，我的手算小的，握住d300s，我感觉，刚刚好，而且手柄凹槽 ，我觉得还不够深，握感不是十分的充盈，这点要像宾得k5学习，而且d7000小了一点，背部操作空间局促，大拇指没地放，果断d300s，而且试机的时候，我给d300s 换上了24-70，可能我练健身比较久了，没感觉有啥重量，蛮趁手的，现在配35 1.8 感觉轻飘飘的，哈哈，
    2	camera#quality	一直潜水，昨天入d300s +35 1.8g，谈谈感受，dx说，标题一定要长！在我们这尼康一个代理商开的大型体验中心提的货，老板和销售mm都很热情，不欺诈，也没有店大欺客，mm很热情，从d300s到d800，d7000，到d3x配各种镜头，全部把玩了一番，感叹啊，真他妈好东西！尤其d3x，有钱了，一定要他妈买一个，还有，就是d800，一摸心中的神机，顿时凉了半截，可能摸她之前，摸了她们的头牌，d3x的缘故，这手感 真是差了点，样子嘛，之所以喜欢尼康，就是喜欢棱角分明的感觉，d3x方方正正 ，甚是讨喜，d800这丫头，变得圆滑了不少，不喜欢。都说电子产品，买新不买旧，我倒不认为这么看，中低端产品的确如此，但顶级的高端产品，真不是这么回事啊，d3x也是51点对焦，我的d300s也是51点，但明显感觉，对焦就是比d300s 快，准，暗部反差较小时，也很少拉风箱，我的d300s就不行，光线不好反差较小，拉回来拉过去，半天合不上焦，说真的，一分价钱一分货啊，d800电子性能 肯定是先进的，但机械性能 跟d3x还是没可比性，传感器固然先进，但三千多万 像素 和两千多万像素 对我们来说，真的差别这么大吗？d800e3万多，有这钱真的不如加点买 d3x啊，真要是d3x烂，为什么尼康不停产了？人说高像素 是给商业摄影师用，我们的音乐老师，是业余的音乐制作人，也拍摄一些商业广告，平时他玩的时候 都是数码什么的，nc 加起来十几个，大三元全都配齐，但干活的时候，还是120的机器，照他那话说，数码 像素太低，不够用啊！废话说太多了，谈谈感受吧，当初一直在纠结d7000和d300s，都说什么d7000画质超越d300s，我也信，但昨天拿到实机后，我瞬间就决定 d300s了，我的手算小的，握住d300s，我感觉，刚刚好，而且手柄凹槽 ，我觉得还不够深，握感不是十分的充盈，这点要像宾得k5学习，而且d7000小了一点，背部操作空间局促，大拇指没地放，果断d300s，而且试机的时候，我给d300s 换上了24-70，可能我练健身比较久了，没感觉有啥重量，蛮趁手的，现在配35 1.8 感觉轻飘飘的，哈哈，
    3	camera#quality	一直潜水，昨天入d300s +35 1.8g，谈谈感受，dx说，标题一定要长！在我们这尼康一个代理商开的大型体验中心提的货，老板和销售mm都很热情，不欺诈，也没有店大欺客，mm很热情，从d300s到d800，d7000，到d3x配各种镜头，全部把玩了一番，感叹啊，真他妈好东西！尤其d3x，有钱了，一定要他妈买一个，还有，就是d800，一摸心中的神机，顿时凉了半截，可能摸她之前，摸了她们的头牌，d3x的缘故，这手感 真是差了点，样子嘛，之所以喜欢尼康，就是喜欢棱角分明的感觉，d3x方方正正 ，甚是讨喜，d800这丫头，变得圆滑了不少，不喜欢。都说电子产品，买新不买旧，我倒不认为这么看，中低端产品的确如此，但顶级的高端产品，真不是这么回事啊，d3x也是51点对焦，我的d300s也是51点，但明显感觉，对焦就是比d300s 快，准，暗部反差较小时，也很少拉风箱，我的d300s就不行，光线不好反差较小，拉回来拉过去，半天合不上焦，说真的，一分价钱一分货啊，d800电子性能 肯定是先进的，但机械性能 跟d3x还是没可比性，传感器固然先进，但三千多万 像素 和两千多万像素 对我们来说，真的差别这么大吗？d800e3万多，有这钱真的不如加点买 d3x啊，真要是d3x烂，为什么尼康不停产了？人说高像素 是给商业摄影师用，我们的音乐老师，是业余的音乐制作人，也拍摄一些商业广告，平时他玩的时候 都是数码什么的，nc 加起来十几个，大三元全都配齐，但干活的时候，还是120的机器，照他那话说，数码 像素太低，不够用啊！废话说太多了，谈谈感受吧，当初一直在纠结d7000和d300s，都说什么d7000画质超越d300s，我也信，但昨天拿到实机后，我瞬间就决定 d300s了，我的手算小的，握住d300s，我感觉，刚刚好，而且手柄凹槽 ，我觉得还不够深，握感不是十分的充盈，这点要像宾得k5学习，而且d7000小了一点，背部操作空间局促，大拇指没地放，果断d300s，而且试机的时候，我给d300s 换上了24-70，可能我练健身比较久了，没感觉有啥重量，蛮趁手的，现在配35 1.8 感觉轻飘飘的，哈哈，



```python
from paddlenlp.datasets import load_dataset
train_ds, test_ds  = load_dataset("seabsa16","phns", data_files={"train": "SE-ABSA16_CAME/train.tsv", "test": "SE-ABSA16_CAME/test.tsv"})
dev_ds  = load_dataset("seabsa16", "phns", data_files={ "test": "SE-ABSA16_CAME/test.tsv"})

print(train_ds[0])
print(train_ds[1])
print(train_ds[2])
```

    {'text': 'camera#design_features', 'text_pair': '千呼万唤始出来，尼康的APSC小相机终于发布了，COOLPIX A. 你怎么看呢？我看，尼康是挤牙膏挤惯了啊，1，外观既没有V1时尚，也没P7100专业，反而类似P系列。2，CMOS炒冷饭。3，OVF没有任何提示和显示。（除了框框)4，28MM镜头是不错，可是F2.8定焦也太小气了。5，电池坑爹，用D800和V1的电池很难吗？6，考虑到1100美元的定价，富士X100S表示很欢乐。***好处是，可以确定，尼康会继续大力发展1系列了***另外体积比X100S小也算是A的优势吧***。等2014年年中跌倒1900左右的时候就可以入手了。', 'label': 0}
    {'text': 'camera#operation_performance', 'text_pair': '千呼万唤始出来，尼康的APSC小相机终于发布了，COOLPIX A. 你怎么看呢？我看，尼康是挤牙膏挤惯了啊，1，外观既没有V1时尚，也没P7100专业，反而类似P系列。2，CMOS炒冷饭。3，OVF没有任何提示和显示。（除了框框)4，28MM镜头是不错，可是F2.8定焦也太小气了。5，电池坑爹，用D800和V1的电池很难吗？6，考虑到1100美元的定价，富士X100S表示很欢乐。***好处是，可以确定，尼康会继续大力发展1系列了***另外体积比X100S小也算是A的优势吧***。等2014年年中跌倒1900左右的时候就可以入手了。', 'label': 0}
    {'text': 'hardware#usability', 'text_pair': '千呼万唤始出来，尼康的APSC小相机终于发布了，COOLPIX A. 你怎么看呢？我看，尼康是挤牙膏挤惯了啊，1，外观既没有V1时尚，也没P7100专业，反而类似P系列。2，CMOS炒冷饭。3，OVF没有任何提示和显示。（除了框框)4，28MM镜头是不错，可是F2.8定焦也太小气了。5，电池坑爹，用D800和V1的电池很难吗？6，考虑到1100美元的定价，富士X100S表示很欢乐。***好处是，可以确定，尼康会继续大力发展1系列了***另外体积比X100S小也算是A的优势吧***。等2014年年中跌倒1900左右的时候就可以入手了。', 'label': 0}


## 2.skep模型加载


```python
# 指定模型名称一键加载模型
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

model = SkepForSequenceClassification.from_pretrained(
    'skep_ernie_1.0_large_ch', num_classes=len(train_ds.label_list))
# 指定模型名称一键加载tokenizer
tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')
```

    [2021-06-19 08:51:17,209] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    [2021-06-19 08:51:30,437] [    INFO] - Found /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.vocab.txt


## 3.数据处理


```python
from functools import partial
import os
import time
import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False,
                    dataset_name="chnsenticorp"):
   
    encoded_inputs = tokenizer(
        text=example["text"],
        text_pair=example["text_pair"],
        max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids
```


```python
from utils import create_dataloader
# 处理的最大文本序列长度
max_seq_length=256
# 批量数据大小
batch_size=40

# 将数据处理成model可读入的数据格式
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)
# 将数据组成批量式数据，如
# 将不同长度的文本序列padding到批量式数据中最大长度
# 将每条数据label堆叠在一起
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(dtype="int64")  # labels
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

## 4.模型训练


```python
# 训练轮次
epochs = 30
# 总共需要训练的step数
num_training_steps = len(train_data_loader) * epochs
# 优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=5e-5,
    parameters=model.parameters())
# 交叉熵损失
criterion = paddle.nn.loss.CrossEntropyLoss()
# Accuracy评价指标
metric = paddle.metric.Accuracy()
```


```python
# 开启训练
ckpt_dir = "SE-ABSA16_CAME"
global_step = 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        # 喂数据给model
        logits = model(input_ids, token_type_ids)
        # 计算损失函数值
        loss = criterion(logits, labels)
        # 预测分类概率
        probs = F.softmax(logits, axis=1)
        # 计算acc
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:

            save_dir = os.path.join(ckpt_dir, "model_%d" % global_step)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 保存模型参数
            model.save_pretrained(save_dir)
            # 保存tokenizer的词表等
            tokenizer.save_pretrained(save_dir)
```

    global step 10, epoch: 1, batch: 10, loss: 0.56722, acc: 0.60750, speed: 0.52 step/s
    global step 20, epoch: 1, batch: 20, loss: 0.70404, acc: 0.61875, speed: 0.52 step/s
    global step 30, epoch: 1, batch: 30, loss: 0.64085, acc: 0.63750, speed: 0.53 step/s
    global step 40, epoch: 2, batch: 7, loss: 0.58592, acc: 0.64872, speed: 0.51 step/s
    global step 50, epoch: 2, batch: 17, loss: 0.68638, acc: 0.65849, speed: 0.51 step/s
    global step 60, epoch: 2, batch: 27, loss: 0.63500, acc: 0.66416, speed: 0.51 step/s
    global step 70, epoch: 3, batch: 4, loss: 0.57357, acc: 0.66929, speed: 0.52 step/s
    global step 80, epoch: 3, batch: 14, loss: 0.59466, acc: 0.67439, speed: 0.52 step/s
    global step 90, epoch: 3, batch: 24, loss: 0.55156, acc: 0.67474, speed: 0.52 step/s
    global step 100, epoch: 4, batch: 1, loss: 0.60782, acc: 0.67201, speed: 0.52 step/s
    global step 110, epoch: 4, batch: 11, loss: 0.64200, acc: 0.67001, speed: 0.28 step/s
    global step 120, epoch: 4, batch: 21, loss: 0.71011, acc: 0.66938, speed: 0.52 step/s
    global step 130, epoch: 4, batch: 31, loss: 0.62669, acc: 0.67290, speed: 0.53 step/s
    global step 140, epoch: 5, batch: 8, loss: 0.59335, acc: 0.67090, speed: 0.50 step/s
    global step 150, epoch: 5, batch: 18, loss: 0.55296, acc: 0.67685, speed: 0.52 step/s
    global step 160, epoch: 5, batch: 28, loss: 0.58558, acc: 0.67862, speed: 0.52 step/s
    global step 170, epoch: 6, batch: 5, loss: 0.52604, acc: 0.67929, speed: 0.51 step/s
    global step 180, epoch: 6, batch: 15, loss: 0.75630, acc: 0.67905, speed: 0.51 step/s
    global step 190, epoch: 6, batch: 25, loss: 0.58693, acc: 0.67792, speed: 0.52 step/s
    global step 200, epoch: 7, batch: 2, loss: 0.57527, acc: 0.67590, speed: 0.52 step/s
    global step 210, epoch: 7, batch: 12, loss: 0.56864, acc: 0.67705, speed: 0.28 step/s
    global step 220, epoch: 7, batch: 22, loss: 0.55729, acc: 0.67934, speed: 0.52 step/s
    global step 230, epoch: 7, batch: 32, loss: 0.57403, acc: 0.68166, speed: 0.54 step/s
    global step 240, epoch: 8, batch: 9, loss: 0.57195, acc: 0.68316, speed: 0.51 step/s
    global step 250, epoch: 8, batch: 19, loss: 0.50223, acc: 0.68404, speed: 0.51 step/s
    global step 260, epoch: 8, batch: 29, loss: 0.62531, acc: 0.68494, speed: 0.52 step/s
    global step 270, epoch: 9, batch: 6, loss: 0.61489, acc: 0.68690, speed: 0.52 step/s
    global step 280, epoch: 9, batch: 16, loss: 0.67492, acc: 0.68853, speed: 0.52 step/s
    global step 290, epoch: 9, batch: 26, loss: 0.60046, acc: 0.68927, speed: 0.51 step/s
    global step 300, epoch: 10, batch: 3, loss: 0.54265, acc: 0.69156, speed: 0.52 step/s
    global step 310, epoch: 10, batch: 13, loss: 0.49282, acc: 0.69248, speed: 0.29 step/s
    global step 320, epoch: 10, batch: 23, loss: 0.71478, acc: 0.69412, speed: 0.52 step/s
    global step 330, epoch: 10, batch: 33, loss: 0.69863, acc: 0.69461, speed: 0.54 step/s
    global step 340, epoch: 11, batch: 10, loss: 0.59941, acc: 0.69595, speed: 0.50 step/s
    global step 350, epoch: 11, batch: 20, loss: 0.68566, acc: 0.69571, speed: 0.52 step/s
    global step 360, epoch: 11, batch: 30, loss: 0.50205, acc: 0.69701, speed: 0.53 step/s
    global step 370, epoch: 12, batch: 7, loss: 0.58069, acc: 0.69757, speed: 0.51 step/s
    global step 380, epoch: 12, batch: 17, loss: 0.61234, acc: 0.69895, speed: 0.52 step/s
    global step 390, epoch: 12, batch: 27, loss: 0.58433, acc: 0.69988, speed: 0.52 step/s
    global step 400, epoch: 13, batch: 4, loss: 0.38205, acc: 0.70152, speed: 0.52 step/s
    global step 410, epoch: 13, batch: 14, loss: 0.44822, acc: 0.70178, speed: 0.28 step/s
    global step 420, epoch: 13, batch: 24, loss: 0.52814, acc: 0.70335, speed: 0.52 step/s
    global step 430, epoch: 14, batch: 1, loss: 0.42370, acc: 0.70421, speed: 0.52 step/s
    global step 440, epoch: 14, batch: 11, loss: 0.37624, acc: 0.70571, speed: 0.52 step/s
    global step 450, epoch: 14, batch: 21, loss: 0.55443, acc: 0.70686, speed: 0.52 step/s
    global step 460, epoch: 14, batch: 31, loss: 0.49042, acc: 0.70770, speed: 0.53 step/s
    global step 470, epoch: 15, batch: 8, loss: 0.45924, acc: 0.70844, speed: 0.50 step/s
    global step 480, epoch: 15, batch: 18, loss: 0.39627, acc: 0.70942, speed: 0.52 step/s
    global step 490, epoch: 15, batch: 28, loss: 0.42499, acc: 0.71101, speed: 0.52 step/s
    global step 500, epoch: 16, batch: 5, loss: 0.31916, acc: 0.71245, speed: 0.52 step/s
    global step 510, epoch: 16, batch: 15, loss: 0.57443, acc: 0.71309, speed: 0.29 step/s
    global step 520, epoch: 16, batch: 25, loss: 0.56754, acc: 0.71380, speed: 0.52 step/s
    global step 530, epoch: 17, batch: 2, loss: 0.50241, acc: 0.71289, speed: 0.55 step/s
    global step 540, epoch: 17, batch: 12, loss: 0.50492, acc: 0.71246, speed: 0.52 step/s
    global step 550, epoch: 17, batch: 22, loss: 0.37710, acc: 0.71328, speed: 0.52 step/s
    global step 560, epoch: 17, batch: 32, loss: 0.42509, acc: 0.71434, speed: 0.54 step/s
    global step 570, epoch: 18, batch: 9, loss: 0.41145, acc: 0.71572, speed: 0.51 step/s
    global step 580, epoch: 18, batch: 19, loss: 0.45151, acc: 0.71670, speed: 0.51 step/s
    global step 590, epoch: 18, batch: 29, loss: 0.43464, acc: 0.71714, speed: 0.53 step/s
    global step 600, epoch: 19, batch: 6, loss: 0.50560, acc: 0.71807, speed: 0.52 step/s
    global step 610, epoch: 19, batch: 16, loss: 0.41544, acc: 0.71917, speed: 0.27 step/s
    global step 620, epoch: 19, batch: 26, loss: 0.57100, acc: 0.71951, speed: 0.56 step/s
    global step 630, epoch: 20, batch: 3, loss: 0.28381, acc: 0.72052, speed: 0.52 step/s
    global step 640, epoch: 20, batch: 13, loss: 0.31514, acc: 0.72141, speed: 0.52 step/s
    global step 650, epoch: 20, batch: 23, loss: 0.55895, acc: 0.72201, speed: 0.52 step/s
    global step 660, epoch: 20, batch: 33, loss: 0.49381, acc: 0.72285, speed: 0.55 step/s
    global step 670, epoch: 21, batch: 10, loss: 0.33467, acc: 0.72435, speed: 0.50 step/s
    global step 680, epoch: 21, batch: 20, loss: 0.39927, acc: 0.72513, speed: 0.52 step/s
    global step 690, epoch: 21, batch: 30, loss: 0.45863, acc: 0.72603, speed: 0.53 step/s
    global step 700, epoch: 22, batch: 7, loss: 0.35752, acc: 0.72724, speed: 0.51 step/s
    global step 710, epoch: 22, batch: 17, loss: 0.38762, acc: 0.72824, speed: 0.29 step/s
    global step 720, epoch: 22, batch: 27, loss: 0.32564, acc: 0.72871, speed: 0.52 step/s
    global step 730, epoch: 23, batch: 4, loss: 0.35076, acc: 0.72935, speed: 0.52 step/s
    global step 740, epoch: 23, batch: 14, loss: 0.36434, acc: 0.73051, speed: 0.52 step/s
    global step 750, epoch: 23, batch: 24, loss: 0.34934, acc: 0.73131, speed: 0.52 step/s
    global step 760, epoch: 24, batch: 1, loss: 0.40433, acc: 0.73189, speed: 0.52 step/s
    global step 770, epoch: 24, batch: 11, loss: 0.54736, acc: 0.73320, speed: 0.52 step/s
    global step 780, epoch: 24, batch: 21, loss: 0.37738, acc: 0.73428, speed: 0.52 step/s
    global step 790, epoch: 24, batch: 31, loss: 0.51316, acc: 0.73461, speed: 0.53 step/s
    global step 800, epoch: 25, batch: 8, loss: 0.43568, acc: 0.73597, speed: 0.51 step/s
    global step 810, epoch: 25, batch: 18, loss: 0.44679, acc: 0.73698, speed: 0.29 step/s
    global step 820, epoch: 25, batch: 28, loss: 0.39317, acc: 0.73759, speed: 0.52 step/s
    global step 830, epoch: 26, batch: 5, loss: 0.38251, acc: 0.73793, speed: 0.52 step/s
    global step 840, epoch: 26, batch: 15, loss: 0.44355, acc: 0.73864, speed: 0.52 step/s
    global step 850, epoch: 26, batch: 25, loss: 0.25384, acc: 0.73960, speed: 0.52 step/s
    global step 860, epoch: 27, batch: 2, loss: 0.31328, acc: 0.73996, speed: 0.53 step/s
    global step 870, epoch: 27, batch: 12, loss: 0.33673, acc: 0.74071, speed: 0.52 step/s
    global step 880, epoch: 27, batch: 22, loss: 0.37009, acc: 0.74122, speed: 0.52 step/s
    global step 890, epoch: 27, batch: 32, loss: 0.59494, acc: 0.74174, speed: 0.54 step/s
    global step 900, epoch: 28, batch: 9, loss: 0.44762, acc: 0.74250, speed: 0.53 step/s
    global step 910, epoch: 28, batch: 19, loss: 0.31324, acc: 0.74330, speed: 0.29 step/s
    global step 920, epoch: 28, batch: 29, loss: 0.36967, acc: 0.74411, speed: 0.52 step/s
    global step 930, epoch: 29, batch: 6, loss: 0.36852, acc: 0.74480, speed: 0.52 step/s
    global step 940, epoch: 29, batch: 16, loss: 0.32877, acc: 0.74587, speed: 0.51 step/s
    global step 950, epoch: 29, batch: 26, loss: 0.37547, acc: 0.74678, speed: 0.52 step/s
    global step 960, epoch: 30, batch: 3, loss: 0.44908, acc: 0.74719, speed: 0.52 step/s
    global step 970, epoch: 30, batch: 13, loss: 0.45284, acc: 0.74812, speed: 0.52 step/s
    global step 980, epoch: 30, batch: 23, loss: 0.29721, acc: 0.74893, speed: 0.52 step/s
    global step 990, epoch: 30, batch: 33, loss: 0.53376, acc: 0.74963, speed: 0.54 step/s


## 5.预测提交结果


```python
@paddle.no_grad()
def predict(model, data_loader, label_map):
    """
    Given a prediction dataset, it gives the prediction results.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
    """
    model.eval()
    results = []
    for batch in data_loader:
        input_ids, token_type_ids = batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results
```


```python
# 处理测试集数据
label_map = {0: '0', 1: '1'}
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    is_test=True)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
): [data for data in fn(samples)]
test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```


```python
# 根据实际运行情况，更换加载的参数路径
params_path = 'SE-ABSA16_CAME/model_900/model_state.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)

results = predict(model, test_data_loader, label_map)
```

    Loaded parameters from SE-ABSA16_CAME/model_900/model_state.pdparams



```python
# 写入预测结果
with open(os.path.join("results", "SE-ABSA16_CAME.tsv"), 'w', encoding="utf8") as f:
    f.write("index\tprediction\n")
    for idx, label in enumerate(results):
        f.write(str(idx)+"\t"+label+"\n")
```
