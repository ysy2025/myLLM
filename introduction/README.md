https://zhuanlan.zhihu.com/p/660397901

NLP从0到1之HuggingFace实战：第一讲 dataset - 知乎 (zhihu.com)
0 Huggingface 代理
因为AIGC应用都需要备案的缘故，Huggingface无法直接访问。最开始需要非常折腾才能下载开源模型权重，现在huggingface 代理已经非常简单了，只需要运行python时加一个参数即可。

HF_ENDPOINT=https://hf-mirror.com python 你的python脚本
其他代理方法参考：huggingface镜像网站下载模型_huggingface资源mm_sd_v15_v2-CSDN博客
https://link.zhihu.com/?target=https%3A//blog.csdn.net/zaf0516/article/details/135926004

1 加载数据集
1.1 Huggingface 处理数据的通用用法
无论是从 Hugging face Hub上获取的数据集还是本地的数据集，均是如下用法。

Data format	        Loading script	Example
CSV & TSV	        csv	            load_dataset("csv", data_files="my_file.csv")
Text files	        text	        load_dataset("text", data_files="my_file.txt")
JSON & JSON Lines	json	        load_dataset("json", data_files="my_file.jsonl")
Pickled DataFrames	pandas	        load_dataset("pandas", data_files="my_dataframe.pkl")

1.2 来源一：从Huggingface Hub上获取 Dataset
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

# train data
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
# 查看label ID 对应的标签名称
print(raw_train_dataset.features)
# 可以看到 label 字段是 ClassLabel 类型
# 0 corresponds to not_equivalent, and 1 corresponds to equivalent.

1.3 来源二：本地的数据集
1.3.1 准备数据
# 准备数据：SQuAD-it dataset
!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
!gzip -dkv SQuAD_it-*.json.gz
# 解压缩后会出现两个文件：SQuAD_it-train.json and SQuAD_it-test.json
1.3.2 加载数据
from datasets import load_dataset

# 用法一
squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
# 可以看到: features 表示 特征列。num_rows 表示行数
print(squad_it_dataset)
print(squad_it_dataset["train"][0])


# 用法二：传入一个dict
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset)

# 除此之外，data_files 字段还可以支持：
# 1. list：多个文件路径
# 2. 通配符形式，比如 *.json
# 3. 压缩文件
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

1.4 来源三：网络上的数据集
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")


2 操作数据集：Dataset.map() 方法
这里为了演示 map 方法，特意找了一个有“缺陷”的数据集。

2.1 准备数据
!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
!unzip drugsCom_raw.zip

from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# 注意这个 delimiter 属性
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

2.2 peek一下数据集
好的习惯：先peek一下数据集，别直接上手就用，通常数据都是有一些错误的，要洗干净再用。

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
print(drug_sample[:3])
# 可以看到数据集有如下问题：
# 有一个 Unnamed: 0 列，有ID
# condition 列，大小写混合
# review 列，\r\n 换行符，包含 &\#039; HTML标签

# 还可以使用 Dataset.shuffle() and Dataset.select() 随机挑选几条数据集
# 留作课后作业 :)
# 例子：
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
drug_sample[:3]

shuffled_dataset = drug_dataset["train"].shuffle(seed=42)
shuffled_dataset["label"][:10]

small_dataset = drug_dataset["train"].select([0, 10, 20, 30, 40, 50])
len(small_dataset)



NLP从0到1之HuggingFace实战：第二讲 从头训练tokenizer - 知乎 (zhihu.com)
NLP从0到1之HuggingFace实战：第三讲 数据预处理 - 知乎 (zhihu.com)
NLP从0到1之HuggingFace实战：第四讲 创建Model - 知乎 (zhihu.com)
NLP从0到1之HuggingFace实战：第五讲 如何 fine-tuning - 知乎 (zhihu.com)
NLP从0到1之HuggingFace实战：NLP项目实战 - 知乎 (zhihu.com)
