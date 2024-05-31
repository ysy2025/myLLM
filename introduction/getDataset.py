from datasets import load_dataset

if __name__ == '__main__':
    raw_datasets = load_dataset("glue", "mrpc")
    print(raw_datasets)

    # train data
    raw_train_dataset = raw_datasets["train"]
    print(raw_train_dataset[0])
    # 查看label ID 对应的标签名称
    print(raw_train_dataset.features)
    # 可以看到 label 字段是 ClassLabel 类型