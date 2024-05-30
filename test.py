from transformers import pipeline, AutoTokenizer

if __name__ == '__main__':
    # ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    # tk = AutoTokenizer.from_pretrained(ckpt)
    # model = pipeline("text-classification", model=ckpt)
    #
    # text = "This movie is really good!"
    # inputs = tk(text, return_tensors="pt")
    # print(inputs)
    # print("1"*32)

    # outputs = model(**inputs)
    # print("2" * 32)
    #
    # print(f"Input text: {text}")
    # print(f"Predicted label: {outputs[0]['label']}, score: {outputs[0]['score']:.2f}")

    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")  # 情感分析
    classifier("I've been waiting for a HuggingFace course my whole life.")