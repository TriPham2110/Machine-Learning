if __name__ == '__main__':
    from pandas import read_csv
    import pandas as pd
    import numpy as np
    import string, os
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    from nltk.tokenize import word_tokenize

    dataset = read_csv("data/Shakespeare_data.csv")
    dataset.dropna(axis="columns", how="any", inplace=True)

    all_lines = [_ for _ in dataset['PlayerLine']]


    def clean_text(text):
        text = "".join(_ for _ in text if _ not in string.punctuation).lower()
        text = text.encode("utf8").decode("ascii", 'ignore')
        return text


    corpus = [clean_text(x) for x in all_lines]

    tokenized_sents = [word_tokenize(i) for i in corpus]

    print("hello")