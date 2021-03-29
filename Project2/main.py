if __name__ == '__main__':
    from pandas import read_csv
    import pandas as pd
    import numpy as np
    import string, os

    dataset = read_csv("data/Shakespeare_data.csv")
    dataset.dropna(axis="columns", how="any", inplace=True)

    all_lines = [_ for _ in dataset['PlayerLine']]


    def clean_text(text):
        text = "".join(_ for _ in text if _ not in string.punctuation).lower()
        text = text.encode("utf8").decode("ascii", 'ignore')
        return text


    corpus = [clean_text(x) for x in all_lines]

    from hmm import hmm
    model = hmm(num_hidden_states=5, max_iter=2)
    model.train_model(corpus)

    print("hello")