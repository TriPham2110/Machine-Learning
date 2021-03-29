from pandas import read_csv
import string
from hmm import hmm


def clean_text(text):
    text = "".join(_ for _ in text if _ not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii", 'ignore')
    return text


if __name__ == '__main__':
    dataset = read_csv("data/Shakespeare_data.csv")
    dataset.dropna(axis="columns", how="any", inplace=True)

    all_lines = [_ for _ in dataset['PlayerLine']]

    corpus = [clean_text(x) for x in all_lines]

    # For training
    # model = hmm(num_hidden_states=5, max_iter=15)
    # model.train_model(corpus, filename='train')

    transitions, emissions, pi = hmm.load('train.pickle')

    option = None

    model = hmm(num_hidden_states=5, transitions=transitions, emissions=emissions, pi=pi)

    while option != '3':
        print("Select your option:")
        print("1/ Generate text")
        print("2/ Predict text given a sequence of words")
        print("3/ Exit")
        option = input()

        if option == "1":
            print("Input your target amount of words to generate: ")
            model.generate(int(input()))
        if option == "2":
            print("Input you sequence of words: ")
            text = str(input())
            print("Input your target amount of words to predict: ")
            num = int(input())
            model.predict(text, num)
            print('\n\n')