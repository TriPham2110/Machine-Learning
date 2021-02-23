if __name__ == '__main__':
    import numpy as np
    from pandas import read_csv
    from matplotlib import pyplot as plt

    # Load dataset
    url = "data/breast_cancer_wisconsin_diagnostic_data.csv"
    dataset = read_csv(url)
    dataset.dropna(axis="columns", how="any", inplace=True)

    # array = dataset.values
    # x = array[:, :-1]
    # y = array[:, -1]
