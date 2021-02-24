if __name__ == '__main__':
    import numpy as np
    from pandas import read_csv
    from matplotlib import pyplot as plt
    from histogram import histogram

    # Load dataset
    dataset1 = read_csv("data/breast_cancer_wisconsin_diagnostic_data.csv")
    dataset1.dropna(axis="columns", how="any", inplace=True)

    dataset2 = read_csv("data/red_wine_quality.csv")
    dataset2.dropna(axis="columns", how="any", inplace=True)

    """
    Observing the target value column in the dataset
    print(dataset1['diagnosis'].unique())
    print(dataset2['quality'].unique())
    print(np.sum(dataset1['diagnosis'] == 'M'))
    print(np.sum(dataset1['diagnosis'] == 'B'))
    print(dataset2[dataset2['quality'].isin([3])])
    for i in range(min(dataset2['quality'].unique()), max(dataset2['quality'].unique())+1):
        print(np.sum(dataset2['quality'] == i))
    """
    hist = histogram(dataset2['quality'])
    lower_bounds, histogram_vals = hist.make_histogram(dataset2['quality'], 5, 3, 8)

    hist.plot_histogram(lower_bounds, histogram_vals, 'quality', 'samples', 'Histogram of wine quality')
