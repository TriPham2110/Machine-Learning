if __name__ == '__main__':
    import numpy as np
    from pandas import read_csv
    from matplotlib import pyplot as plt
    from histogram import histogram
    import gmm

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
    bins = len(dataset2['quality'].unique())-1
    lower_bounds, histogram_vals = hist.make_histogram(bins)

    hist.plot_histogram(lower_bounds, histogram_vals, 'quality', 'samples', 'Histogram of wine quality')
    hist.plot_pdf(bins)

    # Breast cancer dataset clustered with two attributes texture mean and radius mean
    array = dataset1.values
    X = dataset1[['texture_mean', 'radius_mean']].values

    model = gmm.gmm(n_clusters=2, n_iterations=1)
    model.train(X)
    model.plot_contours(X, model.mu, model.sigma, 'Initial clusters')

    model = gmm.gmm(n_clusters=2, n_iterations=50, seed=4)
    model.train(X)
    model.plot_contours(X, model.mu, model.sigma, 'Final clusters')