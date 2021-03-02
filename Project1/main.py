if __name__ == '__main__':
    from pandas import read_csv
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
    bins = len(dataset2['quality'].unique()) - 1
    lower_bounds, histogram_vals = hist.make_histogram(bins)

    hist.plot_histogram(lower_bounds, histogram_vals, 'quality', 'samples', 'Histogram of wine quality')
    hist.plot_distribution(bins, x_label='quality')

    # Breast cancer dataset clustered with two attributes texture mean and radius mean
    X1 = dataset1[['texture_mean', 'radius_mean']]

    model1 = gmm.gmm(n_clusters=2, n_iterations=1)

    normalized_X1 = model1.normalize(X1)

    model1.train(normalized_X1)
    model1.plot_contours(normalized_X1, model1.mu, model1.sigma, x_label=X1.columns.values[0], y_label=X1.columns.values[1],
                        title='Initial clusters (breast cancer dataset)')

    model1 = gmm.gmm(n_clusters=2, n_iterations=50, seed=4)
    model1.train(normalized_X1)
    model1.plot_contours(normalized_X1, model1.mu, model1.sigma, x_label=X1.columns.values[0], y_label=X1.columns.values[1],
                        title='Final clusters (breast cancer dataset)')

    model1.plot_distribution(normalized_X1, model1.mu, model1.sigma, x_label=X1.columns.values[0],
                            y_label=X1.columns.values[1],
                            title='GMM distribution for final clusters (breast cancer dataset)')

    # Red wine quality clustered with two attributes fixed acidity and alcohol
    X2 = dataset2[['fixed acidity', 'alcohol']]

    model2 = gmm.gmm(n_clusters=3, n_iterations=1)

    normalized_X2 = model2.normalize(X2)

    model2.train(normalized_X2)
    model2.plot_contours(normalized_X2, model2.mu, model2.sigma, x_label=X2.columns.values[0],
                         y_label=X2.columns.values[1], title='Initial clusters (red wine dataset)')

    model2 = gmm.gmm(n_clusters=3, n_iterations=50, seed=4)
    model2.train(normalized_X2)
    model2.plot_contours(normalized_X2, model2.mu, model2.sigma, x_label=X2.columns.values[0],
                         y_label=X2.columns.values[1], title='Final clusters (red wine dataset)')

    # model2.plot_distribution(normalized_X2, model2.mu, model2.sigma, x_label=X2.columns.values[0],
    #                          y_label=X2.columns.values[1],
    #                          title='GMM distribution for final clusters (red wine dataset)')
