def feature_support_combined(X: pd.DataFrame, y: pd.Series, num_feats: int,
                             classification: bool = True) -> pd.DataFrame:
    """Kombination der Unterstützung von Features im Zusammenhang mit mehreren :ref:`Feature Selection` Methoden.

    :param X: Datensatz
    :type X: pd.DataFrame
    :param y: Labels
    :type y: pd.Series
    :param num_feats: Anzahl der Top-Features
    :type num_feats: int
    :param classification: Indikator für Klassifikationsproblem
    :type classification: bool
    :return: DataFrame mit den Ergebnissen
    :rtype: pd.DataFrame
    """
    cor_support = feature_support_by_correlation(X, y)
    chi2_support = feature_support_by_chi2(X, y, num_feats, classification)
    rfe_support = feature_support_by_rfe(X, y, classification)
    lasso_support = feature_support_by_lasso(X, y, num_feats, classification)
    rf_support = feature_support_by_random_forest(X, y, num_feats, classification)

    features = pd.DataFrame({
        'name':  X.columns.tolist(),
        'correlation': cor_support,
        'chi2': chi2_support,
        'rfe': rfe_support,
        'lasso': lasso_support,
        'random_forest': rf_support
    })

    features['support'] = np.sum(features, axis=1)
    features = features.sort_values(['support', 'name'], ascending=False)
    features.index = range(1, len(features) + 1)

    return features


def drop_features_by_support(
        X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, classification: bool,
        top_feature_count: int, top_feature_support_min: int, experiment_report_path: str)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Features anhand der Unterstützung aus Trainings- und Testdaten entfernen.

    :param X_train: Trainingsdaten
    :type X_train: pd.DataFrame
    :param X_test: Testdaten
    :type X_test: pd.DataFrame
    :param y_train: Trainingslabels
    :type y_train: pd.Series
    :param classification: Indikator für Klassifikationsproblem
    :type classification: bool
    :param top_feature_count: Anzahl der Top-Features für Selektionsalgorithmen
    :type top_feature_count: int
    :param top_feature_support_min: minimaler Feature Support
    :type top_feature_support_min: float
    :param experiment_report_path: Pfad für Reporting
    :type experiment_report_path: str
    :return: Trainings- und Testdaten mit selektierten Features
    :rtype: tuple
    """
    feature_support = feature_support_combined(X_train, y_train, top_feature_count, classification=classification)
    feature_support.to_csv(os.path.join(experiment_report_path, 'feature_support.csv'))

    scatter_data = feature_support.sort_values(by='support', ascending=False).head(30)
    scatter = sns.scatterplot(data=scatter_data, x="support", y="name")
    scatter.set(xlabel="Feature Support (Importance)", ylabel="Feature")
    scatter.get_figure().savefig(os.path.join(experiment_report_path, 'feature_support.jpg'),
                                 pad_inches=0.0, bbox_inches='tight')
    plt.clf()

    features_correlated = pairwise_correlation(X_train).query('corr >= 1.0').reset_index().A.unique().tolist()
    features = feature_support.query('support >= {}'.format(top_feature_support_min)).name
    features = features[~features.isin(features_correlated)]
    return X_train[features], X_test[features]