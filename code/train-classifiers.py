def conduct_classification_experiment(experiment_name, input_file_path, config: dict):
    """Trainieren und Evaluieren von Klassifikationsmodellen.

    :param experiment_name: Name des Experiments
    :type experiment_name: str
    :param input_file_path: Pfad zum Feature-CSV Datei (z.B. *data/processed/default.csv*)
    :type input_file_path: str
    :param config: Konfiguration
    :type config: dict
    """
    logger = logging.getLogger(__name__)
    logger.info('conduct "{}" ...'.format(experiment_name))

    df = typed_view(input_file_path)

    label_column_name = config['label_column_name']

    experiment_report_path = os.path \
        .join('reports/classification', label_column_name, experiment_name)

    os.makedirs(experiment_report_path, exist_ok=True)

    # extract rows with classification unknown
    if 'label_unknown_column_name' in config and 'label_unknown_value' in config:
        df_unknown_index = df.index[df[config['label_unknown_column_name']]
                                    == config['label_unknown_value']]
        df_unknown = df.loc[df_unknown_index].copy()
        df_unknown.drop(inplace=True, columns=label_column_name)
        df_unknown.to_csv('data/processed/{}_unknown.csv'.format(label_column_name))
        df.drop(inplace=True, index=df_unknown_index)

    if 'test_set' in config:
        df_test = typed_view(config['test_set'])
        y_train = df[label_column_name].copy()
        X_train = df.copy()
        y_test = df_test[label_column_name].copy()
        X_test = df_test.copy()
        logger.info('using test set from "{}"'.format(config['test_set']))
    else:
        # Split Sets
        X_train, X_test, y_train, y_test = stratified_train_test_split(df, label_column_name)
        logger.info('using stratified train/test split')

    # Handle Labels
    drop_labels([X_train, X_test], more=(config['drop']))

    # Scale Data
    X_train = z_score(X_train)
    X_test = z_score(X_test)

    # Select Features
    logger.info('selecting features ...'.format(experiment_name))
    X_train, X_test = drop_features_by_support(
        X_train, X_test, y_train, classification=False,
        top_feature_count=config['top_feature_count'], top_feature_support_min=config['top_feature_support_min'],
        experiment_report_path=experiment_report_path)

    results = pd.DataFrame()
    for name, model in _create_models(config['classifiers']):
        logger.info('Evaluating {}'.format(name))

        try:
            result = cross_validate(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=90210), n_jobs=-1,
                scoring=['precision_weighted', 'recall_weighted', 'f1_weighted'])

            results = results.append(pd.DataFrame([[
                name,
                np.mean(result['test_f1_weighted']),
                np.mean(result['test_precision_weighted']),
                np.mean(result['test_recall_weighted'])
            ]], columns=['Model', 'F1', 'Precision', 'Recall']))

            classification_performance(model, X_train, X_test, y_train, y_test,
                                       classes=(config['label_values']),
                                       save_fig_to=os.path.join(experiment_report_path, name + '.jpg'))

        except Exception as e:
            logger.error(e)

    results.sort_values(['F1']).to_csv(os.path.join(experiment_report_path, 'vergleich.csv'))

    results.set_index('Model')\
        .sort_values(['F1'])\
        .plot.barh().get_figure().savefig(
            os.path.join(experiment_report_path, 'vergleich.jpg'),
            pad_inches=0.0, bbox_inches='tight')
