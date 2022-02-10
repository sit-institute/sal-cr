import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class MissingValuesIndication(BaseEstimator, TransformerMixin):
    """
    Transformierer, der zu jedem Attribut ein Binärattribut anlegt,
    welches das Fehlen von Werten signalisiert.

    Parameters
    ----------
    suffix : str
        Suffix des Namens des anzulegenden Binärattributes.
    """

    def __init__(self, suffix='unknown'):
        self.suffix = suffix
        self.features_ = []

    def fit(self, X: pd.DataFrame):
        assert type(X) == pd.DataFrame, 'MissingValuesIndication takes DataFrame only'
        self.features_ = X.columns.to_list()
        return self

    def transform(self, X: pd.DataFrame):
        assert type(X) == pd.DataFrame, 'MissingValuesIndication takes DataFrame only'
        X_ = X.copy()

        for col in X.columns.to_list():
            X_[col + '_' + self.suffix] = X_[col].isnull().astype('int32')
            self.features_.append(col + '_' + self.suffix)

        return X_

    def get_feature_names(self, input_features=None):
        """Feature-Namen

        :param input_features: Parameter wird nicht verwendet
        :type input_features: list[str]
        :return: Liste von Feature-Namen
        :rtype: list[str]
        """
        return np.array(self.features_, dtype=object)
        