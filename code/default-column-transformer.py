from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, Binarizer

from sal.data import Attrs
from sal.data.cleaning import MissingValuesIndication
from sal.data.encode import BinaryNominalEncoder
from sal.data.load import typed_view
from sal.features.selection import DefaultAttributeSelector, drop_constant_attributes

ZWEI_JAHRE = 24 # Monate

class DefaultColumnTransformer(ColumnTransformer, Attrs):
    """Transformierer, der die Spalten des SAL-Datensatzes in Features umwandelt.
    """

    def __init__(self):
        self._attrs_nominal = \
            self.quali_nominal + \
            self.quali_bin + \
            self.quali_nominal_mutation_indicators + \
            self.quant_discrete_mutation_indicators

        self._attrs_eln = ['ELNRisk', 'CGELN', 'CGSTUD']
        self._attrs_discrete = self.quant_discrete + self.dates
        self._attrs_continuous = self.quant_continuous

        super().__init__(transformers=[
            ('OSTM', Pipeline(steps=[
                ('dichotomization', Binarizer(threshold=ZWEI_JAHRE))]),
             ['OSTM']),

            ('ECOG', Pipeline(steps=[
                ('missing_indicator', MissingValuesIndication(suffix='unknown')),
                ('imputer', SimpleImputer(strategy='median'))]),
             ['ECOG']),

            ('nominal', Pipeline(steps=[
                ('binary_encoder', BinaryNominalEncoder()),
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('one_hot_encoder', OneHotEncoder(sparse=False))
            ]), self._attrs_nominal),

            ('ELN', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('ordinal_encoder', OrdinalEncoder(
                    categories=[['unknown', 'adv', 'int', 'fav']] * len(self._attrs_eln)))
            ]), self._attrs_eln),

            ('discrete', Pipeline(steps=[
                ('missing_indicator', MissingValuesIndication(suffix='unknown')),
                ('imputation', SimpleImputer(strategy='median'))]),
             self._attrs_discrete),

            ('continuous', Pipeline(steps=[
                ('imputation', SimpleImputer(strategy='median'))]),
             self._attrs_continuous)
        ])

    def _feature_names_from(self, path: str, attrs: List[str]) -> List[str]:
        pathList = path.split('/')

        return self \
            .named_transformers_[pathList[0]] \
            .named_steps[pathList[1]] \
            .get_feature_names(attrs).tolist()

    def get_feature_names(self, params: dict):
        """Feature-Namen

        :return: Liste von Feature-Namen
        :rtype: list[str]
        """
        try:
            ostm_threshold = params['preprocessor__OSTM__dichotomization__threshold']
        except KeyError:
            ostm_threshold = ZWEI_JAHRE

        return np.array([
            f'OSTM_above_{ostm_threshold}',
            *self._feature_names_from('ECOG/missing_indicator', attrs=['ECOG']),
            *self._feature_names_from('nominal/one_hot_encoder', attrs=self._attrs_nominal),
            *self._attrs_eln,
            *self._feature_names_from('discrete/missing_indicator', attrs=self._attrs_discrete),
            *self._attrs_continuous], dtype=object)
