import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
from scipy import stats

class PreparingData:
    def __init__(self):
        self.categorical_columns = []
        self.processed_data = None
        self.insignificant_columns = []
        self.significant_columns = []

    def preprocess_data(self, data):

        # Remove rows with zeros in DayOfWeekClaimed and MonthClaimed
        self.processed_data = data[(data['DayOfWeekClaimed'] != 0) & (data['MonthClaimed'] != 0)]

        # Replace age 0 with 17
        self.processed_data.loc[self.processed_data['Age'] == 0, 'Age'] = 17

        # One Hot Encoding
        ohe = OneHotEncoder()
        self.categorical_columns = self.processed_data.select_dtypes(include=object).columns.tolist()
        feature_array = ohe.fit_transform(self.processed_data[self.categorical_columns]).toarray()
        feature_labels = ohe.categories_
        feature_labels = np.hstack(feature_labels)
        ohe_categories = pd.DataFrame(feature_array, columns=feature_labels)
        self.processed_data = pd.concat([self.processed_data, ohe_categories], axis=1)
        self.processed_data = self.processed_data.drop(self.categorical_columns, axis=1)

        # Removing duplicated columns
        self.processed_data = self.processed_data.loc[:, ~self.processed_data.columns.duplicated()]

        

    def chi_square_test(self, X, y):
        significant_columns = []
        for col in X.columns:
            contingency = pd.crosstab(y, X[col])
            pvalue = stats.chi2_contingency(contingency).pvalue
            if pvalue <= 0.05:
                significant_columns.append(col)
        
        return X[significant_columns]

