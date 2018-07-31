import pandas as pd
import numpy as np

# scikit learn models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib


class RandomForestWrapper():
    """Wrapper for the Random Forest classifier

    Attributes:
        rf_model (str): Trained random forest classifier
        scaling_features (:obj:`list`): List of numeric columns that have to be scaled
        cat_features (:obj:`list`): List of categorical columns that do not have to be scaled
        stdsc (:obj:`Standardizer`): Standardizer class fit on training data
    """

    def __init__(self):
        self.scaling_features= ['operational_setting_1',
                                 'operational_setting_2',
                                 'sensor_measurement_1',
                                 'sensor_measurement_2',
                                 'sensor_measurement_3',
                                 'sensor_measurement_4',
                                 'sensor_measurement_5',
                                 'sensor_measurement_6',
                                 'sensor_measurement_7',
                                 'sensor_measurement_8',
                                 'sensor_measurement_9',
                                 'sensor_measurement_10',
                                 'sensor_measurement_11',
                                 'sensor_measurement_12',
                                 'sensor_measurement_13',
                                 'sensor_measurement_14',
                                 'sensor_measurement_15',
                                 'sensor_measurement_16',
                                 'sensor_measurement_17',
                                 'sensor_measurement_18',
                                 'sensor_measurement_19',
                                 'sensor_measurement_20',
                                 'sensor_measurement_21']
        self.cat_features='operational_setting_3_High'
        self.rf_model=joblib.load('../models/task1_random_forest.pkl') # load model
        self.stdsc=joblib.load('../models/task1_standardizer.pkl') # load model

    def bind_scaled_non_scaled(self, data, scaled_feat, unscaled_feat, model):
        """
        This function scales numeric features and binds other features to the matrix

        Args:
            data: Features and label data
            scaled_feat: Feature column names to be scaled
            unscaled_feat: Columns which do not have to be scaled
            model: Method to be used for scaling numeric values (default is Standard scaler)

        Output:
            bind_data: pandas dataframe
                Data frame consisting of scaled and unscaled features for model

        """

        scaled_data = model.transform(data.loc[:,scaled_feat])
        bind_data =  pd.concat([pd.DataFrame(scaled_data, columns = scaled_feat),
                                data.loc[:,unscaled_feat].reset_index(drop=True)],
                               axis = 1
                              )
        return bind_data

    def predict(self, x):
        """
        Predict values on test set

        Args:
            x (:obj: `pandas dataframe`): Test set to be predicted (without date or Turbine id.)

        """
        if x.shape[1] != 24:
            print('Incorrect number of input columns')
            return -1
        elif 'operational_setting_3' not in x.columns:
            print('Operational setting 3 column not present.')
            return -1
        elif 'High' not in x['operational_setting_3'].unique():
            x.drop('operational_setting_3', axis = 1, inplace=True)
            x['operational_setting_3_High'] = 0
        else:
            print('Creating one-hot encoding vector for operational setting 3 column')
            x = pd.concat([x.drop('operational_setting_3', axis = 1),
                        pd.get_dummies(x.operational_setting_3,
                                       prefix = 'operational_setting_3').loc[:,'operational_setting_3_High']],
                        axis = 1)

        x_test = self.bind_scaled_non_scaled(x,
                                             self.scaling_features,
                                             self.cat_features,
                                             self.stdsc)

        return self.rf_model.predict(x_test.values)
