import pandas as pd
import numpy as np

# scikit learn models
from sklearn.externals import joblib

class XGBoostWrapper():
    """Wrapper for the Random Forest classifier

    Attributes:
        xgb_model (str): Trained XGB regressor model
        num_features (:obj:`list`): List of numeric columns that have to be scaled
        cat_features (:obj:`list`): List of categorical columns that do not have to be scaled
        stdsc (:obj:`Standardizer`): Standardizer class fit on training data
        pcamode (:obj:`user-defined function`): PCA model to reduce dimensions
    """

    def __init__(self):
        self.num_features=['rh',
                             't',
                             'c6h6',
                             's5',
                             'co_gt',
                             'no2',
                             's2',
                             'nox',
                             's4',
                             's3',
                             'nhmc',
                             'ah']
        self.cat_features = ['level_High',
                             'level_Low',
                             'level_Moderate',
                             'level_VeryHigh',
                             'level_Verylow']
        self.xgb_model=joblib.load('../models/task2_model.pkl') # load model
        self.pcamodel=joblib.load('../models/task2_PCA.pkl') # load model
        self.stdsc=joblib.load('../models/task2_scaler.pkl') # load model

    def pre_process(self, X_test):
        """
        This function scales numeric features and binds other features to the matrix

        Args:
            X_test: input dataframe

        Output:
            X_test_std: processed test set 
                Data frame consisting of scaled and unscaled features for model

        """

        X_test_std = self.stdsc.transform(X_test.loc[:,self.num_features])
        X_test_std = self.pcamodel.transform(X_test_std)
        X_test_std = pd.DataFrame(X_test_std, columns = ['comp_' + str(i + 1) for i in range(X_test_std.shape[1])])
        # If level does not exist create column with 0s
        for i in self.cat_features:
            print(i)
            if i not in X_test.columns:
                X_test[i] = 0

        print(X_test.columns)
        X_test_std = pd.concat([X_test_std,
                                X_test.loc[:,self.cat_features].reset_index(drop = True)
                                ], axis = 1
                                )

        return X_test_std

    def predict(self, x):
        """
        Predict values on test set

        Args:
            x (:obj: `pandas dataframe`): Test set to be predicted (without date or Turbine id.)

        """
        print('Creating one-hot encoding vector for operational setting 3 column')
        x = pd.concat([x.drop(['level'],axis = 1),
                      pd.get_dummies(x.level,prefix = 'level')],
                      axis = 1,
                     )

        x_test = self.pre_process(x)
        x_test = x_test[sorted(x_test.columns)]

        return self.xgb_model.predict(x_test.values)
