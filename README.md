# Canvass_codingchallenge
Answers to the challenge

## Task 1

Link for this notebook can be found [here](https://github.com/abimur-123/Canvass_codingchallenge/blob/master/scripts/Task1.ipynb)

Models used can be found in this [folder](https://github.com/abimur-123/Canvass_codingchallenge/blob/master/models), prefixed with `Task1`.

### Assumptions

1. For few turbines there are readings beyond today's date. For the interest of this analysis, I am going to assume these are valid readings
2. Dealing with NAs in the dataset - Mean value imputation?
3. Date range isn't continuous. Should I assume that the unit does not fail in that period, or, should I be ignoring the gap in the range of days? (right censored data)
4. If the window length of 40 isn't satisfied, padding with 0s. This assumes that there isn't going to be a failure. `Check if padding with another number such as -1 helps with predictions.`

### Ideas

1. LSTM using keras to model dependancy over period of 40 days. Used recall and auc-roc(area under curve) to evaluate model performance. This can be attributed to the imbalance in the dataset.
2. Feature enginnering -- create a column `possible_failure`; mark it `1` if there is failure within a 40 day window, prior to the current observation for a turbine. This column can then be used as a predictor, as it now holds information about failures within a 40 days.
3. Basic multi-label ensemble models using sci-kit learn (https://arxiv.org/pdf/1609.08349.pdf) - Explore hamming distance to evaluate models and scikit learn functions (http://scikit.ml/api/classify.html#ensemble-approaches)
4. Kaplan meir estimates -- survival analysis based on failure events


Code to load LSTM - keras

```
from keras.models import model_from_json
# load json and create model
json_file = open(<file_name>.json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(<file_name>.h5)
print("Loaded model from disk")
```

Code to load other models
```
from sklearn.externals import joblib

clf = joblib.load('../models/<file_name>.pkl') 
```

## Task 2

Link for this notebook can be found [here](https://github.com/abimur-123/Canvass_codingchallenge/blob/master/scripts/Task2.ipynb)

Models used can be found in this [folder](https://github.com/abimur-123/Canvass_codingchallenge/blob/master/models), prefixed with `Task2`.

### Assumptions 

1. No break in 1hour observation of data of the parameters. Fetching the 6th record to model pollution 6 hours later (ease of analysis)
2. There are 365 rows with negative reading for pollution. Although this seems like an anomaly, ignoring it for the purpose of the analysis due to lack of domain knowledge. *Try removing these observations and predicting*

### Ideas

1. Imputing missing values with mean/mode of the column based on datatype.
2. Compared various boosting (ensemble algorithms) as I'm dealing with a small dataset. 
3. Looking at adjusted $R^2$ along with mean square error to evaluate model performance. Tune best performing model.
4. PCA shows that 5 features can explain close to 90% of the variance in the dataset. Perform PCA to reduce number of features for analysis. 
