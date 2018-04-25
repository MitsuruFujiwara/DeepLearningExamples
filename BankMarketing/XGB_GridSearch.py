import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def main():
    # load Data
    df = pd.read_hdf('db.h5', key='train', mode='r')

    # set data
    trX, trY = np.array(df.drop('y', axis=1)), np.array(df['y'])

    # define Xgboost Classifier
    XGB = XGBClassifier()

    prm_learning_rate = [0.01, 0.05, 0.10, 0.15, 0.20]
    prm_max_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    prm_n_estimators = [10, 20, 30]
    prm_min_child_weight = [0.5, 0.75, 1.0]

    param_grid = [{'learning_rate':prm_learning_rate, 'max_depth':prm_max_depth,
                    'n_estimators': prm_n_estimators, 'min_child_weight': prm_min_child_weight}]

    gs = GridSearchCV(estimator=XGB, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)

    gs.fit(trX, trY)

    result = pd.DataFrame(gs.cv_results_)

    result.to_csv('result_xgb.csv')

    joblib.dump(gs.best_estimator_, 'xgb.pkl')

    print(gs.best_score_)
    print(gs.best_params_)
    print(result)

if __name__ == '__main__':
    main()
