import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def main():
    # load Data
    df = pd.read_hdf('db.h5', key='train', mode='r')

    # set data
    trX, trY = np.array(df.drop('y', axis=1)), np.array(df['y'])

    # define Random Forest Classifier
    RF = RandomForestClassifier()

    prm_n_estimators =np.power(10, np.arange(0,4,1))
    prm_max_depth = [5, 10, 15, 20, 25, 30, 35, 40, None]

    # set grid
    param_grid = [{'n_estimators':prm_n_estimators, 'max_depth':prm_max_depth}]

    gs = GridSearchCV(estimator=RF, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)

    gs.fit(trX, trY)

    result = pd.DataFrame(gs.cv_results_)

    result.to_csv('result_rf.csv')

    joblib.dump(gs.best_estimator_, 'rf.pkl')

    print(gs.best_score_)
    print(gs.best_params_)
    print(result)

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

if __name__ == '__main__':
    main()
