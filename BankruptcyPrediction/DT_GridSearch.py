import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize

def main():
    # load Data
    df = pd.read_hdf('data.h5', key='data5_train', mode='r')

    # set data
    trX, trY = np.array(df.drop('class', axis=1)), np.array(df['class'])

    # define Decision Tree Classifier
    DT = DecisionTreeClassifier()

    prm_criterion =['gini', 'entropy']
    prm_max_depth =[8, 16, 24, 32, 40, 48, 56, 64, None]

    # set grid
    param_grid = [{'criterion':prm_criterion, 'max_depth':prm_max_depth}]

    gs = GridSearchCV(estimator=DT, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=-1)

    gs.fit(trX, trY)

    result = pd.DataFrame(gs.cv_results_)

    result.to_csv('result_dt5.csv')

    joblib.dump(gs.best_estimator_, 'dt5.pkl')

    print(gs.best_score_)
    print(gs.best_params_)
    print(result)

if __name__ == '__main__':
    main()
