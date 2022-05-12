import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

def training(pipeline, df, target='claim', fold=1):
    train = df[df['k_fold'] != fold].reset_index(drop=True)
    val = df[df['k_fold'] == fold].reset_index(drop=True)

    X_train = train.drop([target, 'k_fold'], axis=1)
    y_train = train[target]

    X_val = val.drop([target, 'k_fold'], axis=1)
    y_val = val[target]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    score = f1_score(y_val, y_pred)
    print(f"Fold-{fold} | f1-score = {score:.3f}")


def classifiers(class_weight='balanced', seed=0):
    clfs = {}
    clfs['DummyClassifier'] = DummyClassifier(
        random_state=seed,
        strategy='constant',
        constant=1
    )
    clfs['LogisticRegression'] = LogisticRegression(
        random_state=seed,
        solver='lbfgs',
        class_weight=class_weight,
        max_iter=5000
    )
    clfs['KNeighborsClassifier'] = KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )
    clfs['SVC'] = SVC(
        random_state=seed,
        class_weight=class_weight,
        max_iter=1000
    )
    clfs['DecisionTreeClassifier'] = DecisionTreeClassifier(
        random_state=seed,
        max_depth=7,
        class_weight=class_weight,
    )
    clfs['RandomForestClassifier'] = RandomForestClassifier(
        random_state=seed,
        max_depth=7,
        class_weight="balanced_subsample"
    )
    clfs['XGBClassifier'] = XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        max_depth=7,
        scale_pos_weight=67
    )
    return clfs




































# end of code
