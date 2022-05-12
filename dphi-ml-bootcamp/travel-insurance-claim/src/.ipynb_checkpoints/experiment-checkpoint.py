"""
Framework for Imbalanced Classification Task
1. Select a metric:
    predict probabilities: PR-AUC (we only care about minority class)
    predict class labels: fbeta-score (beta depends on business case)

2. Evaluate resampling algorithm
    oversampling: SMOTE
    undersampling: Random Undersampling
    combination: SMOTE and Random Undersampling

3. Data preprocessing
    categorical: encoder
    numerical: function transformer

4. Evaluate various classifiers
    naive: Dummy Classifier
    linear: Logistic Regression
    non-linear: K Nearest Neighbors, Decision Tree, Support Vector Machine
    ensemble: Random Forest, XGB
"""
import numpy as np
from IPython.display import display
from dataset import pandas_setting, load_dataset, create_folds
from train import training, classifiers
from preprocessing import resampler, encoder

from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_transformer
from imblearn.pipeline import make_pipeline

from sklearn.metrics import f1_score

def main():
    pandas_setting()
    path = "~/practice-lab/dphi-ml-bootcamp/travel-insurance-claim/dataset/Training_set_label.csv"
    df = load_dataset(path)

    features = ['net_sales', 'commision_(in_value)', 'claim', 'k_fold']
    df = create_folds(df, target='claim')[features]

    majority_class = df[df['claim'] == 0].shape[0]
    minority_class = df[df['claim'] == 1].shape[0]
    scale_pos_weight = majority_class / minority_class

    scores = []
    for fold in range(5):
        f1 = train_the_model(
            fold,
            df=df,
            target='claim',
            drop_features=['claim', 'k_fold'],
            preprocessor=RobustScaler(),
            model=classifiers()['LogisticRegression'],
            filename='model_lr'
        )
        scores.append(f1)

    print(f"f1-score mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

def main(fold=1):
    pandas_setting()
    path = "~/practice-lab/dphi-ml-bootcamp/travel-insurance-claim/dataset/Training_set_label.csv"
    df = load_dataset(path)

    features = ['net_sales', 'commision_(in_value)', 'distribution_channel', 'claim', 'k_fold']
    df = create_folds(df, target='claim')[features]

    ohe = encoder()['OneHotEncoder']
    preprocessor = make_column_transformer(
        (RobustScaler(), ['net_sales', 'commision_(in_value)']),
        (ohe, ['distribution_channel']),
        remainder='passthrough'
    )
    res = resampler()['SMOTE']
    clf = classifiers()['LogisticRegression']

    pipeline = make_pipeline(
        preprocessor,
        res,
        clf
    )

    training(pipeline, df, target='claim', fold=fold)

def main(fold=1):
    pandas_setting()
    path = "~/practice-lab/dphi-ml-bootcamp/travel-insurance-claim/dataset/Training_set_label.csv"
    df = load_dataset(path)

    features = ['net_sales', 'commision_(in_value)', 'distribution_channel', 'claim', 'k_fold']
    df = create_folds(df, target='claim')[features]

    res = resampler()['SMOTEENN']
    clf = classifiers()['LogisticRegression']

    for name, enc in encoder().items():
        preprocessor = make_column_transformer(
            (RobustScaler(), ['net_sales', 'commision_(in_value)']),
            (enc, ['distribution_channel']),
            remainder='passthrough'
        )
        pipeline = make_pipeline(
            preprocessor,
            res,
            clf
        )
        print(name)
        training(pipeline, df, target='claim', fold=fold)

main(fold=1)





































# end of code
