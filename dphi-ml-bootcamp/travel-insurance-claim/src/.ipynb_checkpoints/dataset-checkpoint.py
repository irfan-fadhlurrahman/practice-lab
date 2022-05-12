import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def pandas_setting(max_columns=40):
    import pandas as pd
    pd.options.display.max_columns = max_columns

def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

def split_dataset(path, target='reached.on.time_y.n', test_size=0.1, seed=42):
    df = load_dataset(path)
    return train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target]
    )

def create_folds(df, target="claim", seed=0):
    df = df.copy()
    df["k_fold"] = -1

    k = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X = df.drop(target, axis=1)
    y = df[target]

    for fold, (train_idx, val_idx) in enumerate(k.split(X, y)):
        df.loc[val_idx, 'k_fold'] = fold

    return df

def missing_values(df):
    table = pd.DataFrame(
        columns=['variable',
                 'no_unique',
                 'pandas_dtype',
                 'missing_value',
                 '%_missing_values',
                 'unique_value']
    )

    for i, var in enumerate(df.columns):
        table.loc[i] = [var,
                        df[var].nunique(),
                        df[var].dtypes,
                        df[var].isnull().sum(),
                        df[var].isnull().sum() * 100 / df.shape[0],
                        df[var].unique().tolist()
        ]
    return table




































# end of code
