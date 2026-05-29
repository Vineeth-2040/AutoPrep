import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn import set_config 
from sklearn.preprocessing import PowerTransformer,FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,KFold
from itertools import product 
from sklearn.linear_model import  LogisticRegression

# for some tranformers to return pands as output not np
set_config(transform_output="pandas")
# missing value dealing genral 

def load_data(path):
    temp = pd.read_csv(path)
    return temp

def drop_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"This column '{col}' is not found!")

    return df.drop(columns=columns)

def drop_missing_target(df, target_col):
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataframe")
    return df.dropna(subset=[target_col]) 


def check_missing(df):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    report = pd.DataFrame({
        "column_name": df.columns,
        "total_missing": missing_count.values,
        "percentage_missing": missing_percent.round(2).values
    })
    return report



def drop_threshold(df, report, threshold=40):
    cols_to_drop = report.loc[
        report["percentage_missing"] > threshold,
        "column_name"
    ]
    return df.drop(columns=cols_to_drop)




# here in next update , thsi place is a branch place for regression or classification decison ok 
#  we are not handling date time in v1 it will be in version 2 
# Not handling multiple output regression for now please focus here. 

def detect_column_types(
    df,
    target: str = None,
    unique_threshold=20,
    unique_ratio_threshold=0.05
):
    result = {
        "numerical": [],
        "categorical": [],
        "datetime": [],    # This si dropped further will not be imputed in v1 
        "mixed": [],
        "encoded_categorical": [],
        "target": None
    }

    if target is not None:
        if not isinstance(target, str):
            raise TypeError("target must be a column name string, example: target='price'")

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        target_series = df[target].dropna()

        if target_series.empty:
            raise ValueError(f"Target column '{target}' is empty.")

        if pd.api.types.is_numeric_dtype(target_series):
            result["target"] = {
                "name": target,
                "type": "numerical"
            }
        else:
            numeric_target = pd.to_numeric(target_series, errors="coerce")
            numeric_success_ratio = numeric_target.notna().mean()

            if numeric_success_ratio > 0.8:
                result["target"] = {
                    "name": target,
                    "type": "numerical"
                }
            else:
                raise ValueError(
                    f"Target column '{target}' looks categorical. "
                    "v1 is for regression only."
                )

    feature_df = df.drop(columns=[target]) if target is not None else df.copy()

    for col in feature_df.columns:
        s = feature_df[col].dropna()

        if s.empty:
            result["mixed"].append(col)
            continue

        unique_count = s.nunique()
        unique_ratio = unique_count / len(s)

        if pd.api.types.is_datetime64_any_dtype(feature_df[col]):
            result["datetime"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(feature_df[col]):
            if unique_count <= unique_threshold or unique_ratio <= unique_ratio_threshold:
                result["encoded_categorical"].append(col)
            else:
                result["numerical"].append(col)
            continue

        datetime_converted = pd.to_datetime(s, errors="coerce")
        datetime_success_ratio = datetime_converted.notna().mean()

        if datetime_success_ratio > 0.8:
            result["datetime"].append(col)
            continue

        numeric_converted = pd.to_numeric(s, errors="coerce")
        numeric_success_ratio = numeric_converted.notna().mean()

        if numeric_success_ratio > 0.8:
            if unique_count <= unique_threshold or unique_ratio <= unique_ratio_threshold:
                result["encoded_categorical"].append(col)
            else:
                result["numerical"].append(col)
            continue

        has_numbers = numeric_converted.notna().any()
        has_text = numeric_converted.isna().any()

        if has_numbers and has_text:
            result["mixed"].append(col)
        else:
            result["categorical"].append(col)

    return result

def group_rare_categories(df, columns, threshold=0.05, other_label="Other"):
    df = df.copy()

    for col in columns:
        freq = df[col].value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index
        df[col] = df[col].apply(lambda x: other_label if x in rare_categories else x)

    return df


# here ends big algorithm to detect column whcih is important for encoding and also missing value 
#in next versiaon we will make algo more robust , hright noiw it buts both one-hot encoding and ordinal to one calumn 
# but ordinal can be traformed or scalled but one hot must pass through without scallign or tranforming




# this funtion not necessary for now , we will do cross val internally when pipeline is buit .
# def cross_val_splits(df, target_col, n_splits=5):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     folds = []

#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     X = X.drop(columns=X.select_dtypes(include=["datetime", "datetimetz"]).columns)

#     for train_idx, test_idx in kf.split(df):
#         train_df = X.iloc[train_idx].copy()
#         test_df = X.iloc[test_idx].copy()
#         y_train = y.iloc[train_idx].copy()
#         y_test = y.iloc[test_idx].copy()

#         folds.append({
#             "train_df": train_df,
#             "test_df": test_df,
#             "y_train": y_train,
#             "y_test": y_test
#         })

#     return folds



def prepare_x_y(df, target_col, mixed_cols=[], date_cols=[]):
    X = df.drop(columns=[target_col])
    
    # drop mixed and datetime cols detected by detect_column_types
    cols_to_drop = mixed_cols + date_cols
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
    
    # second safety check — drop any remaining datetime by dtype
    # select_dtypes returns empty list if no datetime cols exist — no crash
    datetime_cols = X.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    if datetime_cols:
        X = X.drop(columns=datetime_cols)
    
    y = df[target_col]
    return X, y



# missing imputation 
# simpleimputer: mean ,media , knn , Mice ,
# custom : end of distruntion , iqr , random_sample 

#, numerical : simple imputor mean , meadian ,  then knn , mice ,  missing indicator , 3 custom : random_sample, end_ of_distribution , then iqr 
#then cate : naming missinign as missing category , then most_frequent and missing indicator

# categorical : group small categories as group missign as missing categories , most_frequent 
#missign indicator 



class RandomSampleImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.fill_values_ = {col: X[col].dropna().values for col in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            mask = X[col].isna()
            if mask.any() and len(self.fill_values_[col]) > 0:
                X.loc[mask, col] = np.random.choice(self.fill_values_[col], size=mask.sum())
        return X.values


class EndOfDistributionImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.fill_values_ = {col: X[col].mean() + 3 * X[col].std() for col in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].fillna(self.fill_values_[col])
        return X.values


class IQRImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.fill_values_ = {}
        for col in X.columns:
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            self.fill_values_[col] = X[col][(X[col] >= q1) & (X[col] <= q3)].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].fillna(self.fill_values_[col])
        return X.values

class CategoricalRandomSampleImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.fill_values_ = {col: X[col].dropna().values for col in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            mask = X[col].isna()
            if mask.any() and len(self.fill_values_[col]) > 0:
                X.loc[mask, col] = np.random.choice(self.fill_values_[col], size=mask.sum())
        return X.values


# ── ready to use lists ────────────────────────────────────────────────────────

numerical_imputers = [
    ('mean',             SimpleImputer(strategy='mean')),
    ('median',           SimpleImputer(strategy='median')),
    ('mean+indicator',   SimpleImputer(strategy='mean',   add_indicator=True)),
    ('median+indicator', SimpleImputer(strategy='median', add_indicator=True)),
    ('knn',              KNNImputer(n_neighbors=5)),
    ('mice',             IterativeImputer(random_state=0, max_iter=10)),
    ('random_sample',    RandomSampleImputer()),
    ('end_of_dist',      EndOfDistributionImputer()),
    ('iqr',              IQRImputer()),
]

categorical_imputers = [
    ('most_frequent',        SimpleImputer(strategy='most_frequent')),
    ('missing_cat',          SimpleImputer(strategy='constant', fill_value='missing_cat')),
    ('Random_category',      CategoricalRandomSampleImputer()),
]

# thsi library will consider if too many caterory are found we will group- the samll care to one category and it will send to encoding 


#Encoding part 
# only the catergorical data is encodered , other stuff move to scaller 

encoders = [
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
]
 

# transforming  then move to scalling 

# all_transformers = [
#     ('passthrough',  'passthrough'),
#     ('yeojohnson',   PowerTransformer(method='yeo-johnson')),  # works on everything
# ]

# positive_only_transformers = [
#     ('log',   FunctionTransformer(np.log1p, validate=True)),
#     ('sqrt',  FunctionTransformer(np.sqrt,  validate=True)),
# ]

# nonzero_transformers = [
#     ('reciprocal', FunctionTransformer(np.reciprocal, validate=True)),
# ]



def get_valid_transformers(X, num_cols):
    
    if not num_cols:  # no numerical columns at all
        return [('passthrough', 'passthrough')]
    
    min_val  = X[num_cols].min().min()
    has_zero = (X[num_cols] == 0).any().any()

    valid = [
        ('passthrough', 'passthrough'),
        ('yeojohnson',  PowerTransformer(method='yeo-johnson')),
    ]

    if min_val >= 0:
        valid.append(('log',  FunctionTransformer(np.log1p)))
        valid.append(('sqrt', FunctionTransformer(np.sqrt)))

    if not has_zero:
        valid.append(('reciprocal', FunctionTransformer(np.reciprocal)))

    return valid



def to_df(arr,columns):
    return pd.DataFrame(arr,Columns=columns)


# remember in pipleline building you need to put logic such that if column as + or neg  or like that so permuation handl;es it clean
# example logic for that 
# min_val = df[num_cols].min().min()
# has_zero = (df[num_cols] == 0).any().any()

# transformers = all_transformers.copy()

# if min_val >= 0:
#     transformers += positive_only_transformers

# if not has_zero:
#     transformers += nonzero_transformers


scalers = [
    ('passthrough',        'passthrough'),
    ('standard',           StandardScaler()),
    ('minmax',             MinMaxScaler()),
    ('robust',             RobustScaler()),
    ('maxabs',             MaxAbsScaler()),
    ('mean_normalisation', FunctionTransformer(lambda X: (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0)),validate=True)),
]


# final pipe line 
def build_pipeline(num_imp, cat_imp, encoder, transformer, scaler, num_cols, cat_cols,encoded_cols):
    
    num_pipeline = Pipeline([
        ('imputer',     num_imp),
        ('transformer', transformer),
        ('scaler',      scaler),
    ])

    cat_pipeline = Pipeline([
        ('imputer',  cat_imp),
        ('encoder',  encoder),
    ])

    encoded_pipline=Pipeline([
        ('imputer',  num_imp),
    ])

    preprocessor = ColumnTransformer([
        ('num',     num_pipeline,       num_cols),
        ('enc_cat', encoded_pipline,   encoded_cols),
        ('cat',     cat_pipeline,       cat_cols),
    ], remainder='drop')

    return preprocessor  




def run_experiments(X, y, num_cols, cat_cols,encoded_cols, numerical_imputers, categorical_imputers, encoders, transformers, scalers):

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # total combos for progress tracking
    total = (len(numerical_imputers) * len(categorical_imputers) *
             len(encoders) * len(transformers) * len(scalers))
    current = 0

    for (num_imp_name, num_imp), \
        (cat_imp_name, cat_imp), \
        (enc_name,     enc),     \
        (trans_name,   trans),   \
        (scl_name,     scl)      \
        in product(numerical_imputers, categorical_imputers, encoders, transformers, scalers):

        current += 1
        print(f"[{current}/{total}] {num_imp_name} | {cat_imp_name} | {enc_name} | {trans_name} | {scl_name}", end=" → ")

        try:
            # build data path
            preprocessor = build_pipeline(num_imp, cat_imp, enc, trans, scl, num_cols, cat_cols,encoded_cols)

            # attach model separately
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model',        LogisticRegression(max_iter=1000))
            ])

            # run cross validation
            scores = cross_val_score(full_pipeline, X, y, cv=kf, scoring='accuracy')

            print(f"r2 = {scores.mean():.4f} ± {scores.std():.4f}")

            results.append({
                'num_imputer':  num_imp_name,
                'cat_imputer':  cat_imp_name,
                'encoder':      enc_name,
                'transformer':  trans_name,
                'scaler':       scl_name,
                'r2_mean':      round(scores.mean(), 4),
                'r2_std':       round(scores.std(),  4),
                'status':       'ok'
            })

        except Exception as e:
            print(f"FAILED — {e}")
            results.append({
                'num_imputer':  num_imp_name,
                'cat_imputer':  cat_imp_name,
                'encoder':      enc_name,
                'transformer':  trans_name,
                'scaler':       scl_name,
                'r2_mean':      None,
                'r2_std':       None,
                'status':       f'failed: {e}'
            })

    # sort by score, failures at bottom
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('r2_mean', ascending=False).reset_index(drop=True)

    return results_df


def show_results(results_df, output_path='autoprep_results.xlsx'):

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # sheet 1 — full results sorted by score
        results_df.to_excel(writer, sheet_name='Full Results', index=False)

        # sheet 2 — top 10 only
        top10 = results_df[results_df['status'] == 'ok'].head(10)
        top10.to_excel(writer, sheet_name='Top 10', index=False)

        # sheet 3 — best combo per num_imputer
        best_per_imputer = (
            results_df[results_df['status'] == 'ok']
            .sort_values('r2_mean', ascending=False)
            .groupby('num_imputer')
            .first()
            .reset_index()
            [['num_imputer', 'cat_imputer', 'encoder', 'transformer', 'scaler', 'r2_mean', 'r2_std']]
        )
        best_per_imputer.to_excel(writer, sheet_name='Best Per Imputer', index=False)

        # # sheet 4 — health report
        # health_df = pd.DataFrame(health_report)
        # health_df.to_excel(writer, sheet_name='Data Health', index=False)

        # # sheet 5 — failed combos
        # failed = results_df[results_df['status'] != 'ok']
        # if not failed.empty:
        #     failed.to_excel(writer, sheet_name='Failed Combos', index=False)

    print(f"\nresults saved to {output_path}")
    print(f"total combos:   {len(results_df)}")
    print(f"successful:     {len(results_df[results_df['status'] == 'ok'])}")
    print(f"failed:         {len(results_df[results_df['status'] != 'ok'])}")
    print(f"best r2:        {results_df['r2_mean'].max():.4f}")
    print(f"best combo:")
    best = results_df[results_df['status'] == 'ok'].iloc[0]
    print(f"  num_imputer:  {best['num_imputer']}")
    print(f"  cat_imputer:  {best['cat_imputer']}")
    print(f"  encoder:      {best['encoder']}")
    print(f"  transformer:  {best['transformer']}")
    print(f"  scaler:       {best['scaler']}")