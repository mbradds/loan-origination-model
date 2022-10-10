import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


dependent_variable = "action_taken"

categorical_variables = ["activity_year",
                         "derived_msa-md",
                         "state_code",
                         "derived_loan_product_type",
                         "derived_dwelling_category",
                         "conforming_loan_limit",
                         "purchaser_type",
                         "preapproval",
                         "loan_type",
                         "loan_purpose",
                         "lien_status",
                         "reverse_mortgage",
                         "open-end_line_of_credit",
                         "business_or_commercial_purpose",
                         "hoepa_status",
                         "negative_amortization",
                         "interest_only_payment",
                         "balloon_payment",
                         "other_nonamortizing_features",
                         "construction_method",
                         "occupancy_type",
                         "manufactured_home_secured_property_type",
                         "manufactured_home_land_property_interest",
                         "total_units",
                         "debt_to_income_ratio",
                         "applicant_credit_score_type",
                         "co-applicant_credit_score_type",
                         "applicant_age",
                         "co-applicant_age",
                         "applicant_age_above_62",
                         "co-applicant_age_above_62",
                         "submission_of_application",
                         "initially_payable_to_institution",
                         "aus-1",
                         "aus-2",
                         "aus-3",
                         "aus-4",
                         "aus-5",
                         "denial_reason-1",
                         "denial_reason-2",
                         "denial_reason-3",
                         "denial_reason-4"]

continuous_variables = ["loan_amount",
                        "lei",
                        "county_code",
                        "interest_rate",
                        "census_tract",
                        "rate_spread",
                        "total_loan_costs",
                        "origination_charges",
                        "loan_term",
                        "loan_to_value_ratio",
                        "property_value",
                        "income",
                        "tract_population",
                        "tract_minority_population_percent",
                        "ffiec_msa_md_median_family_income",
                        "tract_to_msa_income_percentage",
                        "tract_owner_occupied_units",
                        "tract_one_to_four_family_homes",
                        "tract_median_age_of_housing_units",
                        "total_points_and_fees",
                        "discount_points",
                        "lender_credits",
                        "prepayment_penalty_term",
                        "intro_rate_period",
                        "multifamily_affordable_units"]


def set_cwd_to_script():
    dname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dname)


def get_data(file_name):
    path = os.path.join(os.getcwd(), "data", file_name)
    if ".csv" in file_name:
        df = pd.read_csv(path)
        return df
    elif ".xlsx" in file_name:
        df = pd.read_excel(path)
        return df
    else:
        print("enter a valid data file name!")
        return None


def find_high_corr(df):
    '''
    Independent variables that correlate strongly with the dependent variable
    (action taken) should be included in the model.

    Alot of the independent variables are correlated with each other. This is
    called multicolinarity and can interfere with the model results.
    https://towardsdatascience.com/multi-collinearity-in-regression-fe7a2c1467ea

    Parameters
    ----------
    df : DataFrame
        loan data prior to one hot encoding.

    Returns
    -------
    None.

    '''
    corr = df.corr()
    for col in corr:
        corr_list = corr[col]
        i = 0
        for corr_value in corr_list:
            if abs(corr_value) > 0.9 and col != corr.index[i]:
                print("High correlation (" + str(abs(corr_value))+") between " + col + " and " +
                      corr.index[i]+" ,condider removing from model to avoid multicolinearity")
            i = i+1


def filter_valid_outcomes(df):
    '''
    According to the problem statement, only issued loans (1) and rejected
    loans (3) should be evaluated.

    Parameters
    ----------
    df : DataFrame
        loan data.

    Returns
    -------
    df : DataFrame
        loan data.

    '''
    df = df[df[dependent_variable].isin([1, 3])].copy()
    df[dependent_variable] = df[dependent_variable].map({1: 1,
                                                         3: 0})

    # df["action_taken"] = df["action_taken"].astype("category")
    print("invalid loan outcomes removed")
    return df


def filter_null_columns(df, threshold=0.5):
    '''
    Features/columns that have a high amount of blank data (more than 50% of
    the column is empty) are unlikely to be useful to the model and should be
    removed.

    Parameters
    ----------
    df : DataFrame
        loan data.
    threshold : float, optional
        DESCRIPTION. The default is 0.5. A value between zero and 1. Eg 0.5
        will delete columns with >50 null values.

    Returns
    -------
    df : DataFrame
        DESCRIPTION. A DataFrame with high null columns deleted.

    '''
    total_rows = len(df)
    drop_count = 0
    for col in df.columns:
        total_nulls = df[col].isnull().sum()
        if total_nulls >= (threshold*total_rows):
            del df[col]
            drop_count = drop_count + 1
    print(str(drop_count) + " variables with high missing variables removed")
    return df


def categorical_steps(df, col):
    df[col] = df[col].fillna("not provided")
    df[col] = pd.Categorical(df[col])
    df[col] = df[col].cat.codes
    df[col] = df[col].astype("category")
    return df, col


def process_categorical_variables(df, variable_list):
    '''
    Applies the following transformations to categorical variables:
        1. switched "not provided" with null
        2. applies the categorical data type to the column
        3. replaces categories with numeic codes (object to float conversion)

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION. Loan data
    variable_list : Array
        DESCRIPTION. List of categorical independent variables in the loan data.

    Returns
    -------
    df : DataFrame
        DESCRIPTION. Loan data.

    '''
    for col in df:
        if col in variable_list:
            df, col = categorical_steps(df, col)
    print("categorical variables processed")
    return df


def process_continuous_variables(df, variable_list, standardize=True):
    '''
    Applies the following transformation to continuous variables:
        1. replaces exempt with null
        2. converts column to numeric data type
        3. fills empty values with the mean of the column
        4. optionally standardizes values between 0 and 1

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION. Loan data
    variable_list : array
        DESCRIPTION. list of continuous independent variables in the loan data.
    standardize : optional, boolean
        DESCRIPTION. The default is True. Whether to standardize column values between zero and 1.

    Returns
    -------
    df : DataFrame
        DESCRIPTION. Loan data

    '''
    for col in df:
        if col in variable_list:
            if col == "lei":
                df, col = categorical_steps(df, col)

            df[col] = df[col].replace({"Exempt": np.nan})
            # TODO look into an exempt/non exempt categorical variable
            df[col] = pd.to_numeric(df[col])
            df[col] = df[col].fillna(df[col].mean())
            if standardize:
                max_value = df[col].max()
                min_value = df[col].min()
                df[col] = (df[col] - min_value) / (max_value - min_value)
    print("continuous variables standardized")
    return df


def filter_low_variance(df):
    '''
    Varaibles with low variance (only one observation) will not benefit any
    model and should be removed prior to training.

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION. Loan data.

    Returns
    -------
    df : DataFrame
        DESCRIPTION. Loan data.

    '''
    drop_count = 0
    for col in df.columns:
        unique = set(list(df[col]))
        if len(unique) == 1:
            del df[col]
            drop_count = drop_count + 1
    print(str(drop_count) + " variables with low variance removed")
    return df


def check_variables_for_nulls(df):
    '''
    Checks if a dataframe has any null values in any column.
    '''
    for col, null_count in zip(df.isnull(), df.isnull().sum()):
        if null_count > 0:
            print(col+" has null values!")


def pre_process_loan_data(df, cat_variables, cont_variables, standardize_cont=True, test_data=True):
    '''
    Combines several data processing functions into one function. Differentiates
    between the test data set, and the evaluation data.

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION. Loan data (test data or evaluation data).
    cat_variables : array
        DESCRIPTION. List of categorical independent variables.
    cont_variables : array
        DESCRIPTION. List of continuous variables.
    standardize_cont : TYPE, optional
        DESCRIPTION. The default is True.
    test_data : Boolean, optional
        DESCRIPTION. The default is True. Where the loan data is the test
        data or the evalution data.

    Returns
    -------
    df : DataFrame
        DESCRIPTION. Processed loan data, ready for additional sklearn
        pre-processing.

    '''
    if test_data:
        df = filter_valid_outcomes(df)
        df = filter_null_columns(df)
        df = filter_low_variance(df)

    df = process_categorical_variables(df, cat_variables)
    df = process_continuous_variables(df, cont_variables, standardize=standardize_cont)

    if test_data:
        df = df.dropna()
        check_variables_for_nulls(df)
        find_high_corr(df)
        before_size = df.shape[0]
        df = df.drop_duplicates()
        after_size = df.shape[0]
        print(before_size - after_size, " duplicate loans removed")
    return df


def limit_data(df, n=20000):
    '''
    Returns a smaller dataframe for faster model training and evaluation.
    '''
    if n:
        df = df.head(n)
    return df


def back_to_df(X):
    '''
    Converts sklearn pipeline output array back to pandas DataFrame for use
    in the models.
    '''
    try:
        df = pd.DataFrame(X.toarray())
    except:
        df = pd.DataFrame(X)
    return df


def get_x_and_y(df, dep_variable, test_data):
    '''
    Splits the data into X and y components and deletes the dependent variable
    from X.
    '''
    if test_data:
        y = df[[dep_variable]]
        y = y.values.ravel()
        df = df.drop(dep_variable, axis=1)
    else:
        y = None
    return df, y


def get_train_test_data(X, y, features=False, test_size=0.25):
    '''
    Splits loan data X and y components into train and test sets.

    Parameters
    ----------
    X : DataFrame
        DESCRIPTION. Loan data independent variables.
    y : Series
        DESCRIPTION. Loan data dependent variable.
    features : array, optional
        DESCRIPTION. The default is False. An array of feature names from
        feature selection step. Only features in the list will appear in
        the train and test data.
    test_size : TYPE, optional
        DESCRIPTION. The default is 0.25.

    Returns
    -------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    '''
    if features:
        X = X[features]

    if test_size != 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    else:
        X_train = X
        y_train = y
        X_test = None
        y_test = None

    return X_train, X_test, y_train, y_test


def feature_selection(df, y, n=500, num_features="best"):
    '''
    Stepwise Feature Selection algorithm chosing the best features. Returns
    a list of feature names.
    '''
    df = df.head(n)
    y = y[:n]
    X_train, X_test, y_train, y_test = get_train_test_data(df, y, False, 0)
    sfs = SFS(LogisticRegression(max_iter=10000),
              k_features=num_features,
              forward=True,
              floating=False,
              scoring='accuracy',
              n_jobs=-1,
              cv=4)
    sfs.fit(X_train, y_train)
    print("feature selection score: ", sfs.k_score_)
    print("SFS chosen features: ", sfs.k_feature_names_)
    return list(sfs.k_feature_names_)


def sklearn_pre_process_loan_data(data, limit=20000, test_data=True):
    '''
    Creates an sklearn pipeline for processing categorical and continuous
    data separately. Categorical data is one hot encoded and continuous
    data is scaled.

    Parameters
    ----------
    data : DataFrame
        DESCRIPTION. Loan data.
    limit : integer, optional
        DESCRIPTION. The default is 20000. Whether to limit the size of
        the data.
    test_data : boolean, optional
        DESCRIPTION. The default is True. Whether the dataset is the test
        data or the evaluation data.

    Returns
    -------
    df : DataFrame
        DESCRIPTION. Processed loan data ready for train test split and
        model training.
    y : Series
        DESCRIPTION. Loan data dependent variable.
    preprocessor : Sklearn pipleine
        DESCRIPTION.

    '''
    data = limit_data(data, limit)
    data, y = get_x_and_y(data, dependent_variable, test_data)

    numerical_columns_selector = selector(dtype_include=float)
    categorical_columns_selector = selector(dtype_exclude=[float, object])

    numerical_columns = numerical_columns_selector(data)
    categorical_columns = categorical_columns_selector(data)

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    ct = ColumnTransformer([
        ('one hot encoder', categorical_preprocessor, categorical_columns),
        ('standard_scaler', numerical_preprocessor, numerical_columns)],
        remainder='passthrough')

    df_convert = FunctionTransformer(back_to_df)
    preprocessor = make_pipeline(ct, df_convert)

    df = preprocessor.fit_transform(data)
    df.columns = ct.get_feature_names_out()
    return df, y, preprocessor


def evaluate_model(results, model, model_name, X_train, X_test, y_train, y_test):
    results[model_name] = {"Train Score": metrics.accuracy_score(y_train, model.predict(X_train)),
                           "Test Score": metrics.accuracy_score(y_test, model.predict(X_test)),
                           "Test Precision": metrics.precision_score(y_test, model.predict(X_test)),
                           "Test Recall": metrics.recall_score(y_test, model.predict(X_test)),
                           "Test AUC": metrics.roc_auc_score(y_test, model.predict(X_test))}
    return results


def column_standardizer(df_test, df_eval):
    '''
    This function makes sure that the processed input data (state_IL_application.csv)
    and the processed evaluation data (X_test.xlsx) have the same feautures prior
    to model training.

    This is important because the evaluation data is smaller, and some categorical
    features may not have a full set of observations. After one hot encoding,
    this could lead to a different number of features between the datasets.
    '''
    test_cols = df_test.columns
    eval_cols = df_eval.columns

    # if there is a column in the train data and not in the eval data, remove it from test
    for tc in test_cols:
        if tc not in eval_cols:
            del df_test[tc]

    for ec in eval_cols:
        if ec not in test_cols:
            del df_eval[ec]

    return df_test, df_eval



