import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
                        "tract_median_age_of_housing_units"]

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


'''
Correlation matrix

Independent variables that correlate strongly with the dependent variable
(action taken) should be included in the model.

Alot of the independent variables are correlated with each other. This is
called multicolinarity and can interfere with the model results.
https://towardsdatascience.com/multi-collinearity-in-regression-fe7a2c1467ea

'''
def find_high_corr(df):
    corr = df.corr()
    for col in corr:
        corr_list = corr[col]
        i = 0
        for corr_value in corr_list:
            if abs(corr_value) > 0.9 and col != corr.index[i]:
                print("High correlation ("+str(abs(corr_value))+") between "+col+" and "+
                      corr.index[i]+" ,condider removing from model to avoid multicolinearity")
            i = i+1


def filter_valid_outcomes(df):
    df = df[df["action_taken"].isin([1, 3])].copy()
    # df["action_taken"] = df["action_taken"].astype("category")
    print("invalid loan outcomes removed")
    return df


def filter_null_columns(df, threshold=0.5):
    total_rows = len(df)
    drop_count = 0
    for col in df.columns:
        total_nulls = df[col].isnull().sum()
        if total_nulls >= (threshold*total_rows):
            del df[col]
            drop_count = drop_count + 1
    print(str(drop_count)+ " variables with high missing variables removed")
    return df


def process_categorical_variables(df, variable_list):
    # switch categories to integers for the model
    # set types as category
    for col in df:
        if col in variable_list:
            df[col] = df[col].fillna("not provided")
            df[col] = pd.Categorical(df[col])
            df[col] = df[col].cat.codes
            df[col] = df[col].astype("category")
    print("categorical variables processed")
    return df

def process_continuous_variables(df, variable_list, standardize=True):
    # standardize column data between 0 and 1
    for col in df:
        if col in variable_list:
            if col == "lei":
                df[col] = df[col].fillna("not provided")
                df[col] = pd.Categorical(df[col])
                df[col] = df[col].cat.codes
                df[col] = df[col].astype("category")

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
    drop_count = 0
    for col in df.columns:
        unique = set(list(df[col]))
        if len(unique) == 1:
            del df[col]
            drop_count = drop_count + 1
    print(str(drop_count) +" variables with low variance removed")
    return df


def check_variables_for_nulls(df):
    for col, null_count in zip(df.isnull(), df.isnull().sum()):
        if null_count > 0:
            print(col+" has null values!")


def pre_process_loan_data(df, cat_variables, cont_variables, standardize_cont=True):
    df = filter_valid_outcomes(df)
    df = filter_null_columns(df)
    df = filter_low_variance(df)
    df = process_categorical_variables(df, cat_variables)
    df = process_continuous_variables(df, cont_variables, standardize=standardize_cont)
    df = df.dropna()
    check_variables_for_nulls(df)
    find_high_corr(df)
    return df


def limit_data(df, n=20000):
    df = df.head(n)
    return df


def back_to_df(X):
    return pd.DataFrame(X.toarray())


def get_x_and_y(df, dep_variable):
    y = df[[dep_variable]]
    y = y.values.ravel()
    df = df.drop(dep_variable, axis=1)
    return df, y


## Create Function to Print Results
def get_results(x1):
    print("\n{0:20}   {1:4}    {2:4}".format('Model','Train','Test'))
    print('-------------------------------------------')
    for i in x1.keys():
        print("{0:20}   {1:<6.4}   {2:<6.4}".format(i,x1[i][0],x1[i][1]))


def get_train_test_data(X, y, features=False):
    if features:
        X = X[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, X


def feature_selection(df, y, n=500, num_features="best"):
    df = df.head(n)
    y = y[:n]
    X_train, X_test, y_train, y_test, X = get_train_test_data(df, y)
    sfs = SFS(LogisticRegression(max_iter=10000),
              k_features=num_features,
              forward=True,
              floating=False,
              scoring = 'accuracy',
              n_jobs=-1,
              cv = 4)
    sfs.fit(X, y)
    print("feature selection score: ", sfs.k_score_)
    print("SFS chosen features: ", sfs.k_feature_names_)
    return list(sfs.k_feature_names_)



