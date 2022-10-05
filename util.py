import os
import pandas as pd
import numpy as np


dependent_variable = "action_taken"

categorical_variables = ["activity_year",
                         "lei",
                         "derived_msa-md",
                         "state_code",
                         "county_code",
                         "census_tract",
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
                        "interest_rate",
                        "rate_spread",
                        "total_loan_costs",
                        "total_points_and_fees",
                        "origination_charges",
                        "discount_points",
                        "lender_credits",
                        "loan_term",
                        "prepayment_penalty_term",
                        "loan_to_value_ratio",
                        "intro_rate_period",
                        "property_value",
                        "multifamily_affordable_units",
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
    
    
def filter_valid_outcomes(df):
    df = df[df["action_taken"].isin([1, 3])].copy()
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
    return df

def process_continuous_variables(df, variable_list):
    # standardize column data between 0 and 1
    for col in df:
        if col in variable_list:
            df[col] = df[col].replace({"Exempt": np.nan})
            # TODO look into an exempt/non exempt categorical variable
            df[col] = pd.to_numeric(df[col])
            df[col] = df[col].fillna(df[col].mean())
            max_value = df[col].max()
            min_value = df[col].min()
            df[col] = (df[col] - min_value) / (max_value - min_value)
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

