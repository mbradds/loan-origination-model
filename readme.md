# Mortgage Loan Solution
This project contains the code for an automated machine learning approach to a mortgage loan origination problem. This readme file explains the problem, how to run the model, and a discussion of the ethical considerations of using machine learning for mortgages.

## Problem description
You work for the Consumer Financial Protection Bureau, the US agency that regulates financial institutions. The VP of Loan Regulation has an upcoming presentation to financial institutions in Illinois. The VP says that in Illinois it takes on average 30.5 days to get a mortgage (from loan application to decision), which carries a cost of $13 billion annually to consumers and financial institutions. The VP is interested in testing a regulatory sandbox, requiring financial institutions to provide a fast approval process for certain applicants, to reduce the administrative burden. Based on data submitted by financial institutions in 2020, in the state of Illinois, as part of the Home Mortgage Disclosure Act (state_IL_applicant.csv and Data fields.docx), you are asked to assess whether:
1. A model can be built, and what features should be included, and a discussion of performance (report performance using X_test.csv);
2.	A model should be built, and a discussion of privacy and bias.

### Notes
- There are over 500,000 loans in the training dataset.
- The dependent variable is the loan outcome.
- There are over 50 independent variables such as loan amount, income, etc which may be used to predict the dependent variable.

## Problem solution

A loan approval or rejection has major implications in a person's life, and loan decisions carry significant regulatory and legal implications. Only a highly accurate model should be selected.
Based on the problem and the dataset, two models were trained and evaluated. 
1. logistic regression with stepwise forward selection
2. Logistic regression LASSO with cross validation

Model 2 produced high (>0.99) accuracy, precision, recall, and AUC scores in the model test data. This indicates that the model will likely perform very well on real loans outside of the provided dataset. Model 2 has several characteristics that apply nicely to this problem. Features in the dataset that are not significantly associated with the loan outcome are not included. Also with cross validation, multiple models are evaluated to further refine accuracy on unseen data. Both of these models select independent variables (features) automatically. This is important because after data preperation, there are over 200 features in the dataset, which makes is infeasible to manually identify which features are best suited for predicting the loan outcome.

While these preliminary results are encouraging, lets put things in perspective. 
Mortgage lenders issued 2.71 million residential loans in the first three months of 2022 across the US. A model that is 99% accurate would have made the incorrect loan decision just over 27 thousand times. That means that 27 thousand people and their families would either be given a loan that they can't afford (default, eviction), or qualified applicants would be rejected, and their dream of home ownership and financial independence could be irreversibly damaged. 

So, we know a model can be built, but its not 100% clear if it would be the right thing to do. The increased speed and potential for cheaper loans would undoubtedly benefit consumers, however, the 27 thousand errors each quarter are a concern.

The consumer financial protection bureau will only support a production model if it furthers its mandate by performing at least as well, or better than human loan officers. It may be possible to use modern statistical methods to significantly reduce bias in the loan origination process. Like many institutions, the financial system continues to struggle with bias and discrimination. In fact, as recently as September of this year, the Justice Department announced a settlement for lending discrimination on the basis of race, sex and national origin.  


Given the potential for errors in the machine learning model, a hybrid model/approach should be taken. For example, the algorithm could be used only for loan pre-approvals, or as an assistant to human loan officers, but doesn’t make the full decision without human supervision. Any use of a model would need to include a program to handle complaints.

## Steps for running the code:

1. Create a sub folder called `/data` in the project root and add the following files `/data/state_IL_application.csv` and `/data/X_test.xlsx`.
2. Make sure that both `model_gm.ipynb` and `util.py` are in the project root. The `model_gm.ipynb` imports code from `util.py`.
3. Create the virtual environment specified in `environment.yml`. In the project root, run: `conda env create -f environment.yml`
4. Open jupyter notebook, navigate to `model_gm.ipynb` and run!


Model output is saved in `X_test_predicted_gm.csv`
