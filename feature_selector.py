import numpy as np
from numpy import ndarray
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

ref_categories = ['sub_grade:G', 'home_ownership:ANY_MORTGAGE',  'verification_status:Not Verified',  'purpose:home_other_debt_moving_medical',  'loan_amnt:>32201', 'term:60', 'int_rate:>25.855', 'installment:>1071.177', 'annual_inc:>298505', 'dti:>39.9522', 'fico_range_low:>771.0', 'fico_range_high:>738.4', 'age:>60', 'pay_status:>0.2']



class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, X):
        self.X = X
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # sub_grade
        X_new = X.loc[:, 'sub_grade:A':'sub_grade:G']
        
        # home ownership
        X_new['home_ownership:OWN'] = X.loc[:, 'home_ownership:OWN']
        X_new['home_ownership:RENT'] = X.loc[:, 'home_ownership:RENT']
        X_new['home_ownership:OTHER'] = X.loc[:, 'home_ownership:OTHER']
        X_new['home_ownership:NONE'] = X.loc[:, 'home_ownership:NONE']
        X_new['home_ownership:ANY_MORTGAGE'] = sum([X['home_ownership:ANY'], X['home_ownership:MORTGAGE']])
        
        # verification_status
        X_new = pd.concat([X_new, X.loc[:, 'verification_status:Not Verified':'verification_status:Verified']], axis=1)
        
        # purpose
        X_new['purpose:small_business'] = X.loc[:, 'purpose:small_business']
        ## Let's combine features with low WoE
        X_new['purpose:home_other_debt_moving_medical'] = sum([X['purpose:home_improvement'], X['purpose:other'], X['purpose:debt_consolidation'], X['purpose:moving']])
        
        # addr_state - inf IV let's not consider it

        # loan_amnt
        X_new['loan_amnt:<12701'] = np.where((X['loan_amnt'] <= 12701), 1, 0)
        X_new['loan_amnt:12701-24401'] = np.where((X['loan_amnt'] > 12701) & (X['loan_amnt'] <= 24401), 1, 0)
        X_new['loan_amnt:24401-32201'] = np.where((X['loan_amnt'] > 24401) & (X['loan_amnt'] <= 32201), 1, 0)
        X_new['loan_amnt:>32201'] = np.where((X['loan_amnt'] > 32201), 1, 0)

        # term
        X_new['term:36'] = np.where((X['term'] == 36), 1, 0)
        X_new['term:60'] = np.where((X['term'] == 60), 1, 0)

        # int_rate
        X_new['int_rate:<13.015'] = np.where((X['int_rate'] <= 13.015), 1, 0)
        X_new['int_rate:13.015-20.719'] = np.where((X['int_rate'] > 13.015) & (X['int_rate'] <= 20.719), 1, 0)
        X_new['int_rate:20.719-25.855'] = np.where((X['int_rate'] > 20.719) & (X['int_rate'] <= 25.855), 1, 0)
        X_new['int_rate:>25.855'] = np.where((X['int_rate'] > 25.855), 1, 0)

        # installment
        X_new['installment:<327.987'] = np.where((X['installment'] <= 327.987), 1, 0)
        X_new['installment:327.987-1071.177'] = np.where((X['installment'] > 327.987) & (X['installment'] <= 1071.177), 1, 0)
        X_new['installment:>1071.177'] = np.where((X['installment'] > 1071.177), 1, 0)

        # annual_inc
        X_new['annual_inc:missing'] = np.where(X['annual_inc'].isnull(), 1, 0)
        X_new['annual_inc:<75357'] = np.where((X['annual_inc'] <= 75357), 1, 0)
        X_new['annual_inc:75357-161183'] = np.where((X['annual_inc'] > 75357) & (X['annual_inc'] <= 161183), 1, 0)
        X_new['annual_inc:161183-195513'] = np.where((X['annual_inc'] > 161183) & (X['annual_inc'] <= 195513), 1, 0)
        X_new['annual_inc:195513-247009]'] = np.where((X['annual_inc'] > 195513) & (X['annual_inc'] <= 247009), 1, 0)
        X_new['annual_inc:247009-264174]]'] = np.where((X['annual_inc'] > 247009) & (X['annual_inc'] <= 264174), 1, 0)
        X_new['annual_inc:264174-281339]]'] = np.where((X['annual_inc'] > 264174) & (X['annual_inc'] <= 281339), 1, 0)
        X_new['annual_inc:281339-298505]]'] = np.where((X['annual_inc'] > 281339) & (X['annual_inc'] <= 298505), 1, 0)
        X_new['annual_inc:>298505]]'] = np.where((X['annual_inc'] > 298505), 1, 0)

        # dti
        X_new['dti:<19.977'] = np.where((X['dti'] <= 19.977), 1, 0)
        X_new['dti:19.977-37.455'] = np.where((X['dti'] > 19.977) & (X['dti'] <= 37.455), 1, 0)
        X_new['dti:37.455-39.9522'] = np.where((X['dti'] > 37.455) & (X['dti'] <= 39.9522), 1, 0)
        X_new['dti:>39.9522'] = np.where((X['dti'] > 39.9522), 1, 0)

        # fico_range_low
        X_new['fico_range_low:<697.01'] = np.where((X['fico_range_low'] <= 697.01), 1, 0)
        X_new['fico_range_low:697.01-771.0'] = np.where((X['fico_range_low'] > 697.01) & (X['fico_range_low'] <= 771.0), 1, 0)
        X_new['fico_range_low:>771.0'] = np.where((X['fico_range_low'] > 771.0), 1, 0)

        # fico_range_high
        X_new['fico_range_high:<691.9'] = np.where((X['fico_range_high'] <= 691.9), 1, 0)
        X_new['fico_range_high:691.9-738.4'] = np.where((X['fico_range_high'] > 691.9) & (X['fico_range_high'] <= 738.4), 1, 0)
        X_new['fico_range_high:>738.4'] = np.where((X['fico_range_high'] > 738.4), 1, 0)

        # mort_acc - do not consider it

        # age
        X_new['age:<30'] = np.where((X['age'] <= 30), 1, 0)
        X_new['age:30-60'] = np.where((X['age'] > 30) & (X['age'] <= 60), 1, 0)
        X_new['age:>60'] = np.where((X['age'] > 60) , 1, 0)
        
        # pay_status
        X_new['pay_status:<-0.9'] = np.where((X['pay_status'] <= -0.9), 1, 0)
        X_new['pay_status:-0.9-0.2'] = np.where((X['pay_status'] > -0.9) & (X['pay_status'] <= 0.2), 1, 0)
        X_new['pay_status:>0.2'] = np.where((X['pay_status'] > 0.2), 1, 0)
        return X_new
    
