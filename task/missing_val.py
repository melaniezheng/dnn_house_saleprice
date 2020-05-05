import numpy as np
import pandas as pd
import pickle
from scipy.stats import boxcox
import random

##################################################
############### Data Pipeline ####################
##################################################

PROJ_DIR = "~/Projects/predicting_house_saleprice/"
np.random.seed(1234)
def impute_missing_values(stage):
    # loads data, remove outliers, impute missing values.
    # inpute stage(str): 'train' or 'test'
    # outputs df and opt_lmbda to perform boxcox transformation for saleprice based on training data.

    ############### Remove Outliers ###############
    # Create a function to remove outliers identified in EDA
    # Removed more outliers based on OverallQual, GarageArea, GrLivArea. These do not 
    #           follow linear relationship with Boxcox(SalePrice)
    def remove_outliers(df):
        BoxcoxPrice, _ = boxcox(df['SalePrice']) # boxcox transformation on target variable, saleprice.
        df['BoxCoxPrice']=BoxcoxPrice
        #df = df.drop(df[(df['OverallQual']==4) & (df['BoxCoxPrice']> 8)].index) # where overall quality is less than averge yet saleprice is extremely large
        #df = df.drop(df[(df['GarageArea']>1200) & (df['BoxCoxPrice']<7.6)].index) # garage Area is large yet saleprice is very low
        df = df.drop(df[(df['GrLivArea']>4000) & (df['BoxCoxPrice']<8)].index) # living area is large yet sale price is low
        #df = df.drop(df[(df['OverallCond']==2) & (df['BoxCoxPrice']>8)].index) # overall condition is low yet price is high
        df = df.drop(columns='BoxCoxPrice')
        return df


    ############### Impute Data ###############
    # Create a function handles imputation of most pseudo-missing and missing values (other than the features related to basements, lot frontage, and garages)
    def impute_pseudo(df):
        # Impute pseudo-missing values with None
        df['Alley'] = df['Alley'].fillna('None')
        df['Fence'] = df['Fence'].fillna('None')
        df['MiscFeature'] = df['MiscFeature'].fillna('None')
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
        df.loc[df['Fireplaces']==0, 'FireplaceQu'] = 'None' # if Fireplace is not 0, that means there is fireplace and we can impute the FireplaceQu.
        df.loc[np.logical_and(df['MasVnrArea']==0, df['MasVnrType'].isnull()), 'MasVnrType'] = 'None'
        df.loc[df['PoolArea']==0, 'PoolQC'] = 'None'
        
        # Some variables in pseudo-missing columns actually represent missing observations (as can be told from feature variables relating to the same category of housing feature)
        # Impute as the mode if the missing value represents a missing observation, impute as "No X" if it does not
        
        df.loc[np.logical_and(df['PoolArea']!=0, df['PoolQC'].isnull()), 'PoolQC'] = 'Gd' # we will use this to generate IsPool but PoolQC column will be dropped in the end.
        len=df['LotConfig'].isnull().sum()
        try:
            df.loc[df['LotConfig'].isnull(),'LotConfig'] = random.choices(k=len,population=['Corner', 'CulDSac', 'FR2','Inside'], weights=[0.18, 0.06, 0.03,0.72])# impute with mode
        except:
            pass
        df.loc[np.logical_and(df['Fireplaces']!=0, df['FireplaceQu'].isnull()), 'FireplaceQu'] = \
            random.choice(['Gd','TA']) # randomly choose between top 2 most frequent FireplaceQu from training data
        df.loc[np.logical_and(df['MasVnrArea']!=0, df['MasVnrType'].isnull()), 'MasVnrType'] = \
            random.choices(population=['BrkCmn', 'BrkFace', 'Stone'], weights=[0.05, 0.75, 0.2]) # randomly choose with respective probabily distribution from training data.
        # Impute true missing values with mode imputation
        df['Electrical'] = df['Electrical'].fillna('SBrkr') # impute with mode


    # Create a function to impute misssing values for the feature variables related to basements.
    def impute_basements_garages(df):
        col_list = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','GarageCond', 'GarageType', 'GarageFinish', 'GarageQual']
        num_col_list = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','GarageYrBlt', 'GarageCars', 'GarageArea']
        
        # impute basement as null.
        for col in col_list:
            df[col]=df[col].fillna('None')
        for col in num_col_list:
            df[col]=df[col].fillna(0)



    # Create a function to re-engineer categorical feature variables for ease of dummification.
    def impute_categorical(df):
        df['Condition1'] = df['Condition1'].apply(lambda x: "Norm" if x == "Norm" else "Other")
        df['LotShape'] = df['LotShape'].apply(lambda x: "Reg" if x == "Reg" else "IReg")
        df['Functional'] = df['Functional'].apply(lambda x: "Y" if x=="Y" else "N")
        df['Electrical'] = df['Electrical'].apply(lambda x: "SBrkr" if x=='SBrkr' else 'Other')
        df['RoofMatl'] = df['RoofMatl'].apply(lambda x: "Other" if x != "CompShg" else x)
        df['RoofStyle'] = df['RoofStyle'].apply(lambda x: "Other" if (x !="Gable" and x != "Hip") else x)
        df['Heating'] = df['Heating'].apply(lambda x: "GasA" if x == "GasA" else 'Other')
        df['Foundation'] = df['Foundation'].apply(lambda x: "Other" if (x !="PConc" and x != "CBlock" and x != "BrkTil") else x)
        df['SaleType'] = df['SaleType'].apply(lambda x: "Other" if (x != "WD" and x != "New" and x != "COD") else x)
        df['Exterior1st'] = df['Exterior1st'].apply(lambda x: "Other" if (
            x !="VinylSd" and x != "MetalSd" and x != "HdBoard" and x != "Wd Sdng" and x != "Plywood") else x)
        len = df['MSZoning'].isnull().sum()
        if len > 0:
            df.loc[df['MSZoning'].isnull(), 'MSZoning']=random.choices(k=len, population=\
                ['C (all)', 'FV', 'RH', 'RL', 'RM'], weights=[0.01, 0.04, 0.01, 0.79, 0.15])
        len = df['Exterior2nd'].isnull().sum()
        if len > 0:
            df.loc[df['Exterior2nd'].isnull(), 'Exterior2nd'] = random.choices(k=len, population=\
                ['AsbShng', 'AsphShn', 'Brk Cmn', 'BrkFace', 'CBlock', 'CmentBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'Wd Shng'], \
                    weights=[0.01, 0.0, 0.0, 0.02, 0.0, 0.04, 0.14, 0.01, 0.15, 0.0, 0.1, 0.0, 0.02, 0.35, 0.13, 0.03]) # distribution from training data
        len = df['KitchenQual'].isnull().sum()
        try:
            if len > 0:
                df.loc[df['KitchenQual'].isnull(),'KitchenQual'] = random.choices(k=len, population=\
                    ['Ex', 'Gd', 'TA'], weights=[0.1, 0.4, 0.5])
        except Exception as e:
            print(e)

    def impute_data(df):
        # function that combines all the above functions.
        try:
            df=remove_outliers(df)
        except:
            pass
        impute_pseudo(df)
        impute_basements_garages(df)
        impute_categorical(df)
        return df

    ############### Import Data ###############
    if stage == 'train':
        df = pd.read_csv(PROJ_DIR + '/data/raw/train.csv')
    elif stage == 'test':
        df = pd.read_csv(PROJ_DIR + '/data/raw/test.csv')
    else:
        raise ValueError("please provide valid stage. 'train' or 'test'.")

    return impute_data(df)


