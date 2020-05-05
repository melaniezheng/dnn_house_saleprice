import numpy as np
import pandas as pd
import pickle
from scipy.stats import boxcox
import random
from missing_val import impute_missing_values

##################################################
############### Feature Engineering ####################
##################################################

def generate_feat(stage, dummify = False):
    ### map ordinal features to numeric numbers, dummify feature, generate new features, drop features.
    ### input - stage(str) - 'train' or 'test'
    ### output - df of final features
    def knn_lotfrontage(df):
        columns = ['LotConfig','LotArea','Street','Neighborhood','LandContour']
        feat = df.loc[df['LotFrontage'].isnull(), columns]

        col_list=['LotConfig','Street','Neighborhood','LandContour']
        temp = feat.copy() 
        with open('./pkl/ohe_lotfrontage.pkl', 'rb') as f:
            ohe= pickle.load(f) # load one hot encoder trained using training data
        enc = ohe.transform(temp[col_list])
        enc = pd.DataFrame(enc, columns=ohe.get_feature_names(col_list))
        feat = pd.concat((temp.drop(col_list, axis=1).reset_index(drop=True), enc), axis=1)
       
        with open('./pkl/knn_lotfrontage.pkl', 'rb') as f:
            knn=pickle.load(f)  # load knn regressor trained using training data
        df.loc[df['LotFrontage'].isnull(), 'LotFrontage'] = knn.predict(feat)

    def categorical_to_ordinal(df):
        # function to convert categorical to ordinal features. These will act like a numeric features. 
        ord_list = ['ExterCond','ExterQual','BsmtQual','BsmtCond', 'HeatingQC','KitchenQual', \
            'FireplaceQu','GarageFinish', 'GarageQual', 'GarageCond','BsmtFinType1']
        for col in ord_list:
            df.loc[df[col].isin(['Ex','ALQ','GLQ']), col] = 10
            df.loc[df[col].isin(['Gd','GdPrv','BLQ']), col] = 8
            df.loc[df[col].isin(['TA','Av','MnPrv','Fin','Rec','Y']), col] = 6
            df.loc[df[col].isin(['Fa','Mn','GdWo','RFn','LwQ','P']), col] = 4
            df.loc[df[col].isin(['Po','No','MnWw','Unf','N']), col] = 2
            df.loc[df[col] == 'None', col] = 0


    ############### Feature Engineering/Creation ###############
    
    def add_features(df):
        # Create a function to add new feature variables based on variables already in the data
        df['IsPool']=df['PoolQC'].apply(lambda x: 1 if x !="None" else 0)
        df['IsGarage']=df['GarageYrBlt'].apply(lambda x: 0 if x==0 else 1)
        df['IsFireplace']=df['FireplaceQu'].apply(lambda x: 0 if x=='None' else 1)
        df['TotalFullBath'] = df['BsmtFullBath'] + df['FullBath']
        df['TotalHalfBath'] = df['BsmtHalfBath'] + df['HalfBath']
        df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
        # Dummify features related to what type of porch (if any) the house had by hard encoding zeros and ones
        df['3SsnPorch'] = df['3SsnPorch'].apply(lambda x: 1 if x>0 else 0)
        df['ScreenPorch'] = df['ScreenPorch'].apply(lambda x: 1 if x>0 else 0)
        df['EnclosedPorch'] = df['EnclosedPorch'].apply(lambda x: 1 if x>0 else 0)
        df['IsOpenPorch'] = df['OpenPorchSF'].apply(lambda x: 1 if x>0 else 0)
    
    ############### Bucketize  ###############
    def bucketize_features(df):
        cols = ['YearBuilt','YearRemodAdd']
        with open('./pkl/kbd_yr.pkl', 'rb') as f:
            kbins = pickle.load(f)
        # make a copy of the df
        temp = df.copy()
        names = ['YrBuilt_'+str(int(np.round(i)))for i in kbins.bin_edges_[0][1:]]
        YrRemodAdd_colname=['YrRemodAdd_'+str(int(np.round(i)))for i in kbins.bin_edges_[0][1:]]
        names.extend(YrRemodAdd_colname) # generate colnames based on original colname and bins
        enc = pd.DataFrame(kbins.transform(temp[cols]).toarray(), columns=names)
        temp = pd.merge(df, enc, left_index=True, right_index=True)
        return temp.drop(columns=cols)

    ############## Feature Dummification ###############
    def dummify_features(df):
        # Create a function to dummify the categorical variables for use in regression
        # Dummify all other necessary features using One Hot Encoding
        col_list = ['MSSubClass','MoSold', 'YrSold', 'Exterior1st', 'Exterior2nd', 'Condition1', 'LotShape', 
        'Functional', 'Electrical', 'RoofMatl', 'RoofStyle', 'Heating', 'Foundation', 'SaleType', "LandContour", 'MSZoning',
        'Street', 'Alley', 'HouseStyle', 'BldgType', 'LandSlope', 'LotConfig', 'Neighborhood', 
        'GarageType', 'PavedDrive',  'Fence', 'MasVnrType', 'CentralAir', 'SaleCondition', 'BsmtExposure',
        'Condition2', 'BsmtFinType2', 'PoolQC', 'MiscFeature']

        temp = df.copy()
        with open('./pkl/ohe.pkl', 'rb') as f:
            ohe=pickle.load(f) # load trained one hot encoder
        enc = ohe.transform(temp[col_list]) 
        enc = pd.DataFrame(enc, columns=ohe.get_feature_names(col_list))
        temp = pd.concat((temp.drop(col_list, axis=1).reset_index(drop=True), enc), axis=1)
        return temp


    ############### Feature Removal ###############
    # Create a function to remove features that are either not useful for prediction (ex. 'Id') or have been rendered redundant by the above steps
    def remove_features(df):
        df = df.drop(columns=['Id', 'Utilities'])#,'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
                # 'BsmtHalfBath','FullBath','HalfBath','OpenPorchSF','Condition2', #"Exterior2nd", "GarageQual",
                # 'PoolQC', 'MiscFeature','BsmtFinType2','GarageYrBlt'])
        return df

    ############### Data Processing and Saving Files ###############
    # Create a function that handles all data processing steps above at once:
    def process_data(df, dummify):
        knn_lotfrontage(df)
        categorical_to_ordinal(df)
        add_features(df)
        if dummify:
            df = bucketize_features(df)
            df = dummify_features(df)
        df = remove_features(df)
        return df


    ############# call impute missing values function ###############
    df = impute_missing_values(stage)

    return process_data(df, dummify)