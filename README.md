### Prediction house saleprice using Deep Neural Networks.

Motivation: 

I led a team of three to participate in the kaggle challenge house-prices-advanced-regression-techniques. We tried various machine learning techniques including linear estimators, regularized linear estimator (lasso, ridge, elastic net), GradientBoost, XGBoost, CatBoost, and Stacked regressor.  We built a data pipe line to clean data, impute missing values, add new features, one hot encode categorical features, and drop unhelpful features to preprocess the data that's going into the models. That was our first ever machine learning project and our prediction ranked top 14% on the Leadership board. <br> Here is a link to the github repo. https://github.com/melaniezheng/predicting_house_saleprice

Since then, I've been obssessed with various machine learning techniques, reading a lot about data processing and machine learning techning techniques, especially deep neural networks. I took some deep learning course offered by deeplearning.ai on Coursera and wanted to put what I've learned into an actual project. Since I am already familiar with the housing data, I will be building the neural network to predict house saleprice using the same kaggle dataset. I'm interested in finding out how is the neural net performance compared to boosting tree and random forest models.

#### Updates to data processing pipeline:
- remove only 2 outliers related to GrLivArea.
- impute data from the knowledge of training data only and applied the same process to test data.
- used random imputation for categorical features with specified weights (from training data distribution).
- build one hot encoder using training data only for data processing.
- use knn to impute LotFrontage using the knowledge from training data only.
- Did not drop these features: BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','OpenPorchSF','Condition2', #"Exterior2nd", "GarageQual",'PoolQC', 'MiscFeature','BsmtFinType2','GarageYrBlt'
- Bucketized these two features: 'YearBuilt','YearRemodAdd'. (We dummified them previously)

#### Deep Neural Net Models(w/ Keras):
Here are the models I will be trying in this project.
- Basic Neural Net: X -> dim(X) -> Y. Note: dim(X) is really the number of features. My data had 257 features after preprocessing.
- Wide/Narrow Neural Net: X --> N --> Y, where N is less than or greater than 257, respectively.
- Deep Neural Nets: X --> N --> N --> N --> Y. I will experiment with different N here and discover what works best for my specific dataset. 

#### Hyperparameter tuning techniques: 
- batch_size
- epoch 
- early stopping
- dropout
- batch normalization
