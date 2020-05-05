from generate_features import generate_feat
import pandas as pd
import sys

#### use dummify for sklearn models and keras models.
#### tensorflow had tf.feature_column.categorical_column_with_vocabulary_list for dummifying categorical
#### no need for dummification when using tensorflow.
#### Running script ...
#### first argument: Train or Test
#### second argument: True or False for dummifying categorical columns.
#### i.e. process_data.py Train True 
#### outputs train_processed.csv and test_processed.csv in data/processed/

try:
    if sys.argv[1] == 'Train':
        if sys.argv[2] == 'True': # dummify categorical columns
            train = generate_feat('train', dummify = True)
            train.to_csv('../data/processed/train_processed.csv', index = False)
        elif sys.argv[2] == 'False': # dummify categorical columns:
            train = generate_feat('train', dummify = False)
            train.to_csv('../data/processed/train_processed.csv', index = False)
        else:
            print('Provide True or False for second argument. Dummify - True, else False')
    elif sys.argv[1] == 'Test':
        if sys.argv[2] == 'True': # dummify categorical columns
            train = generate_feat('test', dummify = True)
            train.to_csv('../data/processed/test_processed.csv', index = False)
        elif sys.argv[2] == 'False': # dummify categorical columns:
            train = generate_feat('test', dummify = False)
            train.to_csv('../data/processed/test_processed.csv', index = False)
        else:
            print('Provide True or False for second argument. Dummify - True, else False')
    else:
        print('Provide Train or Test as the first argument')
except Exception as e:
    print(e)