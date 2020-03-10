# Author: Yuhang (10, Mar)
# compile environment: python3.8

# Libraries: 
import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups


# Import 20 groups train dataset
twenty_train = fetch_20newsgroups(subset='train', remove=(['headers','footers', 'quotes']), shuffle=True)

twenty_train_data = twenty_train.data

twenty_train_target = twenty_train.target


# Import 20 groups test dataset
twenty_test = fetch_20newsgroups(subset='test', remove=(['headers','footers', 'quotes']), shuffle=True)

twenty_test_data = twenty_test.data

twenty_test_target = twenty_test.target


# Import IMDB train dataset
IMDb_train = datasets.load_files("/Users/YuhangZhang/desktop/gh/comp551/P2/IMDB_train", description=None, categories=None, load_content=True, shuffle=True, encoding='utf-8', decode_error='strict', random_state=0)

IMDb_train_data = IMDb_train.data

IMDb_train_target = IMDb_train.target


# Import IMDB test dataset
IMDb_test = datasets.load_files("/Users/YuhangZhang/desktop/gh/comp551/P2/IMDB_test", description=None, categories=None, load_content=True, shuffle=True, encoding='utf-8', decode_error='strict', random_state=0)

IMDb_test_data = IMDb_test.data

IMDb_test_target = IMDb_test.target