import numpy as np

# Import procedure 
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', remove=(['headers','footers', 'quotes']))

# Procedure to check th name of the data set 
#from pprint import pprint
#pprint(list(newsgroups_train.target_names))
