from Logistic_Regression import Logistic_Regression
from Naive_Bayes import Naive_Bayes

#================ Test Logistic_Regression ==================#

### Analytical Solution:

# lg1 = Logistic_Regression("ionosphere", 'A')
# lg2 = Logistic_Regression("adult", 'A')
# lg3 = Logistic_Regression("breast-cancer", 'A')
# lg4 = Logistic_Regression("bank", 'A')

# Run .predict() for a single time test
# lg1.predict()
# lg2.predict()
# lg3.predict()
# lg4.predict()

# ### Run .kfoldCrossValidation() for k-fold CV accuracy score
# lg1.kfoldCrossValidation(5)
# lg2.kfoldCrossValidation(5)
# lg3.kfoldCrossValidation(5)
# lg4.kfoldCrossValidation(5)


### GD Solution:

lg5 = Logistic_Regression("ionosphere", 'G')
lg6 = Logistic_Regression("adult", 'G')
lg7 = Logistic_Regression("breast-cancer", 'G')
lg8 = Logistic_Regression("bank", 'G')

# ### Run .predict() for a single time test
lg5.predict()
lg6.predict()
lg7.predict()
lg8.predict()

# ### Run .kfoldCrossValidation() for k-fold CV accuracy score
# lg5.kfoldCrossValidation(5)
# lg6.kfoldCrossValidation(8)
# lg7.kfoldCrossValidation(5)
# lg8.kfoldCrossValidation(8)

#================ Test Naive_Bayes ==================#

# lga = Naive_Bayes("ionosphere")
# lgb = Naive_Bayes("adult")
# lgc = Naive_Bayes("breast-cancer")
# lgd = Naive_Bayes("bank")

## Run .predict() for a single time test
# lga.predict()
# lgb.predict()
# lgc.predict()
# lgd.predict()

### Run .kfoldCrossValidation() for k-fold CV accuracy score
# lga.kfoldCrossValidation(5)
# lgb.kfoldCrossValidation(5)
# lgc.kfoldCrossValidation(5)
# lgd.kfoldCrossValidation(5)
