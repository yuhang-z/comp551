from Logistic_Regression import Logistic_Regression

lg1 = Logistic_Regression("ionosphere")
lg2 = Logistic_Regression("adult")
lg3 = Logistic_Regression("breast-cancer")
lg4 = Logistic_Regression("bank")

### Run .predict() for a single time test
# lg1.predict()
# lg2.predict()
# lg3.predict()
# lg4.predict()

### Run .kfoldCrossValidation() for k-fold CV accuracy score
lg1.kfoldCrossValidation(5)
lg2.kfoldCrossValidation(5)
lg3.kfoldCrossValidation(5)
lg4.kfoldCrossValidation(5)


