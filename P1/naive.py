#author: Olivia Xu

########################### --- Naive Bayes --- ##############################

#prior class probability
def prior(label,Y_data):
    num_result = 0

    for i in range(len(Y_data)):
        if label ==  Y_data[i]:
            num_result= num_result + 1

    return num_result/len(Y_data)

#print(adult_y[0])
#print(prior(b' <=50K',adult_y))

#likelihood of input features given the class label

def likelihood(label,features,X_data,Y_data):
    list_of_label = []
    for i in range(len(Y_data)):
        if Y_data[i]==label:
            list_of_label.append(i)
    X_datasubset = X_data[list_of_label,:]
    all = len(X_datasubset)+1
    multiply_result= 1
    of_the_feature = 0

    for i in range(len(features)):
        feature = features[i]

        for j in range(len(X_datasubset)):
            if X_datasubset[j,i]==feature:
                of_the_feature = of_the_feature+1

        term_to_multiply = of_the_feature/all
        of_the_feature = 0
        multiply_result = multiply_result*term_to_multiply

    return multiply_result

#fea = [b'no-recurrence-events',b'30-39',b'premeno',b'30-34',b'0-2',b'no',b'3',b'left']
#print(likelihood(b'no',fea,breast_X,breast_y))

#marginal probability of the input
def evidence():

def naive():
