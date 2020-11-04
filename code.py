import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#extracting data
df = pd.read_csv("/Users/vineethkondisetty/Downloads/Project1_vxk2021/iris.data", names=["sepal_length","sepal_width","petal_length","petal_width","species"],header = None)
#print(df)
df[df.species=='Iris-setosa'].plot(kind='scatter',x='sepal_length', y='sepal_width',color='r')
df[df.species=='Iris-versicolor'].plot(kind='scatter',x='sepal_length', y='sepal_width',color='b')
df[df.species=='Iris-virginica'].plot(kind='scatter',x='sepal_length', y='sepal_width',color='g')
plt.figure(figsize=(8,8))
sns.set_style('whitegrid')
sns.FacetGrid(df,hue='species',height=5).map(plt.scatter,'petal_length','petal_width').add_legend()
plt.show()
plt.figure(figsize=(8,8))
sns.set_style('whitegrid')
sns.FacetGrid(df,hue='species',height=5).map(plt.scatter,'sepal_length','sepal_width').add_legend()
plt.show()
sns.violinplot(x='species',y='sepal_length',data=df)
plt.show()
sns.violinplot(x='species',y='sepal_width',data=df)
plt.show()
sns.violinplot(x='species',y='petal_length',data=df)
plt.show()
sns.violinplot(x='species',y='petal_width',data=df)
plt.show()
#print(df)
df = df.replace( {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2 })
print('\n')
print("Dataset is as follows:")
#print('\n')
print(df)
df = df.to_numpy()
#print(df)
array = np.delete(df, 4,axis=1)
X = array
Y= df[:,-1]

#print (X)
#print('\n')
#print(Y)
#print('\n')
#print(df)

#calculate beta
def betacap(X,Y):
    xt = np.transpose(X)        # Transpose of X
    xtx = np.dot(xt,X)          # Transpose of X * X
#print(xtx)
    xtxinv = inv(xtx)           # Inverse of (Transpose of X * X)
    xty = np.dot(xt,Y)          # Transpose of X * Y
#print(xty)
    beta = np.dot(xtxinv,xty)   # Betacap = (Inverse of (Transpose of X * X)) * (Transpose of X * Y)
    return beta


def crossvalidation(folds):
    accuracy_list = []
    fold = int(len(X)/folds) 
    X_split = np.array_split(X,folds)
    Y_split = np.array_split(Y,folds)
    print('\n')
    print("------------------------------------------------------------------------------")
    print("Confusion matrix for "+str(folds)+" fold cross-validation is as follows: ")
    for i in range(folds):
        X_test = np.array(X_split[i])
        Y_test = np.array(Y_split[i])
        X_train = np.delete(X_split,i,0)
        X_train = np.concatenate(X_train)
        #print(X_train)
        Y_train = np.delete(Y_split,i,axis=0)
        Y_train = np.concatenate(Y_train)
        #print(Y_train)
        betavar = betacap(X_train,Y_train)
        #print(betavar)
        exp_Y = np.dot(X_test,betavar)
        exp_Y = exp_Y.round()
        exp_Y = abs(exp_Y)
        results = confusion_matrix(Y_test, exp_Y)
        print(results)
        print('\n')
        #print( accuracy_score(Y_test, exp_Y))
        print( classification_report(Y_test, exp_Y))
        #print("Predicted")
        #print(exp_Y)
        #print("actual")
        #print(Y_test)
        count = 0
        for item in np.equal(Y_test,exp_Y):
            if item == True:
                count+=1
        accuracy = count/fold
        accuracy_list.append(accuracy)
    #print((sum(accuracy_list)/len(accuracy_list))*100)    
    return ((sum(accuracy_list)/len(accuracy_list))*100)
   
for folds in range(2,8):
    #crossvalidation(folds)
    print ("Accuracy after "+str(folds)+" fold cross validation is: {}".format(crossvalidation(folds)))
    


def classification(sl,sw,pl,pw):
    sl = float(sl)
    sw = float(sw)
    pl = float(pl)
    pw = float(pw)
    if sl >= 3.2 and sl<=5.8 and sw >=2.3 and sw <=4.4 and pl >= 0.8 and pl<=2.0 and pw >= 0.1 and pw<=0.7:
        print("Flower may be IRIS-SETOSA")
    elif sl >= 4.5 and sl<= 7 and sw >= 2 and sw <= 3.4 and pl >= 3 and pl<= 4.8 and pw >= 0.9 and pw<=1.6:
        print("Flower may be IRIS-VERSICOLOR")
    elif sl >= 4.8 and sl<= 7.8 and sw >= 2.2 and sw <= 3.8 and pl >= 4.9 and pl<= 7 and pw >= 1.7 and pw<= 2.6:
        print("Flower may be IRIS-VIRGINICA")
    else:
        print("Iris Flower NOT recognized")
    #values() 
    
def values():
    print("------------------------------------------------------------------------------")
    print("Enter inputs below to know the type")
    sl = input("Enter Sepal Length:")
    sw = input("Enter Sepal width:")
    pl = input("Enter Petal Length:")
    pw = input("Enter Petal width:")
    print('\n')
    print("Given Inputs")
    print("Sepal Length: "+sl+" Sepal Width: "+sw+" Petal Length: "+pl+" Petal Width: "+pw+" ")
    classification(sl, sw, pl, pw)

values()

  
# MULTIVARIATE LINEAR REGRESSION ALGORITHM #
print("-------------------------------------------------------------------------------------------")
print("Using Multivariate Regression Algorithm, (Without using Linear Regression model formula)")
rows = 150
Xb = np.hstack(((np.ones((rows,1))), X))
Yr = Y.reshape(rows,1)
np.random.seed(0)
theta = np.random.randn(1,5)
print("Theta : %s" % (theta))
iteration = 100000
learning_rate = 0.005
J = np.zeros(iteration)   # Cost_function
for i in range(iteration):
    J[i] = (1/(2 * rows) * np.sum((np.dot(Xb, theta.T) - Yr) ** 2 ))
    theta -= ((learning_rate/rows) * np.dot((np.dot(Xb, theta.T) - Yr).reshape(1,rows), Xb))

prediction = np.round(np.dot(Xb, theta.T))
accuracy = (sum(prediction == Yr)/float(len(Yr)) * 100)[0]
print("The Mutli-variate Linear Regression model predicted values with an overall accuracy of %s at 100000 iterations and learning rate of 0.005" % (accuracy))  


