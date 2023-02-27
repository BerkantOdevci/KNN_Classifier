#######################################################################
#   KNN Algorithm
#######################################################################
#
#   @Class Name(s): CustCat_Class
#
#   @Description:   KNN Algorithm
#
#
#   @Note:  Predict data, calculate accuracy and compare neighbors in KNN
#
#   Version 0.0.1:  CustCat_Class()
#                   array_value(self)
#                   train_test_split(self)
#                   prediction(self)
#                   accuracy_evaluation(self)
#                   other_neighbors(self)
#                   16 Feb 2023 Wednesday, 14:30 - Hasan Berkant Ödevci
#
#
#
#   @Author(s): Hasan Berkant Ödevci
#
#   @Mail(s):   berkanttodevci@gmail.com
#
#   it is created on 16 Feb 2023 Wednesday, at 14:30.
#
#
########################################################################

# Library
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
except ImportError:
    print("Check the library...")

class CusCat_Class():
    def __init__(self):
        self.df = pd.read_csv("D:/Projects/Python/Machine_Learning/KNN/Dataset/teleCust1000t.csv")

    def array_value(self):
        # Assign the values and labels into array
        self.x= self.df[["region","tenure","age","marital","address","income","ed","employ","retire","gender","reside"]].values
        self.y = self.df["custcat"].values

        # Data Normalize
        self.x = preprocessing.StandardScaler().fit(self.x).transform(self.x.astype(float))

    def train_test_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=0)

        # Print shape of x_train/test and y_train/test
        print("x_train shape is {}".format(self.x_train.shape))
        print("y_train shape is {}".format(self.y_train.shape))
        print("x_test shape is {}".format(self.x_test.shape))
        print("y_test shape is {}".format(self.y_test.shape))
    
    def prediction(self):
        # Define KNN classifiers
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(self.x_train,self.y_train)
        self.prediction = self.knn.predict(self.x_train)
        self.y_hat = self.knn.predict(self.x_test)

        print("Prediction on train set is {}".format(self.prediction))
        print("------------------------")
        print("Prediction on test set is {}".format(self.y_hat))

    def accuracy_evaluation(self):
        # Print Accuracy
        print("Train set Accuracy: ",metrics.accuracy_score(self.y_train,self.prediction))
        print("Test set Accuracy: ",metrics.accuracy_score(self.y_test,self.y_hat))

    def other_neighbors(self):
        k = 10
        mean_accuracy = np.zeros([k-1])
        std_accuracy = np.zeros([k-1])

        print(mean_accuracy)
        print(std_accuracy)

        for i in range(1,k):
            # Train Model and Predict
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(self.x_train,self.y_train)
            y_hat = knn.predict(self.x_test)
            mean_accuracy[i-1] = metrics.accuracy_score(self.y_test,y_hat)
            std_accuracy[i-1] = np.std(y_hat == self.y_test)/np.sqrt(y_hat.shape[0])

        # Plot which is right neighbors in KNN algorithm and the value of accuracy
        plt.plot(range(1,k),mean_accuracy,'g')
        plt.fill_between(range(1,k),mean_accuracy - 1 * std_accuracy,mean_accuracy + 1 * std_accuracy, alpha=0.10)
        plt.fill_between(range(1,k),mean_accuracy - 3 * std_accuracy,mean_accuracy + 3 * std_accuracy, alpha=0.10,color="green")
        plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
        plt.ylabel('Accuracy ')
        plt.xlabel('Number of Neighbors (K)')
        plt.tight_layout()
        plt.show()

        # Print The best accuracy value and K neighbor value
        print("The best accuracy was with", mean_accuracy.max(), "with k=", mean_accuracy.argmax()+1) 

# Create an object
model = CusCat_Class()

# Assign Attributes
model.array_value()

model.train_test_split()

model.prediction()

model.accuracy_evaluation()

model.other_neighbors()