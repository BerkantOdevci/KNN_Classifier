#######################################################################
#   KNN Algorithm
#######################################################################
#
#   @Class Name(s): Iris_Class
#
#   @Description:   KNN Algorithm
#
#
#   @Note:  Predict data, calculate accuracy and compare neighbors in KNN
#
#   Version 0.0.1:  Iris_Class()
#                   shape_of_data(self)
#                   look_data(self)
#                   prediction(self)
#                   evaluation(self)
#                   compare_neighbors(self)
#                   16 Feb 2023 Wednesday, 13:30 - Hasan Berkant Ödevci
#
#
#
#   @Author(s): Hasan Berkant Ödevci
#
#   @Mail(s):   berkanttodevci@gmail.com
#
#   it is created  on 16 Feb 2023 Wednesday,  at 13:30.
#
#
########################################################################

# Libraries
try:
    import numpy as np
    import pandas as pd
    import mglearn
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
except ImportError:
    print("Check the library...")

# Load Dataset
iris_dataset = load_iris()

# Show key features and data shape
print("Iris datasets keys = {}".format(iris_dataset.keys()))
print("-----------------------")
print("Data of iris dataset shape is {}".format(iris_dataset["data"].shape))
print("-----------------------")

# Seperate train and test set
x_train,y_train,x_test,y_test = train_test_split(iris_dataset["data"], iris_dataset["target"],random_state=0)


class Iris_Class():
    def __init__(self,irish_data,x_train,y_train,x_test,y_test):
        self.irish_data =irish_data
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def shape_of_data(self):
        # Define x_train/test and y_train/test shape
        print("X_train shape is {}".format(self.x_train.shape))
        print("Y_train shape is {}".format(self.y_train.shape))
        print("X_test shape is {}".format(self.x_test.shape))
        print("Y_test shape is {}".format(self.y_test.shape))

    def look_data(self):
        iris_dataframe = pd.DataFrame(self.x_train, columns=self.irish_data.feature_names)
        graph = pd.plotting.scatter_matrix(iris_dataframe, c = self.y_train, figsize=(20,20), marker="o", hist_kwds={"bins": 20}, s = 60, alpha=.8, cmap = mglearn.cm3)
        plt.show()

    def prediction(self):
        # Create KNN
        self.knn = KNeighborsClassifier(n_neighbors=4)
        # Fit x_train set and y_train set 
        self.knn.fit(self.x_train,self.y_train)

        prediction = self.knn.predict(self.x_train)

        #print("Prediction target name: {}".format(self.irish_data["target_names"][prediction]))
        print("Training accuracy is {:3f}".format(self.knn.score(self.x_train,self.y_train)))
    
    def evaluation(self):
        # Evaluation between train set and test set
        y_pred = self.knn.predict(self.x_test)
        print("Test set predictions are {}".format(y_pred))

        # Test Score
        print("Test set score: {:.3f}".format(np.mean(y_pred == self.y_test)))

    def compare_neighbors(self):
        
        # Create list for training and test accuracy
        training_accuracy = []
        test_accuracy = []

        # Set n_neighbors with range(1,11)
        n_neightbors = range(1,11)

        for n_neighbor in n_neightbors:
            knn = KNeighborsClassifier(n_neighbors=n_neighbor)
            knn.fit(self.x_train,self.y_train)
            training_accuracy.append(knn.score(self.x_train,self.y_train))
            test_accuracy.append(knn.score(self.x_test,self.y_test))
        
        plt.plot(n_neightbors,training_accuracy,label = "training accuracy")
        plt.plot(n_neightbors,test_accuracy,label = "test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()
        plt.show()           

if __name__ == '__main__':     
    # Create an object
    model = Iris_Class(iris_dataset,x_train,x_test,y_train,y_test)

    # Assign attributes
    #model.shape_of_data()

    #model.look_data()

    model.prediction()

    model.evaluation()

    model.compare_neighbors()


