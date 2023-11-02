import numpy as np 
import sys
import pdb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder

def get_data(x_path, y_path):
    '''
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    '''
    x = np.load(sys.argv[1])
    y = np.load(sys.argv[2])

    y = y.astype('float')
    x = x.astype('float')

    #normalize x:
    x = 2*(0.5 - x/255)
    return x, y

def get_metric(y_true, y_pred):
    '''
    Args:
        y_true: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
        y_pred: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
                
    '''
    results = classification_report(y_pred, y_true)
    print(results)


class perceptron:
    def __init__(self, layer: int,batch_size: int, features: int, learning_rate: str, activation: str):
        self.batch_size = batch_size
        self.features = features
        self.learning_rate = learning_rate
        self.activation = activation
        self.layer = layer


class neural_network:
    def __init__(self,batch_size: int, features: int, hidden: list[int], target: int, learning_rate: str, activation: str, method: str, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
        self.batch_size = batch_size
        self.features = features           #1024*(features==None) + features*(features!=None) #Chooses 1024 if features is not none, else the value passed
        self.hidden = hidden
        self.target = target               #5*(target==None) + target*(target!=None) #Chooses 5 if target is not none, else the value passed
        self.learning_rate = learning_rate #'Const'*(learning_rate==None) + learning_rate*(learning_rate!=None)
        self.activation = activation       #'sigmoid'*(activation==None) + activation*(activation!=None)
        self.method = method               #'backprop'*(method==None) + method*(method!=None)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.parameters = np.empty() # Would need to update with number of parameters required -> need to calculate ig
        self.loss = np.empty() # Would need to update size -> need to calculate ig

    def create_network(self):
        if self.method == "backprop":
            self.backpropogation()
        else:
            MLPClassifier(hidden_layer_sizes=self.hidden,activation='relu',solve='sgd',alpha=0,batch_size=32,learning_rate='invscaling')
        pass

    def backpropogation(self): # aka train the network
        pass

    def predict(self): #Predict on the test set
        pass

if __name__ == '__main__':

    x_train_path = sys.argv[1]
    y_train_path = sys.argv[2]

    X_train, y_train = get_data(x_train_path, y_train_path)

    x_test_path = sys.argv[3]
    y_test_path = sys.argv[4]

    X_test, y_test = get_data(x_test_path, y_test_path)

    #you might need one hot encoded y in part a,b,c,d,e
    label_encoder = OneHotEncoder(sparse_output = False)
    label_encoder.fit(np.expand_dims(y_train, axis = -1))

    y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))
    y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis = -1))
