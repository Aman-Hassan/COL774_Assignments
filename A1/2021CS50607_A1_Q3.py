import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Imput handling
#Taking input data using pandas and converting to numpy arrays (for faster indexing)
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'relative/path/to/file/you/want')

inputdf = pd.read_csv(os.path.join(dirname, 'data\\q3\\logisticX.csv'),header=None)
logisticX = pd.DataFrame(inputdf).to_numpy().T
outputdf = pd.read_csv(os.path.join(dirname, 'data\\q3\\logisticY.csv'),header=None)
logisticY = pd.DataFrame(outputdf).to_numpy().T


#Normalising input data to 0 mean and unit variance
for i in range(logisticX.shape[0]):
    logisticX[i]-=np.mean(logisticX[i])
    logisticX[i]/=np.std(logisticX[i])

#Adding x0 to all input
X = np.vstack([[1]*logisticX.shape[1],logisticX])
Y = logisticY

#Part 1

def sigmoid(c):
    return(1/(1+np.exp(-c)))

def Grad(X,Y,t):
    diff = Y - sigmoid(t @ X) #1*m row vector
    grad = (1/(X.shape[1]))*(diff @ X.T) #1*n vector
    return(grad)

def Hessian(X,Y,t):
    D = np.diag(sigmoid(t @ X)*(1-sigmoid(t @ X))) #m*m vector
    H = (1/(X.shape[1]))*(X @ D @ X.T) #n*n vector
    return -H

'''
General Algorithm for logistic regression:
    theta = init()
    t = 0
    J(theta) = 0
    while |J(t+1) - J(t)| > 1e-8{
        theta = theta - H^-1 * (Grad(J))
        t = t+1
    }
    return theta

Calculation of Hessian:
    - By manual calulation we know that H(jk) = np.sum(sigmoid(1-sigmoid)*xj*xk)
    - By observing the required matrixes X and X.T, I made a new diagonal matrix that has the value of sigmoid(1-sigmoid) for all i's
    since this would be same for data point (ie it would be same for all features in x(i) for any i).
    - After some manipution, I derived the required matrix formula of (X @ D) @ X.T where D was the diagonal matrix
'''

def Logistic_regression(X,Y):
    for eps in [1e-8,1e-12,1e-16,1e-20]:
        print("When termination condition ->",eps)
        theta = np.zeros(X.shape[0])
        J = 0
        t = 0
        while True:
            theta_old = theta
            theta = theta - ((Grad(X,Y,theta) @ np.linalg.inv(Hessian(X,Y,theta))))[0]
            if abs(np.sum(theta-theta_old)) < 1e-12:
                break
            t+=1
        print("    Newton's method converges after",t+1,"iterations to the value",theta)
    return theta
'''
Output obtained:
When termination condition -> 1e-08
    Newton's method converges after 9 iterations to the value [ 0.40125316  2.5885477  -2.72558849]
When termination condition -> 1e-12
    Newton's method converges after 9 iterations to the value [ 0.40125316  2.5885477  -2.72558849]
When termination condition -> 1e-16
    Newton's method converges after 9 iterations to the value [ 0.40125316  2.5885477  -2.72558849]
When termination condition -> 1e-20
    Newton's method converges after 9 iterations to the value [ 0.40125316  2.5885477  -2.72558849]
'''


def Plot(X1,Y1,logistic_X,theta):
    x1,x2 = logistic_X
    y = Y1[0]
    colormap = np.array(['b','g'])
    s = plt.scatter(x1,x2,c=y,cmap=ListedColormap(["green","blue"]))
    #New inputs on which hypothes
    input_x = np.linspace(-3.0,3.0,30)
    input_x_1 = np.vstack([[1]*input_x.shape[0],input_x])
    plt.plot(input_x,(theta[:2] @ input_x_1)/(-theta[2]),c = "r")

    plt.title('Scatter plot and Decision Boundary')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(handles=s.legend_elements()[0],labels=[0,1])
    plt.show()



#Testing Area
# X1 = np.array([[1,1,1],[1,2,3],[2,3,4]])
# # Y1 = np.array([1,2,3])
# t1 = np.array([0,0,0])
# D = np.diag(sigmoid(t1 @ X1)*(1-sigmoid(t1 @ X1)))
# print (X1 @ D @ X1.T)

t = Logistic_regression(X,Y)
Plot(X,Y,logisticX,t)

