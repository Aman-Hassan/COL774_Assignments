import pandas as pd
import numpy as np
import matplotlib as plt


# Part 1
'''
General Algorithm for Linear Regression:
t = 0
param (theta) = init() [In the present question it is initialized to 0 vector]
eta = 0.1 (for starting purposes)
while J(t+1)-J(t) > 1e^-10 (some small bound){
    param(t+1) = param(t) - eta*del(J(t))
    t = t+1
}
return (param)
'''

#Taking input data using pandas and converting to numpy arrays (for faster indexing)
inputdf = pd.read_csv("./data/q1/linearX.csv",header=None)
linearX = pd.Series(inputdf[0]).to_numpy()
outputdf = pd.read_csv("./data/q1/linearY.csv",header=None)
linearY = pd.Series(outputdf[0]).to_numpy()

#Normalising input data to 0 mean and unit variance
linearX-=np.mean(linearX)
linearX/=np.std(linearX)

#Adding x0 to all input
LinearX = [[1]*len(linearX),linearX]

#Initialising parameters aka Theta
# theta0 = (np.ones(2)*(-np.inf)) #Arbritarily low value of theta0 for convergence puposes
# theta = (np.zeros(2))
# eta = 0.1 #Starting learning rate -> will update to check which value gives better convergence


#Gradient Descent
for eps in [1e-8,1e-12,1e-16,1e-18]:
    print("When termination condition >",eps)
    for eta in [0.1,0.01,0.001]:
        theta0 = (np.ones(2)*(-np.inf)) #Arbritarily low value of theta0 for convergence puposes
        theta = (np.zeros(2))
        t=0
        while abs(np.sum(theta - theta0)) > eps:
            theta0 = theta

            '''
            Note: .T is shorthand for transpose 
            Note2: in numpy 1D array, transpose doesnt explicitly work so reshaping had to be done -> to avoid too many reshapes theta was taken as row vector instead of the traditional column vector
            
            Working outwards for the gradient calculation:
            i) y - h_theta(x) = linearY - np.dot(theta,LinearX) --> 1*m row vector

            Then computing next part can be done in two ways:
            iia) Analytically by taking each y[i] - h_theta(x[i]) and multiplying with x[i] then taking sum of this vector
            iib) Using matrix manipulation and doing np.dot(X,above.T) -> this would give us the nx1 column vector needed --> This is going to be used
            '''

            diff = linearY - np.dot(theta,LinearX)
            theta = theta + ((eta*(1/len(linearY))*(np.dot(LinearX,diff.reshape(-1,1)))).reshape(1,-1))[0]
            # print(theta[0],theta[1])
            t += 1

        
        print("    When eta =",eta,"-> Converges after",t+1,"iterations to",theta)

'''
Observations for value of eta:
When termination condition > 1e-08
    When eta = 0.1 -> Converges after 155 iterations to [0.99662001 0.0013402 ]
    When eta = 0.01 -> Converges after 1377 iterations to [0.99661912 0.00134019]
    When eta = 0.001 -> Converges after 11506 iterations to [0.99661011 0.00134018]
When termination condition > 1e-12
    When eta = 0.1 -> Converges after 243 iterations to [0.9966201 0.0013402]
    When eta = 0.01 -> Converges after 2293 iterations to [0.9966201 0.0013402]
    When eta = 0.001 -> Converges after 20712 iterations to [0.9966201 0.0013402]
When termination condition > 1e-16
    When eta = 0.1 -> Converges after 335 iterations to [0.9966201 0.0013402]
    When eta = 0.01 -> Converges after 3257 iterations to [0.9966201 0.0013402]
    When eta = 0.001 -> Converges after 30389 iterations to [0.9966201 0.0013402]
'''



#Part 2 (Plotting hypothesis function)
# plt.scatter(linearX,linearY)
# plt.legend("Datapoints")
# plt.title