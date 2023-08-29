import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation


dirname = os.path.dirname(__file__)

#Imput handling
#Taking input data using pandas and converting to numpy arrays (for faster indexing)
inputdf = pd.read_csv(os.path.join(dirname,'data\\q1\\linearX.csv'),header=None)
linearX = pd.Series(inputdf[0]).to_numpy()
outputdf = pd.read_csv(os.path.join(dirname,'data\\q1\\linearY.csv'),header=None)
linearY = pd.Series(outputdf[0]).to_numpy()

#Normalising input data to 0 mean and unit variance
linearX-=np.mean(linearX)
linearX/=np.std(linearX)

#Adding x0 to all input
LinearX = np.array([np.ones(len(linearX)),linearX])


# Part 1
'''
General Algorithm for Linear Regression:
    t = 0
    param (theta) = init() [In the present question it is initialized to 0 vector]
    eta = 0.1 (for starting purposes)
    while |J(t+1)-J(t)| > 1e^-10 (some small bound){
        param(t+1) = param(t) - eta*del(J(t))
        t = t+1
    }
    return (param)


Initialising parameters aka Theta:
    theta0 = (np.ones(2)*(-np.inf)) #Arbritarily low value of theta0 for convergence puposes
    theta = (np.zeros(2))
    eta = 0.1 #Starting learning rate -> will update to check which value gives better convergence

Gradient Descent Calculation:
    Note1: .T is shorthand for transpose 
    Note2: in numpy 1D array, transpose doesnt explicitly work so reshaping had to be done -> to avoid too many reshapes theta was taken as row vector instead of the traditional column vector

    Working outwards for the gradient calculation:
    i) y - h_theta(x) = linearY - np.dot(theta,LinearX) --> 1*m row vector

    Then computing next part can be done in two ways:
    iia) Analytically by taking each y[i] - h_theta(x[i]) and multiplying with x[i] then taking sum of this vector
    iib) Using matrix manipulation and doing np.dot(X,above.T) -> this would give us the nx1 column vector needed --> This is going to be used
'''

def GradientDescent(X,Y):
    #Gradient Descent
    for eps in [1e-8,1e-12,1e-16,1e-20]:
        print("When termination condition ->",eps)
        for eta in [0.1,0.025,0.001]:
            theta0 = (np.ones(2)*(-np.inf)) #Arbritarily low value of theta0 for convergence puposes
            theta = (np.zeros(2))
            t=0
            while abs(np.sum(theta - theta0)) > eps:
                theta0 = theta
                diff = Y - theta @ X
                theta = theta + eta*(1/len(Y))*(diff @ X.T)
                t += 1
            print("    When eta =",eta,"-> Converges after",t+1,"iterations to",theta)
    return theta


'''
Observations for value of eta:
When termination condition -> 1e-08
    When eta = 0.1 -> Converges after 155 iterations to [0.99662001 0.0013402 ]
    When eta = 0.025 -> Converges after 584 iterations to [0.99661971 0.0013402 ]
    When eta = 0.001 -> Converges after 11508 iterations to [0.99661013 0.00134018]
When termination condition -> 1e-12
    When eta = 0.1 -> Converges after 243 iterations to [0.9966201 0.0013402]
    When eta = 0.025 -> Converges after 948 iterations to [0.9966201 0.0013402]
    When eta = 0.001 -> Converges after 20713 iterations to [0.9966201 0.0013402]
When termination condition -> 1e-16
    When eta = 0.1 -> Converges after 335 iterations to [0.9966201 0.0013402]
    When eta = 0.025 -> Converges after 1331 iterations to [0.9966201 0.0013402]
    When eta = 0.001 -> Converges after 30389 iterations to [0.9966201 0.0013402]
When termination condition -> 1e-20
    When eta = 0.1 -> Converges after 336 iterations to [0.9966201 0.0013402]
    When eta = 0.025 -> Converges after 1331 iterations to [0.9966201 0.0013402]
    When eta = 0.001 -> Converges after 30389 iterations to [0.9966201 0.0013402]
'''


def GD_withJ(X,Y):
    for eps in [1e-8,1e-12,1e-16,1e-20]:
        print("When termination condition ->",eps)
        for eta in [0.1,0.025,0.001]:
            J = 0
            theta = np.zeros(2)
            t = 0
            while True:
                diff = Y - theta @ X
                theta = theta + eta*(1/len(Y))*(diff @ X.T)
                J1 = (1/(2*len(Y))) * (np.sum(diff**2))
                if abs(np.sum(J1-J)) < eps:
                    break
                J = J1
                t += 1
            print("    When eta =",eta,"-> Converges after",t+1,"iterations to",theta)
    return(theta)

'''
Observations for a given value of eta:
When termination condition -> 1e-08
    When eta = 0.1 -> Converges after 79 iterations to [0.99637817 0.00133987]
    When eta = 0.025 -> Converges after 293 iterations to [0.99602185 0.00133939]
    When eta = 0.001 -> Converges after 5752 iterations to [0.99346351 0.00133595]
When termination condition -> 1e-12
    When eta = 0.1 -> Converges after 122 iterations to [0.99661749 0.00134019]
    When eta = 0.025 -> Converges after 475 iterations to [0.99661413 0.00134019]
    When eta = 0.001 -> Converges after 10355 iterations to [0.99658854 0.00134015]
When termination condition -> 1e-13
    When eta = 0.1 -> Converges after 133 iterations to [0.99661928 0.00134019]
    When eta = 0.025 -> Converges after 520 iterations to [0.99661819 0.00134019]
    When eta = 0.001 -> Converges after 11506 iterations to [0.99661012 0.00134018]
When termination condition -> 1e-16
    When eta = 0.1 -> Converges after 166 iterations to [0.99662007 0.0013402 ]
    When eta = 0.025 -> Converges after 657 iterations to [0.99662004 0.0013402 ]
    When eta = 0.001 -> Converges after 14958 iterations to [0.99661978 0.0013402 ]
'''


#Part 2 (Plotting hypothesis function)
def ScatterPlot_Hypothesisfunc(X,Y,LinX,theta):
    plt.scatter(X,Y)
    # plt.legend("Datapoints")
    # plt.title("Scatter Chart")
    plt.title('Hypothesis function vs. Inputs (LR = 0.001)')
    plt.xlabel('Wine Acidity (x)')
    plt.ylabel('Wine Density (y)')
    plt.plot(X,theta @ LinX,color="green")
    plt.show()



#Part3 3D Plot of J(theta)
'''
Matrix Calulation of J(theta):
- theta is a matrix of the size n*s^2 (s is number of samples, n is number of parameters)
- J should be a matrix of the size 1*s^2 (ie s*s pairs of theta are used)
- Now for each pair of value in theta (a pair of theta is given by a column), we multiply this pair with the input X (which is of size n*m) and then this produces a 
intermediate matrix of size 1*m which is then subtracted from Y (another 1*m size matrix). This on summing up for each element and squaring (also divided by 1/2m) 
gives J(theta)
- We do this for all columns in theta (ie all possible pairs of values of theta0 and theta1)

Going step by step - 
i) theta.T * X1 gives an s^2*m matrix where each row consists of a particular pair of theta used on a particular x(i)
ii) (i).sum(axis=1) would give a s^2*1 matrix where each row is now the sum of theta used on all x(i)'s
iii) np.sum(Y) - (ii) would give the required sigma(Y-theta*X) for all the pairs of theta
iv) 1/(2m) * np.squar((iii)) finally gives J(theta) for all pairs of theta [it is a s^2*1 matrix]
'''

def Error_plotter(X,Y):
    theta0 = np.linspace(-6.0,6.0,30) #Creates a linear space from -6 to 6 with 30 samples
    theta1 = np.linspace(-6.0,6.0,30) #Creates a linear space from -6 to 6 with 30 samples
    theta0,theta1 = np.meshgrid(theta0,theta1) #Creates a mesh grid using the above
    X1 = [[1]*len(X),X]

    #Now we need to calculate J(theta) at each of the pairs of values from theta0 and theta1
    theta = np.array([np.ravel(theta0),np.ravel(theta1)])
    J = 1/(2*len(Y))*(((Y - theta.T @ X1)**2).sum(axis=1)) 

    #Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta0, theta1, J.reshape(theta0.shape),cmap="viridis")

    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('Cost function')

    # anim = FuncAnimation(fig,animate,interval=200)
    plt.show()

def animate(i):
    pass





# theta = GradientDescent(LinearX,linearY)
# ScatterPlot_Hypothesisfunc(linearX,linearY,LinearX,theta)

# theta = GD_withJ(LinearX,linearY)
Error_plotter(linearX,linearY)



# TESTING AREA: 
# def fun(x, y):
#     return x**2 + y

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = y = np.arange(-3.0, 3.0, 3)
# X, Y = np.meshgrid(x, y)
# zs = np.array(fun(np.ravel(X), np.ravel(Y)))
# Z = zs.reshape(X.shape)

# print(X.shape,Y.shape,Z.shape)
# ax.plot_surface(X, Y, Z)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

