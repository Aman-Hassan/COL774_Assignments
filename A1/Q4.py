import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d


dirname = os.path.dirname(__file__)

#input handling
inputdf = (pd.read_csv(os.path.join(dirname,'data/q4/q4x.dat'),sep=r'\s\s+',header=None,engine='python')).to_numpy().T
outputdf = (pd.read_csv(os.path.join(dirname,'data/q4/q4y.dat'),sep=r'\s\s+',header=None,engine='python')).to_numpy().T

inputdf = ((inputdf.T-np.mean(inputdf,axis=1))/np.std(inputdf,axis=1)).T #normalize data
input_X = np.vstack([np.ones(inputdf.shape[1]),inputdf]) #Gives input X as 3*100 matrix
input_Y = np.where(outputdf == 'Alaska',0,1)[0] #Gives input Y as 1*100 matrix and Alaska is converted to 0 and Canada is converted to 1


#Part 1
def Linear_GDA(X = input_X,Y = input_Y):
    m0 = np.sum(X[1:,Y==0],axis=1)/(Y.shape[0]-np.sum(Y)) # mean when y = 0
    m1 = np.sum(X[1:,Y==1],axis=1)/np.sum(Y) #mean when y = 1
    A = np.where(Y==0,m0.reshape(-1,1),m1.reshape(-1,1)) #We have to convert m0 and m1 to row vectors
    S = (1/Y.shape[0]) * ((X[1:] - A) @ (X[1:] - A).T) #Covariance Matrix when S1 = S2 = S
    # print(f"m0: {m0}")
    # print(f"m1:{m1}")
    # print(f"S: \n{S}")
    return((m0,m1,S))

'''
Output:
m0: [-0.75529433  0.68509431]
m1: [ 0.75529433 -0.68509431]
S:
    [ 0.42953048 -0.02247228]
    [-0.02247228  0.53064579]
'''

#Part 2
def Plot_data(X = input_X,Y = input_Y):
    x1,x2 = X[1:]
    y = Y
    s = plt.scatter(x1,x2,c=y,cmap=ListedColormap(["green","blue"]))
    plt.title('Scatter plot')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(handles=s.legend_elements()[0],labels=['Alaska','Canada'])
    plt.show()

#Part 3
def Plot_Linear_Boundary(X = input_X, Y = input_Y):
    #Plotting points
    x1,x2 = X[1:]
    y = Y
    s = plt.scatter(x1,x2,c=y,cmap=ListedColormap(["green","blue"]))
    plt.title('Linear Boundary plot')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(handles=s.legend_elements()[0],labels=['Alaska','Canada'])

    #Plotting Linear boundary
    phi = (1/Y.shape[0])*(np.sum(Y)) #Bernoulli parameter
    m0,m1,S = Linear_GDA()
    const = -0.5*(m1 @ np.linalg.inv(S) @ m1.T - m0 @ np.linalg.inv(S) @ m0.T) + math.log(phi/(1-phi)) #The constant term
    theta = ((m1-m0) @ np.linalg.inv(S))
    plot_x = np.linspace(-2,2,30)
    plt.plot(plot_x,(-const - theta[0] * plot_x)/(theta[1]),c = "r")
    plt.show()

#Part 4
def General_GDA(X = input_X, Y = input_Y):
    m0 = np.sum(X[1:,Y==0],axis=1)/(Y.shape[0]-np.sum(Y)) # mean when y = 0
    m1 = np.sum(X[1:,Y==1],axis=1)/np.sum(Y) #mean when y = 1
    S0 = ((1/(Y.shape[0]-np.sum(Y))) * ((X[1:,Y==0] - m0.reshape(-1,1)) @ (X[1:,Y==0] - m0.reshape(-1,1)).T)) #Covariance Matrix S0
    S1 = ((1/np.sum(Y)) * ((X[1:,Y==1] - m1.reshape(-1,1)) @ (X[1:,Y==1] - m1.reshape(-1,1)).T)) #Covariance Matrix S0
    # print(f"m0: {m0}")
    # print(f"m1: {m1}")
    # print(f"S0: \n{S0}")
    # print(f"S1: \n{S1}")
    return((m0,m1,S0,S1))

'''
Output:
m0: [-0.75529433  0.68509431]
m1: [ 0.75529433 -0.68509431]
S0: 
    [ 0.38158978 -0.15486516]
    [-0.15486516  0.64773717]
S1: 
    [ 0.47747117 0.1099206 ]
    [ 0.1099206  0.41355441]
'''

#Part 5
def Plot_Quad_Boundary(X = input_X, Y = input_Y):
    #Plotting points
    x1,x2 = X[1:]
    y = Y
    s = plt.scatter(x1,x2,c=y,cmap=ListedColormap(["green","blue"]))
    plt.title('Quadratic Boundary plot')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(handles=s.legend_elements()[0],labels=['Alaska','Canada'])

    #Plotting Quadratic boundary
    phi = (1/Y.shape[0])*(np.sum(Y)) #Bernoulli parameter
    m0,m1,S0,S1 = General_GDA()
    
    #Method-1: Hard Code - direct usage of quadratic formula
    A = np.linalg.inv(S1) - np.linalg.inv(S0)
    theta = -2*( m1@np.linalg.inv(S1) - m0@np.linalg.inv(S0) )
    const = m1.T@np.linalg.inv(S1)@m1 - m0.T@np.linalg.inv(S0)@m0 - 2*math.log(math.sqrt(np.linalg.det(S1)/np.linalg.det(S0))*(phi/(1-phi)))
    plot_x1 = np.linspace(-2,2,30)
    a = A[1,1]
    b = (A[1,0]+A[0,1])*plot_x1 + theta[1]
    c = A[0,0]*(np.square(plot_x1)) + theta[0]*plot_x1 + const
    plot_x2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a) 
    plt.plot(plot_x1,plot_x2,c = 'red')
    plt.show()

#Part 5

def Plot_both_boundaries(X = input_X, Y = input_Y):
    x1,x2 = X[1:]
    y = Y
    s = plt.scatter(x1,x2,c=y,cmap=ListedColormap(["green","blue"]))
    plt.title('Quadratic Boundary plot')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(handles=s.legend_elements()[0],labels=['Alaska','Canada'])

    #Plotting Linear boundary
    phi = (1/Y.shape[0])*(np.sum(Y)) #Bernoulli parameter
    m0,m1,S = Linear_GDA()
    const = -0.5*(m1 @ np.linalg.inv(S) @ m1.T - m0 @ np.linalg.inv(S) @ m0.T) + math.log(phi/(1-phi)) #The constant term
    theta = ((m1 - m0) @ np.linalg.inv(S))
    plot_x = np.linspace(-2,2,30)
    plt.plot(plot_x,(-const - theta[0] * plot_x)/(theta[1]),c = "orange")

    #Plotting Quadratic boundary
    phi = (1/Y.shape[0])*(np.sum(Y)) #Bernoulli parameter
    m0,m1,S0,S1 = General_GDA()
    
    #Method-1: Hard Code - direct usage of quadratic formula
    A = np.linalg.inv(S1) - np.linalg.inv(S0)
    theta = -2*( m1@np.linalg.inv(S1) - m0@np.linalg.inv(S0) )
    const = m1.T@np.linalg.inv(S1)@m1 - m0.T@np.linalg.inv(S0)@m0 - 2*math.log(math.sqrt(np.linalg.det(S1)/np.linalg.det(S0))*(phi/(1-phi)))
    plot_x1 = np.linspace(-2,2,30)
    a = A[1,1]
    b = (A[1,0]+A[0,1])*plot_x1 + theta[1]
    print(A[0,0],theta[0],const)
    c = A[0,0]*(plot_x1)**2 + theta[0]*plot_x1 + const
    plot_x2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a) 
    plt.plot(plot_x1,plot_x2,c = 'red')
    plt.show()

# Plot_both_boundaries()
