import os
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib as plt

dirname = os.path.dirname(__file__)

#Part 1

def sampler(n = 1000000, theta = np.array([3,1,2]),Dist_x1 = (3,4), Dist_x2 = (-1,4), Dist_eps = (0,2)):
    X = np.array([[1]*n,np.random.normal(Dist_x1[0],np.sqrt(Dist_x1[1]),n),np.random.normal(Dist_x2[0],np.sqrt(Dist_x2[1]),n)])
    Y = theta @ X + np.random.normal(Dist_eps[0],np.sqrt(Dist_eps[1]),n)
    return((X,Y))


#Part 2

def SGD(epochs=100,eta=0.001,batch_size=[100],k=1000):
    X0,Y0 = sampler()
    generated_theta = []
    print(f"Starting time: {datetime.datetime.now()}")
    for r in batch_size:
        print(f"Running for batch size: {r}")
        theta = np.zeros(X0.shape[0])
        epoch = 0
        no_of_batches = 0
        is_converged = False
        begin = time.time()
        while epoch < epochs:
            combined_data = np.vstack([X0,Y0])
            rand_shuffle = (np.random.permutation(combined_data.T)).T  #Produces the same order permuation for both x and y
            X1,Y1 = rand_shuffle[:-1],rand_shuffle[-1] #Shuffled X and Y
            epoch += 1
            J_sum = 0 #This holds the current moving sum 
            J_avg = 0 #This will hold the old moving avg
            for batch in range(Y0.size//r):
                X = X1[:,r*batch:r*(batch+1)] #Taking r columns (ie r training examples)
                Y = Y1[r*batch:r*(batch+1)] #Taking r samples
                diff = Y - theta @ X
                theta = theta + eta*(1/len(Y))*(diff @ X.T)
                J_sum += (1/(2*len(Y))) * (np.sum(diff**2))
                no_of_batches += 1
                if ((batch+1)%k == 0): #For moving average we have chose to compute at every k number of iterations
                    J1_avg = J_sum/k
                    if abs(J1_avg - J_avg)<1e-3:
                        is_converged = True
                        break
                    J_avg = J1_avg
                    J_sum = 0
            if is_converged:
                break
        generated_theta.append((r,time.time()-begin,epoch,theta))
        print(f"    Theta obtained: {theta}\n    Number of epochs taken: {epoch}\n    Number of batches taken: {no_of_batches}\n    Time taken: {time.time()-begin}")
    return(generated_theta)

'''
Obtained Result for Termination condition - epochs < 100 :

For absolute diff of moving avg < 1e-3:
    Running for batch size: 1
        Theta obtained: [2.9759403  1.00344442 2.03538132]
        Number of epochs taken: 1
        Number of batches taken: 30000
        Time taken: 0.46651625633239746
    Running for batch size: 100
        Theta obtained: [2.97287722 1.00576369 1.99738041]
        Number of epochs taken: 2
        Number of batches taken: 18000
        Time taken: 0.40770888328552246
    Running for batch size: 10000
        Theta obtained: [2.81313327 1.04114584 1.98577257]
        Number of epochs taken: 100
        Number of batches taken: 10000
        Time taken: 13.17662525177002
    Running for batch size: 1000000
        Theta obtained: [0.24510686 0.91737519 0.46577189]
        Number of epochs taken: 100
        Number of batches taken: 100
        Time taken: 13.477739810943604

    
For absolute diff of moving avg < 1e-5:
    Running for batch size: 1
        Theta obtained: [3.02593379 0.99806822 1.99051211]
        Number of epochs taken: 16
        Number of batches taken: 15992000
        Time taken: 182.60618233680725
    Running for batch size: 100
        Theta obtained: [2.99944606 1.0025087  2.00027943]
        Number of epochs taken: 100
        Number of batches taken: 1000000
        Time taken: 20.810455799102783
    Running for batch size: 10000
        Theta obtained: [2.81382034 1.04045622 1.9865465 ]
        Number of epochs taken: 100
        Number of batches taken: 10000
        Time taken: 13.05558466911316
    Running for batch size: 1000000
        Theta obtained: [0.24439259 0.91539028 0.46487172]
        Number of epochs taken: 100
        Number of batches taken: 100
        Time taken: 12.776516437530518

For absolute diff of moving avg < 1e-7
    Running for batch size: 1
        Theta obtained: [3.04958896 0.97003056 2.05811228]
        Number of epochs taken: 23
        Number of batches taken: 22356000
        Time taken: 260.3874611854553
    Running for batch size: 100
        Theta obtained: [3.00230606 0.99478821 2.0005821 ]
        Number of epochs taken: 100
        Number of batches taken: 1000000
        Time taken: 21.614763975143433
    Running for batch size: 10000
        Theta obtained: [2.8195499  1.03915331 1.98722234]
        Number of epochs taken: 100
        Number of batches taken: 10000
        Time taken: 14.125328540802002
    Running for batch size: 1000000
        Theta obtained: [0.24520819 0.91707902 0.46642046]
        Number of epochs taken: 100
        Number of batches taken: 100
        Time taken: 13.447947978973389

Questions - Why is batch_size of 10000 always taking 100 epochs?
'''

#Test Area
# x = np.array([[1,2,3],[4,5,6]])
# y = np.array([7,8,9])
# print(y.shape,y.size)
# comb = np.vstack([x,y])
# rand = np.random.permutation(comb.T).T
# x1,y1 = rand[:-1],rand[-1]

# print(abs(np.mean(x-x1))+abs(np.mean(y-y1)))
SGD(batch_size=[1,100,10000,1000000])