import os
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



#Part 1

def sampler(n = 1000000, theta = np.array([3,1,2]),Dist_x1 = (3,4), Dist_x2 = (-1,4), Dist_eps = (0,2)):
    X = np.array([[1]*n,np.random.normal(Dist_x1[0],np.sqrt(Dist_x1[1]),n),np.random.normal(Dist_x2[0],np.sqrt(Dist_x2[1]),n)])
    Y = theta @ X + np.random.normal(Dist_eps[0],np.sqrt(Dist_eps[1]),n)
    return((X,Y))


#Part 2
def SGD(epochs=100,eta=0.001,batch_size=[100],k=1000):
    X0,Y0 = sampler()
    generated_theta = []
    theta_iterations = []
    # print(f"Starting time: {datetime.datetime.now()}")
    for r in batch_size:
        # print(f"Running for batch size: {r}")
        theta_r = []
        theta = np.zeros(X0.shape[0])
        theta_r.append(theta)
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
                theta_r.append(theta)
                if ((batch+1)%k == 0): #For moving average we have chose to compute at every k number of iterations
                    J1_avg = J_sum/k
                    if abs(J1_avg - J_avg)<1e-3:
                        is_converged = True
                        break
                    J_avg = J1_avg
                    J_sum = 0
            if is_converged:
                break
        theta_iterations.append(theta_r)
        generated_theta.append((r,time.time()-begin,epoch,theta))
        # print(f"    Theta obtained: {theta}\n    Number of epochs taken: {epoch}\n    Number of batches taken: {no_of_batches}\n    Time taken: {time.time()-begin}")
    return((theta_iterations,generated_theta))

'''
Obtained Result for Termination condition - epochs < 100 :

For absolute diff of moving avg < 1e-2:
    Running for batch size: 1
        Theta obtained: [2.66798434 1.02511091 1.93940873]
        Number of epochs taken: 1
        Number of batches taken: 8000
        Time taken: 0.31285524368286133
    Running for batch size: 100
        Theta obtained: [2.75937793 1.05088433 1.98302832]
        Number of epochs taken: 1
        Number of batches taken: 9000
        Time taken: 0.31638216972351074
    Running for batch size: 10000
        Theta obtained: [2.81873847 1.04050485 1.98780263]
        Number of epochs taken: 100
        Number of batches taken: 10000
        Time taken: 20.768263578414917
    Running for batch size: 1000000
        Theta obtained: [0.24484246 0.91777101 0.46562146]
        Number of epochs taken: 100
        Number of batches taken: 100
        Time taken: 16.1912202835083


For absolute diff of moving avg < 1e-3:
    Running for batch size: 1
        Theta obtained: [3.04445475 0.96776138 2.00765892]
        Number of epochs taken: 1
        Number of batches taken: 82000
        Time taken: 1.0996503829956055
    Running for batch size: 100
        Theta obtained: [2.9145991  1.02132019 1.99451709]
        Number of epochs taken: 2
        Number of batches taken: 13000
        Time taken: 0.3621838092803955
    Running for batch size: 10000
        Theta obtained: [2.81470792 1.04028608 1.98591622]
        Number of epochs taken: 100
        Number of batches taken: 10000
        Time taken: 14.177334070205688
    Running for batch size: 1000000
        Theta obtained: [0.24503307 0.91594349 0.46384932]
        Number of epochs taken: 100
        Number of batches taken: 100
        Time taken: 13.489541053771973

    
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

#Part 3

dirname = os.path.dirname(__file__)
test_data = pd.read_csv(os.path.join(dirname,"./data/q2/q2test.csv")).to_numpy()
test_X = np.vstack([np.ones(test_data.shape[0]),test_data.T[:2]])
test_Y = test_data.T[2]

def predict(X,Y,runs=1):
    J_arr = np.zeros(5)
    run = 0
    while run < runs:
        train_thetas = SGD(batch_size=[1,100,10000,1000000])[1]
        train_thetas.append((np.array([3,1,2]),))
        index = 0
        for i in train_thetas:
            theta = i[-1]
            diff = Y - theta @ X
            J_arr[index] += (1/(2*len(Y))) * (np.sum(diff**2))
            index += 1
        run+=1
        # print(run)
    J_arr/=runs
    batch=[1,100,10000,1000000]
    for i in range(5):
        if i == 4:
            print(f"Test Error for orignal hypothesis: {J_arr[i]}")
        else:
            print(f"Test Error for hypothesis generated by batch size {batch[i]}: {J_arr[i]}")


'''
Obtained Output: (run once)
Test Error for hypothesis generated by batch size 1: 1.2491719628594877
Test Error for hypothesis generated by batch size 100: 0.9874544734500925
Test Error for hypothesis generated by batch size 10000: 1.0797755463751477
Test Error for hypothesis generated by batch size 1000000: 120.73694220436018
Test Error for orignal hypothesis: 0.9829469215

Obtained Output: (averaged over 20 runs)
Test Error for hypothesis generated by batch size 1: 1.1262306695758668
Test Error for hypothesis generated by batch size 100: 1.0024723603378018
Test Error for hypothesis generated by batch size 10000: 1.0816612048778356
Test Error for hypothesis generated by batch size 1000000: 120.90647542640436
Test Error for orignal hypothesis: 0.9829469215000003
'''

#Part 4
def plot_movement():
    b = [1,100,10000,1000000]
    theta_iterations = SGD(batch_size=b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel("theta0")
    ax.set_ylabel("theta1")
    ax.set_zlabel("theta2")
    ax.set_title('Movement of theta for various values of batch_size')
    color = ['red','green','blue','black']
    for i in range(len(theta_iterations)):
        theta = np.array(theta_iterations[i]).T
        ax.plot3D(theta[0],theta[1],theta[2], c=color[i], label=f'b = {b[i]}')
    ax.legend()
    plt.show()

