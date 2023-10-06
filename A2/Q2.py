import os
import random
import pandas as pd
import numpy as np
import time
import math
import itertools
from cvxopt import matrix, solvers
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import re
from PIL import Image # converting images into arrays
import matplotlib.pyplot as plt # for visualizing the data
# from wordcloud import WordCloud, STOPWORDS
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer

dirname = os.path.dirname(__file__)

def image_process(image_path):
    img = Image.open(image_path)
    img = img.resize((16,16))
    pixels = np.asarray(img)
    pixels = pixels.astype('float32')
    pixels /= 255.0
    return pixels.ravel()

# To increase performance trying to process images in batches
def image_process_batch(image_paths, batch_size=16):
    num_images = len(image_paths)
    processed_images = []

    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for image_path in batch_paths:
            img = Image.open(image_path)
            img = img.resize((16, 16))
            pixels = np.asarray(img)
            pixels = pixels.astype('float32')
            pixels /= 255.0
            batch_images.append(pixels.ravel())

        processed_images.extend(batch_images)

    return processed_images

# def process_all(image_paths):
#     num_images = len(image_paths)
#     processed_images = []

#     for i in range(0, num_images):
#         img = Image.open(image_paths[i])
#         img = img.resize((16, 16))
#         pixels = np.asarray(img)
#         pixels = pixels.astype('float32')
#         pixels /= 255.0
#         processed_images.append(pixels.ravel())

#     return processed_images

def get_data(req_dir = [1,2],train = True):
    directories = []
    dict_data = {}
    if train:
        for i in req_dir:
            directories.append(os.path.join(dirname,f'Data/train/{i}'))
    else:
        for i in req_dir:
            directories.append(os.path.join(dirname,f'Data/val/{i}'))

    for i in directories:
        temp = []
        for j in os.listdir(i):
            temp.append(os.path.join(i,j))
        dict_data[int(i[-1])] = np.array(image_process_batch(temp,batch_size=16))
    return dict_data


# print(image_process('./Data/train/0/0.jpg'))
# print(get_data())


# Any class Classification:

#Kernel Functions:
def linear(x,z): #x and z should be training example path (and not the vector themselves)
    return x.T @ z

def gaussian(x,z,g=0.001):
    return math.exp(-g*(np.linalg.norm(x-z))**2)

#Probably should try to get the general form of this to get help in efficiency later
def kernel_matrix(kernel,X,g=0.001): #Returns a matrix with Kij = K(xi,xj) where xi and xj are all paths to training example
    # print("Entered kernel_matrix")
    # X = process_all(X)
    # X = [image_process(x) for x in X]
    if kernel == linear:
        K = np.dot(X, X.T)
    else:
        m = X.shape[0]
        K = np.empty((m,m))
        for i in range(m):
            for j in range(i,m):
                K[i, j] = kernel(X[i],X[j])
                K[j, i] = K[i, j]
                # print(f"{(i,j)} done")
    return K

# NOT COMPLETE:
def kernel_fn(kernel,X,Y,g=0.001):
    if kernel == linear:
        return(X@Y.T)
    else:
        pass

def cvxopt_svm_training(Y,X,c=1.0,use_linear = True): #Y is output vector of training examples [numpy array], X is input vector of training examples [python list]
    kernel = linear if use_linear else gaussian
    m = len(Y)

    # start = time.time()
    #Obtaining required parameters for CVXOPT Solver
    diag_Y = np.diag(Y.ravel())
    P = matrix(diag_Y @ kernel_matrix(kernel,X) @ diag_Y,tc='d')
    # print(f"P matrix calculated in {time.time()-start}")
    q = matrix(np.full(m,-1),tc='d')
    A = matrix(Y.T,tc='d')
    b = matrix([0],tc='d')

    G_upper_half = np.eye(m)
    G_lower_half = -np.eye(m)
    G = np.vstack((G_upper_half, G_lower_half))
    G = matrix(G,tc='d')

    h_upper_half = np.full(m, c)
    h_lower_half = np.zeros(m)
    h = np.concatenate((h_upper_half, h_lower_half))
    h = matrix(h,tc='d')

    # print(f"All required parameters calculated in {time.time()-start}")
    return solvers.qp(P,q,G,h,A,b)

def cvxopt_svm_predict(req_dir=[1,2],c=1.0,use_linear = True,threshold = 1e-6,binary = True,confusion = False):
    begin = time.time()
    if len(req_dir) == 2:
        binary = True
    else:
        binary = False

    kernel = linear if use_linear else gaussian

    kernel_name = "linear" if use_linear else "gaussian"
    model_name = "binary" if binary else "multi"

    print(f"For {model_name} class and {kernel_name} kernel:")
    
    start = time.time()
    train_data = get_data(req_dir,train=True) #Obtained entire training dataset
    print(f"Time taken for training data retrieval: {time.time()-start}s\n")
    
    m = int(len(train_data[req_dir[0]])) #no of training examples in a given class
    
    #Obtaining test data
    start = time.time()
    test_data = get_data(req_dir,train=False)
    print(f"Time taken for test data retrieval: {time.time()-start}s\n")
    
    m_test = int(len(test_data[req_dir[0]])) #no of test examples per class
    X_test = list()
    Y_test = np.array([])
    for i in test_data.keys():
        Y_test = np.concatenate((Y_test,np.full(m_test,i)))
        for j in test_data[i]:
            X_test.append(j)
    X_test = np.array(X_test)
    # print(X_test.shape,Y_test.shape)

    pairs = list(itertools.combinations(req_dir, 2))
    total_predictions = []
    train_test_start = time.time()
    total_train_time = 0
    total_validation_time = 0
    for pair in pairs:
        #Training model i,j
        m1 = pair[0] #model 1
        m2 = pair[1] #model 2
        print(f"Currently going to model {m1},{m2}")
        Y_upper_half = np.full((m,1), -1)
        Y_lower_half = np.full((m,1), 1)
        Y = np.concatenate((Y_upper_half, Y_lower_half))
        X = np.concatenate((train_data[m1],train_data[m2]))

        # print(X.shape,Y.shape)
        start = time.time()
        solvers.options['show_progress'] = False
        sol = cvxopt_svm_training(Y,X,c = c,use_linear = use_linear)
        stop = time.time()
        total_train_time+=stop-start
        print(f"Time taken to train model {m1}_{m2}: {stop-start}s")

        start = time.time()
        alphas = np.array(sol['x'])
        support_vector_idx = (alphas > threshold).flatten()
        nSV = np.count_nonzero(support_vector_idx)
        if binary:
            print(f"No of support vectors: {nSV}")
            print(f"% of support vectors wrt training examples: {(nSV)*100/(2*m)}%")

        support_X = X[support_vector_idx]
        support_Y = Y.flatten()[support_vector_idx]
        support_alphas = alphas.flatten()[support_vector_idx]

        if (binary and use_linear):
            W = support_X.T @ (support_alphas * support_Y)

        b = 0
        for i in range(nSV): 
            for j in range(nSV):
                b-=support_alphas[j]*support_Y[j]*kernel(support_X[j],support_X[i])
            b+=support_Y[i]
        b /= nSV

        # b = support_vector_y - np.sum(kernel_fn(kernel,support_vector,support_vector).T * req_alphas * support_vector_y, axis=0)
        # b = np.mean(b)
        if binary:
            print(f"Obtained b is: {b}")

        print(f"Time taken for calcualtion of SV's and b: {time.time()-start}s")

        prediction_start = time.time()
        predictions = np.zeros(Y_test.shape[0])
        for i in range(Y_test.shape[0]):
            prediction = 0
            for j in range(nSV):
                prediction += support_alphas[j]*support_Y[j]*kernel(support_X[j],X_test[i])
            prediction += b
            predictions[i] = prediction
        # prediction = np.sum(kernel_fn(kernel,support_vector,X_test).T * req_alphas * support_vector_y, axis=0) + b
        result = np.where(predictions >= 0, m2, m1)
        total_predictions.append(result)
        total_validation_time+=time.time()-prediction_start
        print(f"Time taken for prediction on whole data set using model {m1}_{m2}: {time.time()-prediction_start}s\n")

    final_validation_time = time.time()
    all_predictions = np.array(total_predictions)
    k, m_test_total = all_predictions.shape

    # Initialize the collapsed array with zeros
    predicted_Y = np.zeros(m_test_total)

    for j in range(m_test_total):
        column_j = all_predictions[:, j]

        # Find unique values and their counts in the column
        unique_values, counts = np.unique(column_j, return_counts=True)

        # Find the maximum occurring value(s) and choose the largest value in case of ties
        max_occurrence_value = unique_values[counts.argmax()]

        # Set the maximum occurring value in the collapsed array
        predicted_Y[j] = max_occurrence_value
    
    total_validation_time += time.time()-final_validation_time
    print(f"Validation test accuracy = {np.sum(predicted_Y == Y_test)*100/(m_test_total)}%")
    print(f"Time taken for training: {total_train_time}s")
    print(f"Time taken for validation: {total_validation_time}s\n")
    

    if confusion:
        ConfusionMatrixDisplay.from_predictions(Y_test, predicted_Y)
        plt.show()
        if not binary:
            miss_classified = []
            n=0
            for i in range(Y_test.shape[0]):
                if ((predicted_Y[i] != Y_test[i]) and  (n <= 11)):
                    n+=1
                    miss_classified.append((predicted_Y[i],Y_test[i]))
                    img = Image.fromarray((255*X_test[i]).reshape(16,16,3).astype(np.uint8))
                    img = img.resize((320,320), resample=Image.NEAREST)
                    img.save(f'./Images/Q2_{model_name}/miss_classified/cvxopt{n}_{int(predicted_Y[i])}_{int(Y_test[i])}.png')

    if binary:
        top_6 = []
        if use_linear:
            img = Image.fromarray((255*(W/(np.linalg.norm(W)))).reshape(16,16,3).astype(np.uint8))
            img = img.resize((320,320), resample=Image.NEAREST)
            img.save(f'./Images/Q2_{model_name}/{kernel_name}_w.png')
            
        for i in np.argsort(support_alphas.flatten())[:6]:
            top_6.append((255*support_X[i]).reshape(16,16,3).astype(np.uint8))

        for (i,vec) in enumerate(top_6):
            img = Image.fromarray(vec)
            img = img.resize((320,320), resample=Image.NEAREST)
            img.save(f'./Images/Q2_{model_name}/{kernel_name}_sv_{i+1}.png')

    print(f"Total time taken for everything: {time.time()-begin}")

    if binary: 
        return(support_vector_idx.nonzero()[0])

# Q2 Binary Classfication Part a,b:
'''
Output:
For Linear Kernel:
    Time taken for training data retrieval: 22.461912393569946s
    
    No of support vectors: 673
    % of support vectors wrt training examples: 14.138655462184873%
    Obtained b is: -4.0888424817486575
    Time taken to train model: 98.0978832244873s

    Validation test accuracy = 94.0%
    Time taken for validation 2.5156967639923096s

For Gaussian Kernel:
    Time taken for training data retrieval: 25.414992570877075s
    
    No of support vectors: 1067
    % of support vectors wrt training examples: 22.415966386554622%
    Obtained b is: 4.766861877598194
    Time taken to train model: 176.71735429763794s

    Validation test accuracy = 94.25%
    Time taken for validation 4.0758960247039795s


Number of support vectors that match in both linear and gaussian: 571
'''

# Part c)
def sklearn_svm_predict(req_dir=[1,2],c=1.0,use_linear=True,binary = True,confusion = False):
    if len(req_dir) == 2:
        binary = True
    else:
        binary = False

    kernel = 0 if use_linear else 2

    kernel_name = "linear" if use_linear else "rbf"
    model_name = "binary" if binary else "multi"
    print(f"For {model_name} class and {kernel_name} kernel:")

    start = time.time()
    train_data = get_data(req_dir,train=True)
    print(f"Time taken for training data retrieval: {time.time()-start}s\n")
    
    m = int(len(train_data[req_dir[0]])) #no of training examples in a given class
    
    X_train = list()
    Y_train = np.array([])
    for i in train_data.keys():
        Y_train = np.concatenate((Y_train,np.full(m,i)))
        for j in train_data[i]:
            X_train.append(j)
    X_train = np.array(X_train)
    
    
    start = time.time()
    test_data = get_data(req_dir,train=False)
    print(f"Time taken for test data retrieval: {time.time()-start}s\n")
    
    m_test = int(len(test_data[req_dir[0]])) #no of test examples per class
    X_test = list()
    Y_test = np.array([])
    for i in test_data.keys():
        Y_test = np.concatenate((Y_test,np.full(m_test,i)))
        for j in test_data[i]:
            X_test.append(j)
    X_test = np.array(X_test)
    
    
    start = time.time()
    model = svm.SVC(C=c, kernel=kernel_name,gamma=0.001,tol=1e-6)
    model.fit(X_train,Y_train)
    print(f"Time taken to train model: {time.time()-start}s\n")
    
    support_X = model.support_vectors_
    support_idx = model.support_
    support_Y = Y_train[support_idx]
    nSV = sum(model.n_support_)
    print(f"No of support vectors: {nSV}")

    if use_linear and binary:
        weights = model.coef_
        # return weights
        b = 0
        for i in range(nSV): 
            b+=support_Y[i] - weights.T.flatten() @ support_X[i]
        b /= nSV
        print(f"The bias in linear case is {b}")


    start = time.time()
    predicted_Y = model.predict(X_test)
    print(f"Validation test accuracy = {np.sum(np.array(predicted_Y) == Y_test)*100/(Y_test.shape[0])}%")
    print(f"Time taken for validation {time.time()-start}s\n")
    

    if confusion:
        ConfusionMatrixDisplay.from_predictions(Y_test, predicted_Y)
        plt.show()
        if not binary:
            miss_classified = []
            n=0
            for i in range(Y_test.shape[0]):
                if ((predicted_Y[i] != Y_test[i]) and  (n <= 11)):
                    n+=1
                    miss_classified.append((predicted_Y[i],Y_test[i]))
                    img = Image.fromarray((255*X_test[i]).reshape(16,16,3).astype(np.uint8))
                    img = img.resize((320,320), resample=Image.NEAREST)
                    img.save(f'./Images/Q2_{model_name}/miss_classified/sklearn{n}_{int(predicted_Y[i])}_{int(Y_test[i])}.png')

    if binary:
        return np.array(support_idx)

# linear_cvx = cvxopt_svm_predict(req_dir = [1,2])
# gaussian_cvx = cvxopt_svm_predict(req_dir = [1,2],use_linear=False)
# print(f"Number of support vectors that match in both linear and gaussian: {len(set.intersection(set(linear_cvx.flatten()),set(gaussian_cvx.flatten())))}")

# linear_skl = sklearn_svm_predict()
# gaussian_skl = sklearn_svm_predict(use_linear=False)

# print(f"Number of support vectors that match in both linear_cvxopt and linear_sklearn_svm: {len(set.intersection(set(linear_cvx.flatten()),set(linear_skl.flatten())))}")
# print(f"Number of support vectors that match in both gaussian_cvxopt and gaussian_sklearn_svm: {len(set.intersection(set(gaussian_cvx.flatten()),set(gaussian_skl.flatten())))}")
# print(f"Number of support vectors that match in both linear_cvxopt and gaussian_sklearn_svm: {len(set.intersection(set(linear_cvx.flatten()),set(gaussian_skl.flatten())))}")
# print(f"Number of support vectors that match in both gaussian_cvxopt and linear_sklearn_svm: {len(set.intersection(set(gaussian_cvx.flatten()),set(linear_skl.flatten())))}")
# print(f"norm is {np.linalg.norm(linear_cvx-linear_skl)}")

# Q2 Binary Classification part c)
'''
Output:

i) nSV and number of matching Support vectors:
    For linear Kernel:
        CVXOPT: nSV = 673
        sklearn_svm: nSV = 669
    For Gaussian Kernel:
        CVXOPT: nSV = 1067
        sklearn_svm: nSV = 1056

Number of support vectors that match in both linear_cvxopt and linear_sklearn_svm: 669
Number of support vectors that match in both gaussian_cvxopt and gaussian_sklearn_svm: 1056
Number of support vectors that match in both linear_cvxopt and gaussian_sklearn_svm: 566
Number of support vectors that match in both gaussian_cvxopt and linear_sklearn_svm: 569
Number of support vectors that match in both linear_cvx and gaussian_cvx: 571
    
        
ii) Comparison of weight and bias in linear kernel:
    CVXOPT: b = -4.0888424817486575
    sklearn_svm: b = -4.0881856782113015
    norm(w_cv - w_skl) = 0.015257845622513772

iii) Validation accuracy:
    For linear Kernel:
        CVXOPT: 94.0%
        sklearn_svm: 94.75%
    For Gaussian Kernel:
        CVXOPT: 94.25%
        sklearn_svm: 93.75%

iv) Time taken (training time):
    For linear Kernel:
        CVXOPT: 81.49388098716736s (+ 24.363908529281616s for data retrieval)
        sklearn_svm: 1.9442343711853027s (+ 22.59026861190796s for data retrieval)
    For Gaussian Kernel:
        CVXOPT: 154.25795817375183s (+ 23.08076786994934s for data retrieval)
        sklearn_svm: 1.649414300918579s (+ 23.21091604232788s for data retrieval)

Observations:
1) sklearn_svm has lesser number of Support vectors in comparison to the one by cvxopt (by a small amount)
2) All sklearn_svm support vectors match with those of cvxopt support vectors
3) The bias and weights obtained from both methods are mostly identical (weights differ by 1e-2 and bias by 1e-4)
4) Validation accuracy is almost similar for both, .75% higher for sklearn_svm in linear kernel and .5% higher for cvxopt in gaussian kernel)
5) Time taken differs substanstially with sklearn_svm being much much faster than cvxopt
 -Possible reasons - sklearn_svm iteratively obtains the required parameters (ascent algorithm)
'''



# sklearn_svm_predict(req_dir=[0,1,2,3,4,5],use_linear=False,binary=False,confusion=True)
# cvxopt_svm_predict(req_dir=[0,1,2,3,4,5],use_linear=False,binary=False,confusion=True)

'''
Output of above code:
For multi class and rbf kernel:
Time taken for training data retrieval: 78.75837922096252s

Time taken for test data retrieval: 6.559682369232178s

Time taken to train model: 65.871990442276s

No of support vectors: 11930
Validation test accuracy = 56.083333333333336%
Time taken for validation 7.118214845657349s

For multi class and gaussian kernel:
Time taken for training data retrieval: 71.19300150871277s

Time taken for test data retrieval: 5.576226711273193s

Currently going to model 0,1
Time taken to train model 0_1: 158.43934082984924s
Time taken for calcualtion of SV's and b: 24.05388903617859s
Time taken for prediction on whole data set using model 0_1: 13.884382009506226s

Currently going to model 0,2
Time taken to train model 0_2: 127.29201483726501s
Time taken for calcualtion of SV's and b: 49.14700937271118s
Time taken for prediction on whole data set using model 0_2: 18.5403995513916s

Currently going to model 0,3
Time taken to train model 0_3: 124.78682136535645s
Time taken for calcualtion of SV's and b: 60.089139461517334s
Time taken for prediction on whole data set using model 0_3: 20.63145637512207s

Currently going to model 0,4
Time taken to train model 0_4: 116.13642573356628s
Time taken for calcualtion of SV's and b: 95.40940022468567s
Time taken for prediction on whole data set using model 0_4: 26.965206384658813s

Currently going to model 0,5
Time taken to train model 0_5: 145.8709592819214s
Time taken for calcualtion of SV's and b: 48.82317805290222s
Time taken for prediction on whole data set using model 0_5: 19.437594413757324s

Currently going to model 1,2
Time taken to train model 1_2: 165.88792443275452s
Time taken for calcualtion of SV's and b: 5.829539775848389s
Time taken for prediction on whole data set using model 1_2: 6.433315992355347s

Currently going to model 1,3
Time taken to train model 1_3: 163.83112025260925s
Time taken for calcualtion of SV's and b: 7.135587930679321s
Time taken for prediction on whole data set using model 1_3: 7.088313102722168s

Currently going to model 1,4
Time taken to train model 1_4: 143.40084409713745s
Time taken for calcualtion of SV's and b: 13.596635818481445s
Time taken for prediction on whole data set using model 1_4: 9.783212900161743s

Currently going to model 1,5
Time taken to train model 1_5: 127.41238451004028s
Time taken for calcualtion of SV's and b: 34.6133496761322s
Time taken for prediction on whole data set using model 1_5: 15.646651983261108s

Currently going to model 2,3
Time taken to train model 2_3: 146.68378257751465s
Time taken for calcualtion of SV's and b: 48.69501233100891s
Time taken for prediction on whole data set using model 2_3: 19.109719276428223s

Currently going to model 2,4
Time taken to train model 2_4: 127.65542960166931s
Time taken for calcualtion of SV's and b: 69.77258443832397s
Time taken for prediction on whole data set using model 2_4: 22.495891094207764s

Currently going to model 2,5
Time taken to train model 2_5: 151.1695499420166s
Time taken for calcualtion of SV's and b: 21.955137491226196s
Time taken for prediction on whole data set using model 2_5: 12.689659357070923s

Currently going to model 3,4
Time taken to train model 3_4: 129.83402848243713s
Time taken for calcualtion of SV's and b: 70.3451521396637s
Time taken for prediction on whole data set using model 3_4: 22.15795660018921s

Currently going to model 3,5
Time taken to train model 3_5: 140.7557249069214s
Time taken for calcualtion of SV's and b: 25.223090410232544s
Time taken for prediction on whole data set using model 3_5: 13.933132648468018s

Currently going to model 4,5
Time taken to train model 4_5: 133.27530241012573s
Time taken for calcualtion of SV's and b: 47.027114391326904s
Time taken for prediction on whole data set using model 4_5: 18.612504720687866s

Validation test accuracy = 55.416666666666664%
Time taken for training: 2102.4316532611847s
Time taken for validation: 247.43107676506042s

Total time taken for everything: 3106.967389345169
'''



# Q2 Multi-Classification a,b,c:
'''
Output:

a) Validation test accuracy: 55.416666666666664%
   Time taken for train: 2028.4954512119293s ~ 33 mins
   Time taken for validation: 247.43107676506042s ~ 6 mins

b) i) Validation test accuracy: 56.083333333333336%
      Time taken for train: 53.31632876396179s ~ 1 min
      Time taken for validation: 6.054398775100708s
   ii) We can see the the sklearn lib is very fast to train the model in comparison to cvxopt (~33 times faster) 
   while the prediction accuracy is almost simlar (~0.53% difference)

c) Check Q2_multi folder
'''

# Q2 Multi-Classdification d:


def validation_data():
    train_data = get_data([0,1,2,3,4,5],train=True)
    test_data = get_data([0,1,2,3,4,5],train=False)
    
    m = len(train_data[0])
    X_train = list()
    Y_train = np.array([])
    for i in train_data.keys():
        Y_train = np.concatenate((Y_train,np.full(m,i)))
        for j in train_data[i]:
            X_train.append(j)
    X_train = np.array(X_train)

    m_test = len(test_data[0])
    X_test = list()
    Y_test = np.array([])
    for i in test_data.keys():
        Y_test = np.concatenate((Y_test,np.full(m_test,i)))
        for j in test_data[i]:
            X_test.append(j)
    X_test = np.array(X_test)

    return(X_train,Y_train,X_test,Y_test)

def validation(C,g=0.001):
    X_train,Y_train,X_test,Y_test = validation_data()
    prediction_rates = []
    for c in C:
        model = svm.SVC(C=c, kernel='rbf',gamma=g,tol=1e-6)
        model.fit(X_train,Y_train)
        predicted_Y = model.predict(X_test)
        prediction_rates.append(np.sum(np.array(predicted_Y) == Y_test)*100/(Y_test.shape[0]))
    return prediction_rates

def cross_val_data():
    train_data = get_data([0,1,2,3,4,5],train=True)
    
    m = len(train_data[0])
    X_train = list()
    Y_train = np.array([])
    for i in train_data.keys():
        Y_train = np.concatenate((Y_train,np.full(m,i)))
        for j in train_data[i]:
            X_train.append(j)
    X_train = np.array(X_train)
    return X_train,Y_train

def cross_val(C,k = 5,g=0.001):
    X,Y = cross_val_data()

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    splits = k
    m_total = Y.shape[0]
    m_split = int(m_total/splits)
    prediction_rates = []

    for c in C:
        prediction = 0
        print(f"c value of {c}")
        for i in range(k):
            print(f"split ver: {i}")
            X_test = X[m_split*i:m_split*(i+1)]
            Y_test = Y[m_split*i:m_split*(i+1)]

            X_train = np.concatenate((X[0:m_split*i],X[m_split*(i+1):]))
            Y_train = np.concatenate((Y[0:m_split*i],Y[m_split*(i+1):]))

            model = svm.SVC(C=c, kernel='rbf',gamma=g,tol=1e-6)
            model.fit(X_train,Y_train)
            predicted_Y = model.predict(X_test)
            prediction += np.sum(np.array(predicted_Y) == Y_test)*100/(Y_test.shape[0])
        prediction_rates.append(prediction/k)

    return prediction_rates


def plot_cross_vs_normal(k = 5, C=[1e-5,1e-3,1,5,10],g = 0.001):
    cross_rates = cross_val(C,k,g)
    validation_rates = validation(C,g)
    plt.plot(C, validation_rates, label='test')
    plt.errorbar(C, cross_rates,label='CV')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s C')
    plt.legend()
    plt.show()
    # plt.savefig('acc.pdf', bbox_inches='tight')

# plot_cross_vs_normal()


# Things to do 
# a) Need to try and make better kernel functions (ie try and use matrix formulas instead of loops) 
# b) Need to plot weight vector in linear kernel case of cvxopt
# c) Need to test out size change in image saving
# d) Check the number of matching SV's 