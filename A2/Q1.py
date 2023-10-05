import os
import random
import pandas as pd
import numpy as np
import time
import math
import re
from PIL import Image # converting images into arrays
import matplotlib.pyplot as plt # for visualizing the data
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def obtain_data(path):
    dirname = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(dirname,path))
    return df

def basic_clean(df):
    df['BasicClean'] = df['CoronaTweet'].str.split()
    return df

# Part a) i)

def naive_bayes_parameters(train, basicclean = True, stem = False):
    # start = time.time()
    train_df = pd.DataFrame()
    for data in train:
        train_df = pd.concat([train_df,obtain_data(data).rename(columns={'Tweet':'CoronaTweet'})],ignore_index=True)
    
    if basicclean:
        basic_clean(train_df)
        clean_row = 'BasicClean'
    else:
        clean_data(train_df,stem = stem)
        clean_row = 'ImprovedClean'
    
    # print(f"Time taken for training data preprocessing:{time.time()-start}")
    # Training phase
    phi_y_pos = 0 
    phi_y_neg = 0
    phi_y_neu = 0
    d_pos = {}
    d_neg = {}
    d_neu = {}
    
    len_pos = 0 #Total Length of positive tweets
    len_neg = 0 #Total Length of negative tweets
    len_neu = 0 #Total Length of neutral tweets
    vocab_size = 0 #Size of vocabulary

    m = train_df.shape[0]
    for index,row in train_df.iterrows():
        sentiment = row['Sentiment']
        if sentiment == 'Positive':
            phi_y_pos += 1
            len_pos += len(row[clean_row]) #Update required length by length of tweet (denominator)
        elif sentiment == 'Negative':
            phi_y_neg += 1
            len_neg += len(row[clean_row])
        else:
            phi_y_neu += 1
            len_neu += len(row[clean_row])
        for word in row[clean_row]:
            if word not in d_pos.keys(): # If word does not exist in the phi's then add this to all of them
                d_pos[word] = 1 #laplace smoothening
                d_neg[word] = 1
                d_neu[word] = 1
                vocab_size+=1
            if sentiment == 'Positive':
                d_pos[word] += 1 #Update the required phi value by 1 (just numerator)
            elif sentiment == 'Negative':
                d_neg[word] += 1 #Update the required phi value by 1 (just numerator)
            else:
                d_neu[word] += 1 #Update the required phi value by 1 (just numerator)
            

    phi_pos = pd.DataFrame(d_pos,index = [0]) # Phi_j given positive
    phi_neu = pd.DataFrame(d_neu,index = [0]) # Phi_j given neutral
    phi_neg = pd.DataFrame(d_neg,index = [0]) # Phi_j given negative

    phi_y_pos/=m
    phi_y_neg/=m
    phi_y_neu/=m

    phi_pos/=(len_pos+vocab_size) #laplace smoothening applied
    phi_neg/=(len_neg+vocab_size)
    phi_neu/=(len_neu+vocab_size)        

    phi = {"y_pos":phi_y_pos,"y_neg":phi_y_neg,"y_neu":phi_y_neu,"x_pos":phi_pos,"x_neg":phi_neg,"x_neu":phi_neu,"len_pos":len_pos,"len_neg":len_neg,"len_neu":len_neu,"vocab_size":vocab_size}
    return phi

def naive_bayes_prediction(train_data = ['Data/ML-A2/Corona_train.csv'], test_data = 'Data/ML-A2/Corona_validation.csv',basicclean=True,confusion=False,stem=False):
    # start = time.time()
    phi = naive_bayes_parameters(train = train_data,basicclean = basicclean,stem = stem)
    # print(f"Time taken for parameter finding:{time.time()-start}")

    # start = time.time()
    if basicclean:
        test_df = basic_clean(obtain_data(test_data))
        clean_row = 'BasicClean'
    else:
        test_df = clean_data(obtain_data(test_data),stem = stem)
        clean_row = 'ImprovedClean'
    # print(f"Time taken for test-data preprocessing:{time.time()-start}")
    
    # start = time.time()
    y_pos = phi["y_pos"]
    y_neg = phi["y_neg"]
    y_neu = phi["y_neu"]
    x_pos = phi["x_pos"]
    x_neg = phi["x_neg"]
    x_neu = phi["x_neu"]
    len_pos = phi["len_pos"]
    len_neg = phi["len_neg"]
    len_neu = phi["len_neu"]
    vocab_size = phi["vocab_size"]

    confusion_matrix = np.zeros((3,3))
    correct_predictions = 0
    total_predictions = 0
    refer_dict = {'pos':"Positive", 'neg': "Negative", 'neu': "Neutral"} 
    index_mapping = {'Positive':0,'Neutral':1,'Negative':2}
    i = 0
    for index,row in test_df.iterrows():
        actual = row['Sentiment']
        posval = 0
        negval = 0
        neuval = 0
        for word in row[clean_row]:
            try:
                posval += math.log(x_pos.at[0,word])
                negval += math.log(x_neg.at[0,word])
                neuval += math.log(x_neu.at[0,word])
            except:
                posval += math.log(1/(vocab_size+len_pos))
                negval += math.log(1/(vocab_size+len_neg))
                neuval += math.log(1/(vocab_size+len_neu))
        posval += math.log(y_pos)
        negval += math.log(y_neg)
        neuval += math.log(y_neu)
        d = {"pos":posval,"neg":negval,"neu":neuval}
        prediction = refer_dict[max(d,key=d.get)]
        confusion_matrix[index_mapping[actual],index_mapping[prediction]] += 1
        if prediction == actual:
            correct_predictions+=1
        total_predictions+=1
    # print(f"Time taken for prediction:{time.time()-start}")
    if confusion:
        print(f"Confusion Matrix for naive bayes prediction:\n{confusion_matrix}")
    return correct_predictions/total_predictions


# print(f"Accuracy in Validation set using naive bayes = {naive_bayes_prediction()}")
# print(f"Accuracy in Training set using naive bayes = {naive_bayes_prediction(test_data = 'Data/ML-A2/Corona_train.csv')}")

#Possible updates to be made - make the parameter and prediction faster

'''
Output:
Training Set:
Time taken for parameter finding:4.943481206893921
Time taken for prediction:25.203924417495728
Accuracy in Training set using naive bayes = 0.8504648214663004

Validation Set:
Time taken for parameter finding:5.051173686981201
Time taken for prediction:2.271172523498535
Accuracy in Validation set using naive bayes = 0.6705132098390525
'''


# Part a) ii)

def show_word_cloud(data = 'Data/ML-A2/Corona_validation.csv',basicclean = True,stem = False):
    if basicclean:
        df = basic_clean(obtain_data(data))
        clean_row = 'BasicClean'
    else:
        df = clean_data(obtain_data(data),stem)
        clean_row = 'ImprovedClean'
    
    ptext = "" #positive
    ntext = "" #negative
    nutext = "" #neutral

    for index, row in df.iterrows():
        sentiment = row['Sentiment']
        if sentiment == 'Positive':
            ptext = ptext + " " + " ".join(row[clean_row])
        elif sentiment == 'Negative':
            ntext = ntext + " " + " ".join(row[clean_row])
        else:
            nutext = nutext + " " + " ".join(row[clean_row])
        
    stopwords = set(STOPWORDS)
    # instantiate a word cloud object
    for i in [ptext,ntext,nutext]:
        wc = WordCloud(
            background_color='white',
            stopwords=stopwords
        )
        # generate the word cloud
        wc.generate(i)
        import matplotlib.pyplot as plt
        # display the word cloud
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    

# show_word_cloud()


# Part b 

def random_prediction(data = 'Data/ML-A2/Corona_validation.csv',confusion = False):
    df = obtain_data(data)
    correct_predictions = 0
    total_predictions = 0
    choice = ['Positive','Negative','Neutral']
    confusion_matrix = np.zeros((3,3))
    index_mapping = {'Positive':0,'Neutral':1,'Negative':2}
    for sentiment in df['Sentiment']:
        prediction = random.choice(choice)
        if sentiment == prediction:
            correct_predictions+=1
        confusion_matrix[index_mapping[sentiment],index_mapping[prediction]] += 1
        total_predictions+=1
    if confusion:
        print(f"Confusion Matrix for random prediction:\n{confusion_matrix}")
    return correct_predictions/total_predictions

def always_positive(data = 'Data/ML-A2/Corona_validation.csv',confusion = False):
    df = obtain_data(data)
    correct_predictions = 0
    total_predictions = 0
    confusion_matrix = np.zeros((3,3))
    index_mapping = {'Positive':0,'Neutral':1,'Negative':2}

    for sentiment in df['Sentiment']:
        if sentiment == 'Positive':
            correct_predictions+=1
        confusion_matrix[index_mapping[sentiment],index_mapping['Positive']] += 1
        total_predictions+=1
    if confusion:
        print(f"Confusion Matrix for always postive prediction:\n{confusion_matrix}")
    return correct_predictions/total_predictions

# rand = random_prediction()
# always_pos = always_positive()
# naive_bayes = naive_bayes_prediction()
# print(f"Random Prediction accuracy = {rand*100}")
# print(f"Always Positive Prediction accuracy = {always_pos*100}")
# print(f"Naive Bayes Prediction accuracy = {naive_bayes*100}")
# print(f"Improvement of Naive Bayes over random = {((naive_bayes-rand)*100)/rand}")
# print(f"Improvement of Naive Bayes over always positive = {((naive_bayes-always_pos)*100)/always_pos}")

'''
Output:
Random Prediction accuracy = 33.73823261463711
Always Positive Prediction accuracy = 43.85059216519891
Naive Bayes Prediction accuracy = 67.05132098390524
Improvement of Naive Bayes over random = 98.73987398739874
Improvement of Naive Bayes over always positive = 52.908587257617704
'''

# Part c
# naive_bayes_prediction(test_data = 'Data/ML-A2/Corona_train.csv',confusion = True)
# naive_bayes_prediction(test_data = 'Data/ML-A2/Corona_validation.csv',confusion = True)
# random_prediction(data = 'Data/ML-A2/Corona_train.csv',confusion = True)
# random_prediction(data = 'Data/ML-A2/Corona_validation.csv',confusion = True)
# always_positive(data = 'Data/ML-A2/Corona_train.csv',confusion = True)
# always_positive(data = 'Data/ML-A2/Corona_validation.csv',confusion = True)

'''
Output:
Confusion Matrix for naive bayes prediction on training set:
[[15711.   177.   714.]
 [ 2158.  3574.  1364.]
 [ 1078.   171. 12917.]]

Confusion Matrix for naive bayes prediction on validation set:
[[1121.   77.  246.]
 [ 271.  174.  172.]
 [ 272.   55.  905.]]

Confusion Matrix for random prediction on training set:
[[5385. 5595. 5622.]
 [2415. 2359. 2322.]
 [4747. 4703. 4716.]]

Confusion Matrix for random prediction on validation set:
[[479. 468. 497.]
 [206. 186. 225.]
 [415. 409. 408.]]

Confusion Matrix for always postive prediction on training set:
[[16602.     0.     0.]
 [ 7096.     0.     0.]
 [14166.     0.     0.]]

Confusion Matrix for always postive prediction on validation set:
[[1444.    0.    0.]
 [ 617.    0.    0.]
 [1232.    0.    0.]]

In all cases: Model correctly predicts for positive tweets more than other sentiment tweets
Possible reason - More data for positive tweets compared to others
'''

# Part d

# nltk.download('stopwords')
# nltk.download('wordnet')

def clean_tweets(tweet,stem=False):
    '''
    1. Remove #'s and @'s (and their corresponding tags)
    2. Remove special chars like !, . , * etc
    3. Stopword removal
    4. Lemmatize
    '''
    #Removing #'s and @'s using RE
    tweet = re.sub(r'(#\w+)|(@\w+)','',tweet)

    #Removing the special characters like !,.,*,(,) etc and converting to lowercase
    tweet = re.sub(r'[^a-zA-z\s]','',tweet).lower()

    #Removing spaces:
    words = tweet.split()

    #Removing stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    if stem:
        #Stemming
        stemmer = PorterStemmer()
        words  = [stemmer.stem(word) for word in words]
    else:
        #Lemmetizing
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
    return words
    
    
def clean_data(df,stem=False):
    try:
        df['ImprovedClean'] = df['CoronaTweet'].apply(lambda tweet: clean_tweets(tweet,stem))
    except:
        df['ImprovedClean'] = df['Tweet'].apply(lambda tweet: clean_tweets(tweet,stem)) #To Handle the domain_adaptation tweets
    return df

# print(clean_data(obtain_data('Data/ML-A2/Corona_train.csv'))) #Part d i)
# show_word_cloud(data = 'Data/ML-A2/Corona_validation.csv',basicclean=False) #Part d ii) 
# print(f"Naive Bayes prediction accuracy with basic data cleaning = {naive_bayes_prediction()}") #Part d iii)
# print(f"Naive Bayes prediction accuracy with better data cleaning = {naive_bayes_prediction(basicclean=False,stem = True)}") #Part d iii)

'''
Output:

Time taken for training data preprocessing:0.29251766204833984
Time taken for parameter finding:5.399029493331909
Time taken for test-data preprocessing:0.024204254150390625
Time taken for prediction:2.5154170989990234
Naive Bayes prediction accuracy with basic data cleaning = 0.6680838141512299

Time taken for training data preprocessing:9.509536027908325
Time taken for parameter finding:12.049105405807495
Time taken for test-data preprocessing:0.8263640403747559
Time taken for prediction:1.0757641792297363
Naive Bayes prediction accuracy with better data cleaning (stemming) = 0.7060431217734588

Time taken for training data preprocessing:4.635559558868408
Time taken for parameter finding:7.202299118041992
Time taken for test-data preprocessing:0.33674120903015137
Time taken for prediction:1.271291971206665
Naive Bayes prediction accuracy with better data cleaning (lemmatizing) = 0.7127239599149712

Observations:
Accuracy increases since the data has lesser noise compared to earlier which allows for better predictions
Lemmatizing has better accuracy over stemming
'''

# Part e

# Leaving for time being until i get a much better idea about bigrams and stuff

# Part f

def domain_adaptation(splits = [1,2,5,10,25,50,100],source = True):
    train = []
    str1 = ""
    prediction_rate = []
    if source:
        train = ['Data/ML-A2/Corona_train.csv']
        str1 = ' with source domain'
    for split in splits:
        train.append(f'Data/ML-A2/Domain_Adaptation/Twitter_train_{split}.csv')
        # start = time.time()
        prediction = naive_bayes_prediction(train_data=train,test_data='Data/ML-A2/Domain_Adaptation/Twitter_validation.csv',basicclean=False)*100
        prediction_rate.append(prediction)
        # print(f"The prediction accuracy when using split {split}{str1} is {prediction}")
        # print(f"Time taken for above training + prediction {time.time()-start}")
        train.pop()
    return prediction_rate

def plot_domain_adaptation(splits = [1,2,5,10,25,50,100]):
    val_with_source = domain_adaptation(splits = splits)
    val_without_source = domain_adaptation(splits = splits,source = False)
    
    plt.plot(splits,val_with_source, label = "With Source")
    plt.plot(splits,val_without_source, label = "Without Source")
    plt.xlabel("Split Size %")  # add X-axis label
    plt.ylabel("Prediction Accuracy (in %)")  # add Y-axis label
    plt.title("Plot")  # add title
    plt.legend(loc='best')
    plt.show()

# domain_adaptation() #Part f i)
# domain_adaptation(source=False) #Part f ii)
# plot_domain_adaptation() #Part f iii)

'''
Output:
The prediction accuracy when using split 1 with source domain is 47.97879390324719
Time taken for above training + prediction 8.48864483833313

The prediction accuracy when using split 2 with source domain is 48.31013916500994
Time taken for above training + prediction 8.122441530227661

The prediction accuracy when using split 5 with source domain is 48.70775347912525
Time taken for above training + prediction 8.238293647766113

The prediction accuracy when using split 10 with source domain is 50.132538104705105
Time taken for above training + prediction 8.666988849639893

The prediction accuracy when using split 25 with source domain is 50.56328694499669
Time taken for above training + prediction 9.211757898330688

The prediction accuracy when using split 50 with source domain is 51.789264413518886
Time taken for above training + prediction 9.661290168762207

The prediction accuracy when using split 100 with source domain is 54.208084824387015
Time taken for above training + prediction 10.351244449615479

The prediction accuracy when using split 1 is 34.327369118621604
Time taken for above training + prediction 0.6906900405883789

The prediction accuracy when using split 2 is 35.387673956262425
Time taken for above training + prediction 0.7213149070739746

The prediction accuracy when using split 5 is 39.72829688535454
Time taken for above training + prediction 0.94508957862854

The prediction accuracy when using split 10 is 44.26772697150431
Time taken for above training + prediction 0.9971485137939453

The prediction accuracy when using split 25 is 46.02385685884692
Time taken for above training + prediction 1.4523470401763916

The prediction accuracy when using split 50 is 49.07223326706428
Time taken for above training + prediction 2.2998082637786865

The prediction accuracy when using split 100 is 52.51822398939695
Time taken for above training + prediction 3.45170259475708

Observations:
1) We see that the prediction accuracy using source data is higher than the prediction accuracy without source data
2) Initially the model without source data performs considerably worse compared to the one using source data
   This can possibly be attributed to the fact that the intial splits have very less data because of which intial model without souce data cannot predict accurately
3) As the split size increases the model without source data approaches the accuracy of the one with source data.
   This can possibly be attributed to the fact that the model without source data is more generalised and can hence predict better compared to the source data model which has been trained against Corona specific tweets
4) Both models reach just above 50% accuracy
   This can possibly be attributed the fact that there are not info datapoints to train the model accurately (ie small size of general tweet training set compared to Corona training set)
   Possibly also due to the fact that many tweets in the general tweet training set are repeated and just subsets of other tweets?
'''
