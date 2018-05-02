#### HyS3 Model 
#### Hybrid Supervised Semi-Supervised Label Spreading Algorithm with SVM
####
#### Author : Arash N. Kia
#### 2017, July
#### 
#### The algorithm is described in the paper mentioned in readme.txt in details
####
###############################################################################

import numpy as np
import pandas as pd
from sklearn import svm

#Start and End date of the dataset in our paper
start_date = '2007-09-17'
end_date   = '2015-06-04'
#Path of the data file (Must be changed according to where you put your file!)
path       = '/home/freeze/Documents/NetPaper/dataset.xls'

print("Running the models may take several minutes (maybe between 5 to 25 minutes depending on your machine!)...")
print("So thank you for being patient...")
print("")
print("Choose between these models to see the accuracy:")
print("1: By Choosing 1 You will see the accuracy of these models:")
print("*** HyS3 with ConKruG network (BEST MODEL) accuracy (Last row of table 4 in paper)" )
print("*** Complete Correlation-based networks with all weights equal to one, accuracy (row 6 of table 4 in paper)")
print("*** Complete Correlation-based network in simple label spreading accuracy (row 7 of table 4 in paper)")
print("*** Maximum Spanning Tree of Correlation-based network accuracy (row 8 of table 4 in paper)")
print("*** ConKruG network used in simple label spreading accuracy (row 9 of table 4 in paper)")
print("2: HyS3 with Complete Correlation-based Model: (row 10 of table 4 in paper)")
print("")
choose = int(input("Choose between 1 or 2: "))

if choose == 1:        
    #Running the ConKruG algorithm and having the ConKruG network
    import ConKruG
    
    growth = ConKruG.growth
    ConKruG_acc = ConKruG.acc_ConKruG
    MST_acc = ConKruG.acc_MST
    comp_corr_acc = ConKruG.acc_full_corr_net
    one_acc = ConKruG.acc_one
    
#Make it false if you don't want to see prompts during the algorithm execution
debug = True

###############################################################################

#Debug prompter!

def log(message):
    
    if debug:
        print(message)
        
###############################################################################

#Method for reading csv dataset
#Inputs: path of the file in string, start_date and end_date in string format: 'YYYY-MM-DD'
#Output: dataset in numpy array format, headers as list of strings
        
def data_reader(path, start_date, end_date):
    
    log('data_reader()...')
    xl = pd.read_excel(path)
    xl['Date'] = pd.to_datetime(xl['Date'])
    xl = xl.set_index('Date')
    xl = xl.loc[start_date:end_date]
    xl = xl.dropna()
    
    #Making the return series out of the dataset (The board of the function is from minus one to infinity)
    #NOTICE: Using the exact return series formula will make higher prediction performance for our model
    #xl = (xl - xl.shift(1)) / xl.shift(1)
    
    #Making the return series in a way that the board of the function is from minus infinity to one.
    #In this way the emphasis is on the down movements of the time series and somehow works as a de-trend 
    #when the series have ascending trends (Most cases in financial time series)
    xl = (xl - xl.shift(1)) / xl
    xl = xl.dropna()
    
    dataset = xl.as_matrix()
    
    #Getting name of the markets from the dataset
    headers = list(xl.columns)
    
    return [dataset, headers]

###############################################################################

#splits the dataset into train, validation and test datasets
#Input: dataset!, proportion for train, proportion for validation
#Output: train, validation, and test datasets in numpy array format
    
def train_test_maker(dataset, prop_train, prop_val):
    
    log('train_test_maker()...')
    
    rows = dataset.shape[0]
    rows_train = int(rows * prop_train)
    rows_val   = int(rows * prop_val)
    train = dataset[:rows_train, :].copy()
    val   = dataset[rows_train:rows_train + rows_val, :].copy()
    test  = dataset[rows_train + rows_val:, :].copy()
    
    return [train, val, test]

###############################################################################

#For changing the order of dataset columns (markets) according to their time zones
#There are four time zones in our research (America: 1, Europe and Africa: 2, West and Central Asia and West of Russia: 3, Far East and Australia: 4)
#Input: dataset, list of markets as headers (Their order must be changed too), and known_zone which is the time zone with known data that other markets are predicted with
#Output: dataset with changed order, headers with changed order, new order of time zones, and a list that shows total markets in each zone
    
def change_order(dataset, headers, time_zones, known_zone):
    
    log('change_order()...')
    
    #these copies of input parameters are saved for continuation of the algorithm without changing the real data
    ds = dataset.copy()
    hs = np.array(headers.copy())
    tz = time_zones.copy()
    tz2 = time_zones.copy()
    
    #In our research and according to the economic facts, American markets are the ones that have the most influence to other markets and therefore it is better to make zone 1 (American zone) as the known zone
    if known_zone == 1:
        #Making American markets labeled ones
        shift_zones = [1, 2, 3, 4]
    elif known_zone == 4:
        #Making Austrailian and east Asiaian markets labeled ones
        shift_zones = [2, 3, 4, 1]
    elif known_zone == 3:
        #Making Central Asian and Russian markets Labeled ones
        shift_zones = [3, 4, 1, 2]
    elif known_zone == 2:
        #Making Europian markets labeled ones
        shift_zones = [4, 1, 2, 3]
    
    for i, n in enumerate(tz):
        tz[i] = shift_zones[n - 1]
    
    #Counting number of markets in each zone which is necessary in label spreading algorithm
    num_of_lbl  = tz.count(1)
    zone2_count = tz.count(2)
    zone3_count = tz.count(3)
    zone4_count = tz.count(4)
    zone_counts = [num_of_lbl, zone2_count, zone3_count, zone4_count]
    
    #Changing the order of columns as the time zones
    new_order = np.argsort(tz)
    new_order = list(new_order)
    ds = ds[:, new_order]
    hs = hs[new_order]
    tz2 = np.array(tz2)[new_order]
    dataset_out, headers_out, time_zones_out = ds, hs, tz2
    
    return [dataset_out, headers_out, time_zones_out, zone_counts]
    
###############################################################################

#Making the correlation network with 1 day lag, according to the time_zones of markets
#Input: Train dataset, Time Zones, Sigma: which is set according to the Zhou paper refrenced in our work
#Output: Complete 1 day Lagged correlation-distance network of the markets in train dataset
#Notice: If we want to use another network like ConKruG, we change the return of this function to return growth (or any other network that you may want to check)
    
def corr_net_maker(train, time_zones, sigma = 0.4):
    
    log('corr_net_maker()...')
    
    columns = train.shape[1]
    cr = np.zeros((columns, columns))
    
    for i in range(0, columns):
        for j in range(0, columns):
    
            series1 = pd.Series(train[:, i])
            series2 = pd.Series(train[:, j])
            
            #Cheking the time zones of each two markets is necessory for daily correlation calculation
            if time_zones[i] == time_zones[j]:
                cr[i, j] = series1.corr(series2.shift(-1))
            if time_zones[i] > time_zones[j]:
                cr[i, j] = series1.corr(series2)
            if time_zones[i] < time_zones[j]:
                cr[i, j] = series1.corr(series2.shift(-1))
    
    cr_temp = np.zeros((columns, columns))
    
    #Making distance out of correlation
    for i in range(0, columns):
        for j in range(0, columns):
            cr_temp[i, j] = np.sqrt(2 * np.sqrt((1 - cr[i, j]) * (1 - cr[j, i])))
    cr = cr_temp
    
    #Making weights out of distance
    weight = np.zeros((columns, columns))
    for i in range(0, columns):
        for j in range(i + 1, columns):
            weight[i, j] = weight[j, i] = np.exp(-(cr[i, j] ** 2) / (sigma ** 2))

    #For understanding what "choose" is, look at first of the code when "choose" is get by an input from the user
    if choose == 1:
        return growth
    else:
        return weight

###############################################################################

#Making a dataset for each single market that a supervised model can use for prediction
#the dataset consists of "delay" days before and the next day as the label
#Input: dataset, "delay" days before, Market number in dataset
    
def delayMaker(dataset, delay, marketNumber):
    
    log("delayMaker()...")
    log(['delay_maker for market number: ', marketNumber])
    series = pd.Series(dataset[:, marketNumber])
    
    
    new_data = np.zeros((1, dataset.shape[0]))
    # why delay + 1 ? Plus one is another column for the class label :)
    for i in range(0, delay + 1):
        temp = series.shift(-i)
        temp = np.array(temp)
        new_data = np.vstack((new_data, temp))
    
    new_data = new_data.T
    new_data = new_data[:, 1:]
    new_data = pd.DataFrame(new_data)
    new_data = new_data.dropna()
    new_data = np.array(new_data)
    
    #Changing the class label from concrete values to discreat binary classes of 1 and 0 for ups and downs of the market
    new_data[:, -1] = (new_data[:, -1] > 0) * 1
    
    return new_data

###############################################################################

#For probabilistic binary classification (prediction) of markets with SVM
#Inputs: dataset, number of market, up_to_time: predicting the probabilities up an exact time, with "delay" days before
#Output: probabilities of predicted classes
    
def supervised_prediction(dataset, marketNumber, up_to_time, delay):
    
    log("supervised_prediction()...")
    log(['supervised_prediction for market#', marketNumber, 'up to time:', up_to_time])
         
    #Making a dataset suitable for supervised prediction
    new_data = delayMaker(dataset, delay, marketNumber)
    
    #SVM supervised prediction
    train = new_data[:up_to_time - delay + 1, :]
    labels = train[:, -1]
    features = train[:, :-1]
    clf = svm.SVC(C = 10000, kernel = 'rbf', gamma = 0.01, probability = True)

    clf = clf.fit(features, labels)
    X = new_data[up_to_time - delay + 1, :-1]
    X = X.reshape(1, -1)
    
    result = clf.predict_proba(X)
    
    return result

###############################################################################

#Chaning the initial vector of label spreading semi-supervised algorithm to the one in HyS3 Model
#Input: dataset, up_to_time (as in supervised_prediction()), list of time_zones, list of total markets in zones, known_zone, and "delay" days before (for supervised_prediction())
#Output: the HyS3 Initial vector of node states instead of simple initial vector of label spreading algorithm
#In this initial vector some nodes have their probabilities of up and down movement rather than zeros.
#Those nodes are nodes with higher supervised prediction performance in validation dataset

def make_personalized_vector(dataset, up_to_time, time_zones, 
                             zone_count, known_zone, delay):
    
    log("make_personalized_vector()...")
    
    #num_of_lbl is the total markets in the known zone    
    num_of_lbl = zone_count[0]
    
    total_markets = dataset.shape[1]
    
    #personalized will be the initial vector of HyS3 after injecting the probabilities
    personalized = np.zeros((total_markets - num_of_lbl, 2))
    
    #Probability injection with selection
    for i in range(num_of_lbl, total_markets):
        log(['make_personalized_vector for row#:', i, 'for up_to_time:', up_to_time])

        #Selection mechanism (these sup_acc and semisum_acc are prediction accuracies of market i for supervised and semi-supervised algorithms in validation set)
        if sup_acc[i] > semisup_acc[known_zone - 1]:
            #according to the time zone and it's relation to the known zone the supervised prediction should be for day up_to_time - 1 or up_to_time
            if time_zones[i] < known_zone:
                temp = supervised_prediction(dataset, i, up_to_time - 1, delay)
            else:
                temp = supervised_prediction(dataset, i, up_to_time, delay)
        #else: If semi_supervised predicton is better   
        else:
            temp = np.array([0, 0])
                
        personalized[i - num_of_lbl, :] = temp
        
    return personalized
        
###############################################################################

#returns single predition for day: up_to_time
#Inputs: alpha_semi: a parameter in label spreading algorithm which is set to 0.99 in literature
#hybrid_flag: for checking if we want to predict with HyS3 (True) or with simple label spreading (False)
#Other inputs are described in previous functions
#Output: a binary vector of up/down predictions for unknown markets

def predict(dataset, up_to_time, time_zones, known_zone, delay, 
            features, net, zone_counts, alpha_semi, hybrid_flag):
    
    log('predict method()...')
    
    #number of known markets
    num_of_lbl = zone_counts[0]
    
    #Label spreading algorithm (Maths are described in paper`)
    row_sum = np.array(net.sum(axis = 1)).squeeze()
    row_sum = row_sum ** (-0.5)
    Dm12 = np.diag(row_sum)
    S = Dm12 @ net @ Dm12
    
    #columns = number of markets in dataset
    columns = net.shape[1]
    result = np.zeros((columns,1))
    
    #Initial vector in HyS3 and label spreading first filled with zeros
    Y0 = np.zeros((columns, 2))

    #Choosing between HyS3 or simple label spreading    
    if hybrid_flag:        
        #Supervised predictor is inside make_personalized_vector() function
        p = make_personalized_vector(dataset, up_to_time, time_zones, zone_counts,
                                         known_zone, delay)        
        Y0[num_of_lbl:, :] = p
        
    #number of known markets
    num_of_lbl = zone_counts[0]
    
    #filling the first rows of the initial vector with known markets fluctuations
    for j in range(0, num_of_lbl):
        Y0[j, int(features[j] > 0)] = 1
    
    #Yhat is the result matrix
    Yhat = np.linalg.inv(np.identity(columns) - alpha_semi * S) @ Y0
    
    #calculating the result vector
    for j in range(0, columns):
        result[j] = (Yhat[j, 0] < Yhat[j, 1]) * 1
        
    return result

###############################################################################

#Evaluate the result of any prediction by feeding the prediction results and the real values as inputs
#Inouts: prediction result for a day, all the test set, the exact day of prediction, known zone, total number of markets in each zone
#All the test set is fed as input for the sake of evaluating the result while predicting with different zones as input (the problem of time zones and calender in the globe)
#Output: a binary vector of unknown markets showing errors
def evaluate(predicted, test_set, day, known_zone, zone_count_list):
        
    log('evaluate()...')
    num_of_lbl = zone_count_list[0]
    z2 = zone_count_list[1]
    z3 = zone_count_list[2]
    
    #Chaning the test set values to binaries of up and down
    l = test_set.copy()
    l = (l > 0) * 1
    
    #Calender and time zone problems
    if known_zone == 1:
        Yu = l[day + 1, num_of_lbl:]
    elif known_zone == 2:
        Yu = np.concatenate((l[day + 1, num_of_lbl: num_of_lbl + z2 + z3] ,
                             l[day, num_of_lbl+ z2 + z3:]))
    elif known_zone == 3:
        Yu = np.concatenate((l[day + 1, num_of_lbl: num_of_lbl + z2],
                             l[day, num_of_lbl + z2:]))
    elif known_zone == 4:
        Yu = l[day, num_of_lbl:]
                 
    return np.abs(predicted[num_of_lbl:].T - Yu)

###############################################################################
#Main function of HyS3
#Inputs: all inputs are described in previous functions and are fed to called functions from this function
#Output: Accuracy and full error array for unknown markets for each day of the prediction for the test dataset 
#(Differene between this function and evaluate() is that the latter only results one day evaluation by knowing the result of prediction but this function calls all predictors and runs the whole HyS3 model and evaluation results are for the whole test set)
def full_hybrid_predict(dataset, time_zones, delay, 
                      network, train, val, test, zone_count, known_zone, alpha_semi, hybrid_flag):
    
    #Initializations 
    net = network.copy()
    days = test.shape[0]
    markets = test.shape[1]
    num_of_lbl = zone_count[0]
    result_spreading = np.zeros((days, markets))
    error_array = np.zeros((days, markets - num_of_lbl))
    
    #If using HyS3 instead of simple label spreading, we should know the first day of test set.
    #Here train_days mean train lenght + validation length
    if hybrid_flag: 
        train_days = dataset.shape[0] - days
    
    from_day = 0
    to_day = days - 2 # -2 comes for the sake of time zone calibrations and calender problems
    for day in range(from_day, to_day): #############FOR LESS RUNNING TIME changed the day-2, to anything less than day - 2 ! :)
        
        log(['FULL PREDICTION FOR:', day, 'of', to_day - from_day])
        
        #Changing the test set to a binary set of ups and downs
        features = (test[day, :] > 0) * 1
        
        #hybrid_flag is set to True if we want to use HyS3 instead of label spreading
        if hybrid_flag:
            up_to_time = train_days + day
        else:
            #Here up_to_time has no effect in predict() function
            up_to_time = 1
            
        #Calling the predict() function for predicting only one day    
        result = predict(dataset, up_to_time, time_zones, known_zone, delay, features,
                         net, zone_count, alpha_semi, hybrid_flag)
        
        #Calling the evaluate method for evaluating the results for unknown markets for only one day
        error = evaluate(result, test, day, known_zone, zone_count)
        error_array[day, :] = error
        
        result_spreading[day, :] = result.reshape((markets,))
        
    #Accuracy = 1 - Error    
    acc = 1 - error_array[from_day:to_day, :].mean()
    #Standard Deviation of the Accuracy report!
    #log(['std = ', error_array[from_day:to_day].std()])
   
    #Accuracy and all the errors for all the unknown markets in all days of test set
    return acc, error_array

###############################################################################

#Supervised model accuracy for all unknown markets to compare with semi-supervised accuracies of them
#in order to check where to inject the probability injection
#Inputs: train set, validation set, "delay" previous days as features for supervised prediction
#Output: markets_acc: accuracy calculated for each market 
    
def sup_accuracy(train, val, delay):
    
    log('sup_accuracy()...')
    
    markets = train.shape[1] # ofcourse it is = val.shape[1] (number of markets)
    days = val.shape[0]
    
    #for all the markets
    markets_acc = []
    #for markets of unknown zones when the known zone is the first zone (America)
    for_std = []
    
    for i in range(0, markets):
        
        #Making train and validation data for supervised forecasting of i'th market.
        #Train data for training of the model and validation data for testing it
        log(["for train data of market ", i])
        train_data = delayMaker(train, delay, i)
        log(["for validation data of market ", i])
        val_data = delayMaker(val, delay, i)
        labels = train_data[:, -1]
        features = train_data[:, :-1]
        
        #Support Vector Machine method in Scikit Learn package
        clf = svm.SVC(C = 10000, kernel = 'rbf', gamma = 0.01, probability = True)
        clf = clf.fit(features, labels)
        
        #error of prediction for all markets
        acc_calc = []
        
        for day in range(0, days - delay):
                           
            X = val_data[day, :-1]
            X = X.reshape(1, -1)
            
            output = clf.predict(X)
            target = val_data[day, -1]
            
            
            acc_calc.append(np.abs(output - target))
            
            if i >= 9: #9 is the number of markets in the known zone (zone 1), for other zones change 9 to the number of markets in that zone
                for_std.append(acc_calc)
                
        #Accuracy of prediction for mrkets
        markets_acc.append(1 - np.mean(acc_calc))
        
    return markets_acc, for_std

###############################################################################

#Semi-supervised accuracy calculation for the selection of candidates in probability injection process of HyS3.
#Inputs: all are described in previous functions. They are handeled in inner functions of this function
#Output: result: semi-supervised accuracy of markets when only label spreading is used in a network. (without supervised prediction)

def semisup_accuracy(dataset, val_or_test, alpha_semi = 0.99):

    log("semisup_accuracy()...")
    
    result = []
    
    for i in range(1, 5):
        
        [dataset, headers] = data_reader(path, start_date, end_date)        
        [dataset, headers, time_zones_out, zone_count] = change_order(dataset, headers,
                                                                 time_zones, known_zone)
        
        [train, val, test] = train_test_maker(dataset, prop_train, prop_val)
               
        net = corr_net_maker(train, time_zones, sigma)
        
        #Semi-supervised graph-based prediction without probability injection        
        acc = full_semi_predict(net, val_or_test, zone_count, i, alpha_semi)
        
        result.append(acc)
        
    return result

###############################################################################
#Semi-supervised prediction (Simple label-spreading without the probability injection mechanism of HyS3)
#Inputs: all the inputs are explained in previous functions except val_or_test
    #val_or_test: This flag is used in a way that the candidate market selection phase sees only the validation data (For a fair evaluation)
#Output: accuracy of prediction for markets
    
def full_semi_predict(network, val_or_test, zone_count, known_zone, alpha_semi = 0.99):
    
    log("full_semi_predict()...")
    
    #This flag is used in a way that the candidate market selection phase sees only the validation data (For a fair evaluation)
    if val_or_test == 1:
        test = val.copy()
        
    net = corr_net_maker(train, time_zones, sigma)
    
    days = test.shape[0]
    markets = test.shape[1]
    num_of_lbl = zone_count[0]
    result_spreading = np.zeros((days, markets))
    error_array = np.zeros((days, markets - num_of_lbl))
    
    for day in range(0, days - 1): 
        features = (test[day, :] > 0) * 1
        result = simple_predict(features, net, zone_count, alpha_semi)
        
        error = evaluate(result, test, day, known_zone, zone_count)
        error_array[day, :] = error
        
        result_spreading[day, :] = result.reshape((markets,)).copy()
        
    acc = 1 - error_array.mean()
    #log(['std = ', error_array.std()])
   
    return acc

###############################################################################

#Prediction for one day with simple label spreading (Without HyS3 mechansim)
#Inputs and outputs are explained in previous functions
    
def simple_predict(features, net, zone_counts, alpha_semi = 0.5):
    
    log("simple_predict()...")
    
    #Label spreading mathematics (Zhou et al. 2004, also described in our paper)
    row_sum = np.array(net.sum(axis = 1)).squeeze()
    row_sum = row_sum ** (-0.5)
    Dm12 = np.diag(row_sum)
    S = Dm12 @ net @ Dm12
    
    columns = net.shape[1]
    result = np.zeros((columns,1))

    Y0 = np.zeros((columns, 2))
    
    num_of_lbl = zone_counts[0]
    
    for j in range(0, num_of_lbl):
        Y0[j, int(features[j] > 0)] = 1
        
    Yhat = np.linalg.inv(np.identity(columns) - alpha_semi * S) @ Y0
    
    #Converting matrix to a vector of binary classes of up and down
    for j in range(0, columns):
        result[j] = (Yhat[j, 0] < Yhat[j, 1]) * 1
        
    return result

###############################################################################
###############################################################################
#    MAIN PART OF THE CODE
###############################################################################
###############################################################################

# INITIALIZATION

time_zones = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3,
             3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 1, 2, 2]

known_zone = 1

prop_train = 0.7
prop_val   = 0.2

alpha_semi = 0.99
sigma = 0.4
delay = 7

###############################################################################

#              HyS3 main part

###############################################################################

#Data preparation for candidate market selection in injection phase (simple label spreading)

log("Simple label spreading accuracy calculation (in main phase)")

[dataset, headers] = data_reader(path, start_date, end_date)

[dataset, headers, time_zones_out, zone_count] = change_order(dataset, headers,
                                                         time_zones, known_zone)

[train, val, test] = train_test_maker(dataset, prop_train, prop_val)


net = corr_net_maker(train, time_zones, sigma)
semisup_acc = semisup_accuracy(dataset, 1)

###############################################################################

#Data preparation for candidate market selection in injection phase (supervised predcition) 
#(Data is read again to make sure that any changes to any matrixes are reset after the selection phase)

log("Simple supervised accuracy calculation (in main phase)")

[dataset, headers] = data_reader(path, start_date, end_date)

[dataset, headers, time_zones_out, zone_count] = change_order(dataset, headers,
                                                         time_zones, known_zone)

[train, val, test] = train_test_maker(dataset, prop_train, prop_val)

sup_acc, for_std = sup_accuracy(train, val, delay)

###############################################################################

#Calling the HyS3 function (Full Hybrid Prediction)

#output of corr_net_maker can be changed to be any of complete correlational network or ConKruG network

log("HyS3 - Hybrid prediction")

net = corr_net_maker(train, time_zones, sigma)

acc, error_array = full_hybrid_predict(dataset, time_zones_out, delay, net,
                                     train, val, test, zone_count,
                        known_zone, alpha_semi, True)


#Printing the final results
if choose == 1:
        
    print("*** HyS3 with ConKruG network (BEST MODEL) accuracy (Last row of table 4 in paper)", acc )
    print("*** Complete Correlation-based networks with all weights equal to one, accuracy (row 6 of table 4 in paper)", one_acc)
    print("*** Complete Correlation-based network in simple label spreading accuracy (row 7 of table 4 in paper)", comp_corr_acc)
    print("*** Maximum Spanning Tree of Correlation-based network accuracy (row 8 of table 4 in paper)", MST_acc)
    print("*** ConKruG network used in simple label spreading accuracy (row 9 of table 4 in paper)", ConKruG_acc)
else:
    print("2: HyS3 with Complete Correlation-based Model: (row 10 of table 4 in paper)", acc)

