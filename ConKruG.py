#### ConKruG Algorithm 
#### Continuous Kruskal-Based Graph maker to find the graph to be used in semi-supervised learning
####
#### Author : Arash N. Kia
#### 2017, July
#### 
#### The algorithm is described in the paper mentioned in readme.txt in details
####
###############################################################################

import numpy as np
import pandas as pd
import networkx as nx

#Make this false and the prompts wont show!
debug = True

#Start and End date of the dataset in our paper
start_date = '2007-09-17'
end_date   = '2015-06-04'
#Path of the data file (Must be changed according to where you put your file!)
path       = '/home/freeze/Documents/NetPaper/dataset.xls'

###############################################################################
#For prompting and print debugging
def log(message):
    
    if debug:
        print(message)
        
###############################################################################
#Input: an excel file of markets time series, start and end date as string with format of YYYY-MM-DD
#Output: matrix of markets time series (rows as days and columns as markets) in numpy array and names of markets in headers list
        
def data_reader(path, start_date, end_date):
    
    log("data_reader()...")
    
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
    headers = list(xl.columns)
    
    return [dataset, headers]

###############################################################################
#For changing the order of dataset columns (markets) according to their time zones
#There are four time zones in our research (America: 1, Europe and Africa: 2, West and Central Asia and West of Russia: 3, Far East and Australia: 4)
#Input: dataset, list of markets as headers (Their order must be changed too), and known_zone which is the time zone with known data that other markets are predicted with
#Output: dataset with changed order, headers with changed order, new order of time zones, and a list that shows total markets in each zone

def change_order(dataset, headers, time_zones, known_zone):
    
    #log('change_order')
    ds = dataset.copy()
    hs = np.array(headers.copy())
    tz = time_zones.copy()
    tz2 = time_zones.copy()
    
    if known_zone == 1:
        #Making American markets labeled
        shift_zones = [1, 2, 3, 4]
    elif known_zone == 4:
        #Making Austraila and East Asia labeled
        shift_zones = [2, 3, 4, 1]
    elif known_zone == 3:
        #Making Central Asia and Russia Labeled
        shift_zones = [3, 4, 1, 2]
    elif known_zone == 2:
        #Making Europian markets labeled
        shift_zones = [4, 1, 2, 3]
    
    for i, n in enumerate(tz):
        tz[i] = shift_zones[n - 1]
    
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
#splits the dataset into train, validation and test datasets
#Input: dataset!, proportion for train, proportion for validation
#Output: train, validation, and test datasets in numpy array format

def train_test_maker(dataset, prop_train, prop_val):
    
    rows = dataset.shape[0]
    rows_train = int(rows * prop_train)
    rows_val   = int(rows * prop_val)
    train = dataset[:rows_train, :].copy()
    val   = dataset[rows_train:rows_train + rows_val, :].copy()
    test  = dataset[rows_train + rows_val:, :].copy()
    
    return [train, val, test]

###############################################################################
#Making the correlation network with 1 day lag, according to the time_zones of markets
#Input: Train dataset, Time Zones, Sigma: which is set according to the Zhou paper refrenced in our work
#Output: Complete 1 day Lagged correlation-distance network of the markets in train dataset

def corr_net_maker(train, time_zones, sigma = 0.4):
    
    columns = train.shape[1]
    cr = np.zeros((columns, columns))

    for i in range(0, columns):
        for j in range(0, columns):
            
            series1 = pd.Series(train[:, i])
            series2 = pd.Series(train[:, j])
            
            #Geographical date change and working hours problem is resolved here:
            if time_zones[i] == time_zones[j]:
                cr[i, j] = series1.corr(series2.shift(-1))
            if time_zones[i] > time_zones[j]:
                cr[i, j] = series1.corr(series2)
            if time_zones[i] < time_zones[j]:
                cr[i, j] = series1.corr(series2.shift(-1))
    
    cr_temp = np.zeros((columns, columns))
    #Making symmetric distance matrix from correlation matrix
    for i in range(0, columns):
        for j in range(0, columns):
            cr_temp[i, j] = np.sqrt(2 * np.sqrt((1 - cr[i, j]) * (1 - cr[j, i])))
    cr = cr_temp
    
    Weight = np.zeros((columns, columns))
    
    #Making weight matrix for label spreading algorithm from distance matrix
    for i in range(0, columns):
        for j in range(i + 1, columns):
            Weight[i, j] = Weight[j, i] = np.exp(-(cr[i, j] ** 2) / (sigma ** 2))

    return Weight

###############################################################################

#One day ahead prediction
#Inputs: features: for labeling the known markets, net: weight graph, zone_counts: number of markets in each zone, alpha_semi which is 0.99 by default as Zhou et al, 2004
#Output: result of the prediction in a binary vector of ups and downs of markets
    
def predict(features, net, zone_counts, alpha_semi = 0.99):

    #Mathematics of label spreading in Zhou et al, 2004
    row_sum = net.sum(axis = 1)
    row_sum = row_sum ** (-0.5)
    Dm12 = np.diag(row_sum)
    S = Dm12 @ net @ Dm12
    
    columns = net.shape[1]
    result = np.zeros((columns,1))

    Y0 = np.zeros((columns, 2))
    
    #Number of markets in the known zone (in our case zone zero (America))
    num_of_lbl = zone_counts[0]
    
    for j in range(0, num_of_lbl):
        Y0[j, int(features[j] > 0)] = 1
        
    #Math of label spreading algorithm
    Yhat = np.linalg.inv(np.identity(columns) - alpha_semi * S) @ Y0
    
    for j in range(0, columns):
        result[j] = (Yhat[j, 0] < Yhat[j, 1]) * 1
        
    return result

###############################################################################
#Evaluate the result of any prediction by feeding the prediction results and the real values as inputs
#Inouts: prediction result for a day, all the test set, the exact day of prediction, known zone, total number of markets in each zone
#All the test set is fed as input for the sake of evaluating the result while predicting with different zones as input (the problem of time zones and calender in the globe)
#Output: a binary vector of unknown markets showing errors
    
def evaluate(predicted, test_set, day, known_zone, zone_count_list):
    
    #number of markets in the first known zone(Zero Zone or American markets)
    #For other zones zero must be changed to 1, 2, or 3
    num_of_lbl = zone_count_list[0]
    z2 = zone_count_list[1]
    z3 = zone_count_list[2]
    
    l = test_set.copy()
    l = (l > 0) * 1
    
    #Geographical date change and working hours problem is resolved here:
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
# This function outputs the whole test set prediction accuracy result with 
# standart deviation of the accuracy and an error_array which shows all the errors
# in detail
# Inputs: network: adjacency matrix of the graph as numpy array, test set, number of markets in zones, known zone (markets with known ups and downs), alpha_semi: which is a parameter in label spreading and set in works of Zhou et al, 2004 by default to 0.99
# Output: is described above

def full_semi_predict(network, test, zone_count, known_zone, alpha_semi = 0.99):
    
    net = network.copy()
    
    days = test.shape[0]
    markets = test.shape[1]
    num_of_lbl = zone_count[0]
    result_spreading = np.zeros((days, markets))
    error_array = np.zeros((days, markets - num_of_lbl))
    
    #tracing the test set rows
    for day in range(0, days - 1):
        
        features = (test[day, :] > 0) * 1
        result = predict(features, net, zone_count, alpha_semi)
        
        error = evaluate(result, test, day, known_zone, zone_count)
        error_array[day, :] = error
        
        result_spreading[day, :] = result.reshape((markets,))
        
    acc = 1 - error_array[0: days - 1, :].mean()
    std = error_array[0: days - 1, :].std() #std of error is equal as std of accuracy (error = 1 - accuracy)
   
    return acc, std, error_array
    
###############################################################################
#
# Maximum Spanning Tree of a network by Kruskal algorithm
# Input: network's adjacency matrix
# Output: MST!

def MST(net):
    
    network = net.copy()
    markets = network.shape[0]
    Tree = np.zeros((markets, markets))
    
    G = nx.from_numpy_matrix(Tree)
    n = 0

    #Kruskal Algorithm    
    while(n < markets - 1):
        
        x, y = np.unravel_index(network.argmax(), network.shape)
        Tree[x, y] = Tree[y, x] = network[x, y]
        network[x, y] = network[y, x] = 0
        n = n + 1
        
        G = nx.from_numpy_matrix(Tree)
        #Checking if there is loop when an edge is added
        if len(nx.cycle_basis(G)) != 0:
            Tree[x, y] = Tree[y, x] = 0
            n = n - 1
        
    return Tree

#############################################################################
###############################################################################

# INITIALIZATION

#Time zone numbers of each 36 market in dataset
time_zones = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3,
             3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 1, 2, 2]

#Known zone is America continent
known_zone = 1
prop_train = 0.7
prop_val   = 0.2


#alpha_semi is the alpha parameter in label spreading which is by Zhou et al., 2004, work by default 0.99
alpha_semi = 0.99
#sigma is set to 0.4 from a grid search with other parameters in validation set
sigma = 0.4

###############################################################################

#Reading the dataset and names of markets
[dataset, headers] = data_reader(path, start_date, end_date)

#Changing the order of market time series in dataset in order to have a right adjacency
#matrix for label spreading (A matrix where all known markets are come first and markets that are being predicted come after both in rows and columns)
[dataset_reordered, headers, time_zones_out, zone_count] = change_order(dataset, headers,
                                                         time_zones, known_zone)

[train, val, test] = train_test_maker(dataset_reordered, prop_train, prop_val)


###############################################################################
# ConKruG ALGORITHM'S MAIN PART
# The algorithm is described in paper
###############################################################################

#distance correlation matrix builder
net = corr_net_maker(train, time_zones, sigma)

#Maximum Spanning Tree with Kruskal
tree = MST(net)

#Finding links that are not in the tree
others = net - tree

#number of links that are not in the tree
n = np.count_nonzero(others) / 2

#Finding the accuracy of the tree
acc, std, e = full_semi_predict(tree, train, zone_count, known_zone)
acc_val, std, e = full_semi_predict(tree, val, zone_count, known_zone)
acc_test, std, e = full_semi_predict(tree, test, zone_count, known_zone)

#having the accuracies for all phases of the ConKruG
tr = []
v = []
te = []
tr.append(1 - acc)
v.append(1 - acc_val)
te.append(1 - acc_test)

#Saving the networks in each phase of the algorithm
nets = []

#First network will be the Maximum Spanning Tree
nets.append(tree)

result_acc = acc
markets = net.shape[0]

#The last result of the ConKruG in each phase of the algorithm will be saved in "growth"
growth = tree.copy()

#Checking each edge that was not in tree. Checking whether the edge can be added or not
#For having the generalization error in mininimum we can use "growth = nets[v.argmin()]"
#In small test sets we will have the growth itself as the result of the ConKruG
for i in range(0, int(n)):
    
    x, y = np.unravel_index(others.argmax(), others.shape)
    growth[x, y] = growth[y, x] = others[x, y]
    others[x, y] = others[y, x] = 0
    
    new_acc, std, e = full_semi_predict(growth, train, zone_count, known_zone, alpha_semi)
    acc_val, std, e = full_semi_predict(growth, val, zone_count, known_zone)
    acc_test, std, e = full_semi_predict(growth, test, zone_count, known_zone)
    
    #Checking the accuracy after adding the edge
    if new_acc < result_acc:
        #Ignoring the edge
        growth[x, y] = growth[y, x] = 0
        print('IGNORING the edge from' , headers[x], 'to', headers[y],' latest acc:', result_acc, 'in phase' , i, 'of', n)
    else:
        #Adding the edge
        result_acc = new_acc    
        print('ADDING the edge from' , headers[x], 'to', headers[y],' latest acc:', result_acc, 'in phase' , i, 'of', n)
        tr.append(1 - result_acc)
        v.append(1 - acc_val)
        te.append(1 - acc_test)
        nets.append(growth.copy())

acc_ConKruG, std_ConKruG, e_ConKruG = full_semi_predict(growth, test, zone_count, known_zone)
acc_full_corr_net, std_full_corr_net, e_full_corr_net = full_semi_predict(net, test, zone_count, known_zone)
acc_MST, std_MST, e_MST = full_semi_predict(tree, test, zone_count, known_zone)

#For a complete network of all edges with weight one
one_net = np.ones((net.shape[0], net.shape[0]))
np.fill_diagonal(one_net, 0)
acc_one, std_one, e_one = full_semi_predict(one_net, test, zone_count, known_zone)


