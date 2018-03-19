# HyS3
HyS3 model, ConKruG algorithm. (Continuous Kruskal-based Graph to be used in Hyber Supervised-Semi-Supervised Prediction)

In Brief: Run HyS3.py and see the financial prediction performance of our model!

In Detail:
These codes try to solve the financial prediction problem using a novel model and another algorithm both produced by the authors of the paper:
"A hybrid supervised semi-supervised graph-based model to predict one-day ahead movement of global stock markets and commodity prices"
Authors:
Arash Negahdari Kia
Saman Haratizadeh
Saeed Bagheri Shouraki

For running the code and achieving the results:
1) Download the data from this link:
Negahdari Ki, Arash (2018), “36 Stock Indices and Commodity Prices Time Series”, Mendeley Data, v1
http://dx.doi.org/10.17632/x744mgjpkv.1

2) Change the path variable in both codes of ConKruG.py and HyS3.py

3) Run HyS3.py (python HyS3.py) and select which models accuracy do you want to see

Notice:

a) If you are running the code in an IDE like Spyder please clear all environmental variables from last runs of other codes
or this code for example by this command: (reset -f)

b) ConKruG.py itself can be used stand-alone (Of course it has less accuracy comparing to HyS3 with ConKruG).
After first run of ConKruG.py the network is built and prediction of anyday can be done by this function:
result = predict(vector, growth, zone_count, alpha_semi) 
Only vector should be changed to a numpy.array of 36 binary digits which represent ups and downs of the market in a day.
Only first 9 digits are important for the function. (9 known markets of American Continent)
the result numpy array's 27 last digits represent ups/downs of 27 markets and commodity prices in other continents.

c) Using HyS3 in an online prediction software requires API from some stock markets which did not exist or were not accessible
to us. Using the offline data in the Mendeley link can show the performance of HyS3.

d) The link to this code is in our article, so any changes to this code that result in changes in the algorithm and accuracies
must be done after authors permission and ofcourse in other versions.
