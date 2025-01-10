#Non - Hierarchiral "K-Means" Clustering technique and prediction
#Credit based Scoring Analysis Model for Customers/Retailers

#Importing required packages
import pandas as pd
import numpy as np
import seaborn as sn
import datetime as dt
import matplotlib.pyplot as plt

#Loading the dataset..
custo = pd.read_csv("/Users/jobinsamuel/Desktop/ip/MODEL/CreditAnalysisData.csv") 
custo
custo.shape
custo.describe()
custo.info()
custo.head(100)
custo.tail(100)
custo.dtypes

#Checking for any NA Values
custo.isna() 
custo.isna().sum() #171 NA values in 1 column

#Removing unwanted columns
custo = custo.drop(custo.columns[[0,9]],axis=1)
custo
custo.shape #19148,13
custo.describe()
custo.info()

#Checking 'NA' Values after dropping values
custo.isna() 
custo.isna().sum() 
custo.shape

#Removing RetailerID and using only the numbers of the Retailers
custo['retailer_names'] = custo['retailer_names'].replace("RetailerID",' ',regex = True)
custo['retailer_names']
custo['retailer_names'] = custo['retailer_names'].astype(int)


#Removing duplicates
uniq_custo = custo[['retailer_names']].drop_duplicates()
uniq_custo 
uniq_custo.info
uniq_custo.describe()

#Grouping by Retailer ID's and Sorting
uniq_custo.groupby(['retailer_names']).aggregate('count').reset_index().sort_values('retailer_names',ascending = False)

custo.isnull().sum(axis=0) #axis=1

custo['ordereditem_quantity'].min() #Removing the negative values 
custo['ordereditem_unit_price_net'].min()

#Checking the Quantities  which are non-zero and removing it
custo = custo[(custo['ordereditem_quantity'] > 0)] 
custo
custo.info()


custo['created'] = pd.to_datetime(custo['created'])   
custo.head()
custo.tail()

custo['created'].min() #2017-12-18
custo['created'].max() #2018-04-05


custo['value'] ##Quantity(ordereditem_quantity) * Price/unit(ordereditem_unit_price_net)

custo.shape
custo.head()
custo.tail()
custo.dtypes
custo['retailer_names'].astype(int)

#General Visualization plots
custo['value'].mean()   #Mean
custo['value'].median() #Median
custo['value'].var()    #Variance
custo['value'].std()    #Standard Deviation
#Range of customer values
range1 = max(custo['value']) - min(custo['value'])
range1
#Checking the skewness
custo['value'].skew()
#Checking the kurtosis
custo['value'].kurt()

#Plotting a histogram
plt.hist(custo['value'])
#Plotting a box plot
plt.boxplot(custo['value'])

#Removing Outliers for COLUMN 'Value'
IQR = custo["value"].quantile(0.75) - custo["value"].quantile(0.25)
IQR #179.26
ll = custo["value"].quantile(0.25)-1.5*IQR
ll #-187.5324
ul = custo["value"].quantile(0.75)+1.5*IQR
ul #529.507
outli_df = np.where(custo["value"] > ul, True,np.where(custo["value"] < ll,True,False))
outli_df
dm_trim = custo.loc[~(outli_df),]
dm_trim.shape #(17578,13)

#CLTV Metric Calculation and Customer Scoring Mechanism.
from datetime import datetime
latestdate = dt.datetime(2018,4,11) #we can give current date
latestdate

#Making Customer Life Time Value  Analysis using RFM analysis
#Using 'lambda' function to evaluate single statement expressions
rfmvalue = custo.groupby('retailer_names').agg({'created': lambda z: (latestdate - z.max()).days, 'order_id': lambda z: len(z), 'value': lambda z: z.sum()})
rfmvalue 
rfmvalue.dtypes                                                  
                                                   
rfmvalue['created'] = rfmvalue['created'].astype(int)
rfmvalue['created']
rfmvalue.columns

rfmvalue.rename(columns = {'created' : 'Recency', 'order_id' : 'Frequency', 'value' : 'Monetary'}, inplace = True)
rfmvalue
rfmvalue.reset_index().head()
rfmvalue.dtypes


#Splitting up into different segments and Assigning Quantiles to the RFM Parameters
quantiles = rfmvalue.quantile(q = [0.25,0.50,0.75]) #[25%  50%  75%] partitions for quantile processing
quantiles
#Making it into 'Dictionary' format (Quantiles - RFM Parameters)
quantiles = quantiles.to_dict()
quantiles

#User-defined(custom) function to create [R F M] Segments..
#Recency Parameter Block (Recency values should have the lowest gap, so starting from low2high)
def RScoring(x,p,d):
    if x <= d[p] [0.25]:
        return 1
    elif x <= d[p] [0.50]:
        return 2
    elif x <= d[p] [0.75]:
        return 3
    else:
        return 4

#Frequency and Monetary Parameter Blocks(Both are same type) (Frequency and Monetary values should be high, so starting from high2low)
def FMScoring(x,p,d):
    if x <= d[p] [0.25]:
        return 4
    elif x <= d[p] [0.50]:
        return 3
    elif x <= d[p] [0.75]:
        return 2
    else:
        return 1
    
#Calculating the [R F M] Parameters segment values columns in the dataset to grouping
rfmvalue["R"] = rfmvalue['Recency'].apply(RScoring, args = ('Recency',quantiles,))
rfmvalue["F"] = rfmvalue['Frequency'].apply(FMScoring, args = ('Frequency',quantiles,))
rfmvalue["M"] = rfmvalue['Monetary'].apply(FMScoring, args = ('Monetary',quantiles,))
rfmvalue.head()
rfmvalue.tail()

#Calculating the [RFM] Parameter Combined Group value by concatenation of 'string' type in columns
rfmvalue['RFMGROUPVALUE'] = rfmvalue["R"].map(str) + rfmvalue["F"].map(str) + rfmvalue["M"].map(str)
rfmvalue['RFMGROUPVALUE']

#Calculating the [RFM] Parameter value by adding(summation) of [RFM] Parameters
rfmvalue['RFMSCORE'] = rfmvalue[["R","F","M"]].sum(axis = 1)
rfmvalue['RFMSCORE'] 

#Assigning "Labels"/"Loyalty Remarks" to the Customers/Retailers

Loyaltylevel = ['Not Loyal','Likely Loyal','Loyal','Very loyal'] #Defining our own Labels (Where in quan = 1,2,3,4 lowtohigh)
Loyaltylevel
Cust_Scoring = pd.qcut(rfmvalue['RFMSCORE'], q=4, labels = Loyaltylevel)
Cust_Scoring
rfmvalue["RFM_Loyalty_Level"] = Cust_Scoring.values
rfmvalue["RFM_Loyalty_Level"]

#If you want to filter out the best customers/retailers(top ones!)
rfmvalue['RFMSCORE'] == 10
rfmvalue[rfmvalue['RFMGROUPVALUE'] == '433'].sort_values('Monetary',ascending = True).reset_index().head(5)
rfmvalue
rfmvalue[rfmvalue['RFMGROUPVALUE'] == '133'].sort_values('Monetary',ascending = True).reset_index().tail(5)
rfmvalue

#K-Means Clustering Technique Segmentation for Customers/Retailers.
#Writing custom user-defined function to handle negative and zero values..
def neg_zero(num):
    if num < 0:
        return 1
    else:
        return num

#Applying the above code logic in [R F M] Parameters columns..
rfmvalue['Recency'] = [neg_zero(x) for x in rfmvalue['Recency']]
rfmvalue['Recency']
rfmvalue['Monetary'] = [neg_zero(x) for x in rfmvalue['Monetary']]
rfmvalue['Monetary']

#Performing "LOG TRANSFORMATION" method
Log_transf_data = rfmvalue[['Recency','Frequency','Monetary']].apply(np.log,axis = 1).round(3) #Rounding off to 3 places
Log_transf_data

#Standardization 

from sklearn.preprocessing import StandardScaler

scale_object = StandardScaler() #Initialize it..
scaled_data = scale_object.fit_transform(Log_transf_data) #Into Scaled data representation
scaled_data

#Convert into Data frame format 
scaled_data = pd.DataFrame(scaled_data, index = rfmvalue.index, columns = Log_transf_data.columns) 
scaled_data


#KMeans Library Package 
from sklearn.cluster import KMeans

#Kmeans Clustering
ssd={} 
for k in range(1,15):
    kmeans = KMeans(n_clusters=k,init = 'k-means++', max_iter=1000)
    kmeans = kmeans.fit(scaled_data) #Training
    ssd[k] = kmeans.inertia_ #Calculating necessary parameters


#ElbowCurve

sn.pointplot(x = list(ssd.keys()), y = list(ssd.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSD")
plt.title("Elbow Curve for Optimal k Value")
plt.show()

#Kmeans Clustering model building by picking optimal 'k' value
clust_mod = KMeans(n_clusters = 3,init = 'k-means++', max_iter=1000)
clust_mod
clust_mod.fit(scaled_data)
#Labelling it
rfmvalue["Cluster"] = clust_mod.labels_
rfmvalue.head()
rfmvalue.tail()

#Assigning 'Color Codes' to the Cluster column accordingly 
Colors = ['Green','Red','Blue'] #0-Green,1-Red,2-Orange
Colors
rfmvalue["ColorBand"] = rfmvalue["Cluster"].map(lambda s: Colors[s])
rfmvalue["ColorBand"]

#Making into Dummy variable 
onehot = pd.get_dummies(rfmvalue["ColorBand"],prefix = 'ColorBand',dummy_na= False,drop_first=False)
onehot

rfmvalue = pd.concat([rfmvalue,onehot],axis=1)
rfmvalue

rfmvalue = rfmvalue.drop(rfmvalue.columns[[10]],axis=1)
rfmvalue

rfmvalue.reset_index().head(15)
rfmvalue.reset_index().tail(15)

#To make 'RFM_Loyalty_Column' Column as the Last Target 'Y' Column
rfmvalue = rfmvalue.iloc[:,[8,0,1,2,3,4,5,6,7,9,10,11,12]]
rfmvalue
rfmvalue.dtypes

#K - Nearest Neighbours (KNN) algorithm

X=np.array(rfmvalue.iloc[:,1:]) #Input Cols
X 
Y=np.array(rfmvalue.iloc[:,0]) #Output Cols
Y

#Splitting up into Training and Testing data set..
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2) #We can also take (70-30/60-40) ration respectively!
X_train,X_test,Y_train,Y_test

#Checking the Target Column(Label) individual class value count level toknow about the Balance in data..
xx = rfmvalue['RFM_Loyalty_Level'].value_counts().reset_index()
xx
sn.barplot(x = "index", y = "RFM_Loyalty_Level",data = xx)


#Knn algorithmic Specific packages..
from sklearn.neighbors import KNeighborsClassifier  #It can happen in both Classifier/Regressor,depending upon the algorithm implementation and Output column data type, we change into vice-versa accordingly..

#Training a Model..
knn_train = KNeighborsClassifier(n_neighbors = 35)
knn_train
knn_train.fit(X_train,Y_train)

#Testing..
knn_test = knn_train.predict(X_test)
knn_test

#Eval the model Accuracy, Display in CrossTable/Table..
Acc_test = np.mean(Y_test == knn_test)
Acc_test #79.069%
#OR ELSE WE DO BY..
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,knn_test)

#Testing on Training data again..
knn_test_Ontrain = knn_train.predict(X_train)
knn_test_Ontrain
accuracy_score(Y_train,knn_test_Ontrain) #76.744%
pd.crosstab(Y_train,knn_test_Ontrain,rownames=['ActualValue'],colnames=['PredictedValue'])


#Saving the above model for deployment process
#Serialize the model and load it later!
import pickle
pickle.dump(knn_train, open('KnnModel.pkl','wb'))

