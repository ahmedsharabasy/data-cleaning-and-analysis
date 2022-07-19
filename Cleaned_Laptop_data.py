from tkinter import Place
from joblib import PrintTime
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from regex import P
import seaborn as sns
import os

# for dirname, _, filenames in os.walk('C:\\Users\\20100\\Desktop\\data science\\machine learning\\projects\\regression'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

data = pd.read_csv('C:\\Users\\20100\\Desktop\\data science\\machine learning\\projects\\regression\\Cleaned_Laptop_data.csv')
# print('Data:\n',data.head(10))
# print('statistics:\n',data.describe())
# print('data info:',data.info())

#ONE OF ESSENTIAL STEP IS TO CHECK NULL VALUES
print(data.isna().sum())
print()
print(data.ram_gb.value_counts(dropna=False))

#AS NUMBER OF NULL VALUES IS SMALL,WE CAN DROP THE NULL VALUES.
#ALSO RAM VALUE CAN ONLY BE NUMERICAL VALUE,
#IT CANNNOT BE 'NVIDIA','Dual','Acer','Pre-installed','Full','Intel','Access','5'.

filter = data[data['ram_gb'].isin(['NVIDIA','Dual','Acer','Pre-installed','Full','Intel','Access',np.nan,'5'])]
#USED ISIN TO CREATE A NEW DATAFRAME WHERE RAM_GB CONTAINS ONLY THOSE VALUE WHICH WE WANT TO DROP.
index_list = filter.index
data.drop(index_list,inplace=True)
print()
print(data.ram_gb.value_counts(dropna=False))
data.ram_gb = data.ram_gb.apply(lambda x: 16 if x=='15.6' else x).astype('int32')
print(data.brand.value_counts())

#STEPS IN DATA ANALYSIS,
###UNIVARIATE ANALYSIS (analysing one column at a time)
###BIVARIATE ANALYSIS (two columns at a time,for continuous variables we can use scatter plot, for categorical varaibles we can use countplot and many more)
###MULTI-VARIATE ANALYSIS (combination of columns,use groupby, pivot table etc.)

plt.figure(figsize=(7,7))             # fancy way to plot categorical data
sns.countplot(data=data, x='brand', order=data.brand.value_counts(ascending=False).index)   #graph on count specific colomn
plt.tick_params(labelrotation=45)
#plt.show()

plt.figure(figsize=(6,6))
data.brand.value_counts().plot.barh()    # quick and more handy way to visualize categorical data 
#plt.show()

sns.countplot(data=data,x='ram_gb')
#plt.show()

print(data.isna().sum()) # to check if there are still 


#LETS COMBINE THE MODEL AND BRAND SO THAT WE KNOW THE MODEL IS OF WHICH BRAND.
data.model_brand = data.brand + " " + data.model
data.model_brand.astype("str")
print(data.model_brand)


data.model_brand.value_counts().head(13).plot.bar()       # Top 10 Models
#plt.show()

data['processor_brand'].value_counts().plot.bar(color = "g") 
#plt.show()
# Intel is most sold processor followed by AMD

data.brand.value_counts().head(10).plot.bar(color = "b")  
#plt.show()
# ASUS has highest sale, followed by DEll And Lenovo

data['processor_name'].value_counts().head(10).plot.bar() 
#plt.show()
#Most used processor is Core i5

print(data.columns)
cols = ['brand','ram_gb','processor_brand','processor_name']
#IN PREVIOUS CELLS WE HAVE ANALYSE ONE FEATURE AT A TIME, 
#BUT IT MAY BE TIRESOME PROCESS TO PLOT SO MANY FEATURES ONE BY ONE
#SO FOR PLOTTING ALL THE PLOTS AT ONE TIME,WE HAVE SOMETHING CALLED "SUBPLOTS" IN MATPLOTLIB

# fig, axes = plt.subplots(2,2,figsize=(17,10))   
# colors= ['b','g','r','c']   # different colors for each feature
# for i,ax in zip(range(len(cols)),axes.ravel()):   
#     data[cols[i]].value_counts().head(10).plot.barh(ax=ax,title= cols[i],color = colors[i])
fig,axes=plt.subplots(2,2,figsize=(17,10))   # Necessary step to specify number of axes and figure size( here axes refers to individual plots in whole figure,since we are plotting 4 plots, so we mention (2,2))
colors=['r','g','b','c']
for i,axes in zip(range(len(cols)),axes.ravel()):    #zip combines iterators(we can pass as many iterators as we want(here we have two 1.for iterating through columns and colors 2. for axes.ravel())
    data[cols[i]].value_counts().head(10).plot.bar(ax=axes,title= cols[i],color = colors[i])
#plt.show()    

numerical_var_cols = ['latest_price','discount','star_rating','ratings','reviews']
sns.boxplot(data=data,x='latest_price')
#plt.show()

figure, axes = plt.subplots(2,3,figsize=(14,10),sharex= True)
for col,ax in zip(numerical_var_cols,axes.ravel()):
    ax.boxplot(data=data,x=col)
    ax.annotate("Mean",xy=(1,data[col].mean()),xytext=(1.25,data[col].mean()),arrowprops=dict(facecolor='black'))
    ax.set_title(col)
    ax.xaxis.set_ticks_position('none')
    axes[1,2].set_axis_off()
#plt.show()    

print(data['star_rating'].describe())
print(data['star_rating'].value_counts())  
print(data[data['star_rating'] == 0])

#LAPTOPS HAVING STAR_RATINGS== 0 HAS 0 VALUE IN 'ratings' AND 'reviews' COLUMNS, 
#SO WE CANT SAY THE STAR_RATIING ==0 IS EXPLICITLY GIVEN OR THERE IS ALSO CHANCE THAT THERE IS NO RATING GIVEN TO THEM AT ALL.
#IT IS BETTER TO DROP VALUES HAVING STAR RATING ==0.


data2 = data[data.star_rating != 0]
# data2.star_rating.plot(kind='box')
# plt.show()


#WE CAN CONVERT NUMERICAL DATA INTO CATEGORIES.
data2['rating_cat'] = pd.cut(data2['star_rating'],bins=[1,2,3,4,5],labels=['1-2','2-3','3-4','4-5'],include_lowest=True)
print(data['latest_price'].describe())                            
data2['price_cat'] = pd.cut(data2['latest_price'],bins=[0,100000,200000,300000,400000,500000],labels=['<1lakh','1-2lakh','2-3lakhs','3-4lakhs','4-5lakhs'])
print(data.discount.describe())
data2['discount_cat'] = pd.cut(data2['discount'],bins=[0,10,20,30,40,50,60],labels=['<10','10-20','20-30','30-40','40-50','50-60'],include_lowest=True)

category= ['rating_cat','price_cat','discount_cat']
for cat in category:
    print(data2.groupby(cat).brand.count())


fig,axes= plt.subplots(1,3,figsize=(15,5))
for cat,ax in zip(category,axes.ravel()):
    data2.groupby(cat).brand.count().plot(kind='bar',x=cat,ax=ax)    
plt.show()

#MOST OF THE LAPTOPS ARE UNDER 1LAKHS
#4-5 RATING IS MOST GIVEN RATING

print(data2['price_cat'].value_counts())

print(data2[data2['price_cat'] == "4-5lakhs"])  # Costliest laptop

print(data2['rating_cat'].value_counts())
#LETS FIND OUT IS HIGH RATINGS RELATED TO OTHER FEATURES.
data2[['latest_price','star_rating']].plot(kind='scatter',x='latest_price',y='star_rating')
plt.show()

data2[data2['latest_price'] > 100000][['latest_price','star_rating']].plot(kind='scatter',x='latest_price',y='star_rating')  
plt.show()   #WE CAN SEE LAPTOPS HAVING HIGH PRICE ARE MOSTLY RATED MORE THAN 4.

data_low_rating_high_price = data2[(data2['star_rating'] < 1)&(data2['latest_price'] > 100000)]
print(data2[(data2['star_rating'] > 1)&(data2['latest_price'] > 100000)&(data2['star_rating'] < 3.5)])
#ANALYSING LAPTOPS WHICH HAVE PRICE MORE THAN 1 LAKH AND HAVE LOW RATINGS-
#TWO OUT OF FOUR LAPTOPS ARE LENOVO YOGA
#THIS IS THE REASON HIGH DISCONTS ARE GIVEN ON THESE LAPTOPS.
data2[data2['latest_price'] > 100000].brand.value_counts().plot.bar()

data2.star_rating.plot.hist(density=True)
data2.star_rating.plot.kde()#left skewed
plt.show()

data2.plot(kind='scatter',x='star_rating',y='latest_price',alpha=0.25)  # we can see variation in rating with respect to price
plt.show()

data2[(data2['star_rating'] > 1)&(data2['latest_price'] < 100000)].plot(kind='scatter',x='star_rating',y='latest_price',alpha=0.25)
plt.show()

data2[(data2['star_rating'] > 1)&(data2['latest_price'] < 100000)].plot(kind='hexbin',x='star_rating',y='latest_price',alpha=0.25,cmap=plt.cm.Greens,sharex=False,gridsize=25)
plt.show()

#WE CAN SEE DARK AREA RANGE FROM 40000 TO 65000 HAVING RATING BETWEEN 4 AND 5, WHICH MEANS MOST OF THE LAPTOPS ARE IN THIS RANGE.
print(data2[(data2['star_rating'] > 1)&(data2['latest_price'] < 100000)&(data2['star_rating'] <3)].processor_name.value_counts())

#WE CAN SEE CORE I3 AND RYZEN 5 ARE LEAST RATED UNDER 100000 PRICE RANGE.
print(data2[(data2['star_rating'] > 3)&(data2['latest_price'] < 100000)&(data2['star_rating'] <5)].ram_gb.value_counts())

data2.plot(kind="scatter",x="star_rating",y='discount') 
plt.show()



#FINALLY, IT IS VERY IMPORTANT TO MAKE CONCLUSIONS!!
###ASUS IS LEADING BRAND IN SELLING LAPTOPS.
###DELL INSPIRION AND ASUS VIVOBOOK ARE MOST SOLD MODELS.
###LAPTOPS HAVING HIGH PRICE ARE MOSTLY RATED MORE THAN 4.
###LENOVO YOGA HAVE LOW RATING INSPITE OF HAVING HIGH SPECS, HIGH PRICE.
###CORE I3 AND RYZEN 5 ARE LEAST RATED PROCESSORS UNDER 100000 PRICE RANGE. BUYING LAPTOP IN PRICE RANGE OF 40000-60000 PRICE RANGE,WITH INTEL I5 PROCESSOR IS MOST LIKELY THE BEST IDEA AS IN THIS RANGE LAPTOPS HAVE MOST 4-5 STAR_RATING



