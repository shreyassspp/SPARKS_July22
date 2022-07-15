#!/usr/bin/env python
# coding: utf-8



# In[ ]:


#Importing the Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None
sns.set_style('whitegrid')


# In[ ]:


import chart_studio.plotly as py
import plotly.graph_objs as go 
import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline() #Allows to use cuffly offline.


# In[ ]:


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# In[ ]:


#Reading the data into a Pandas Dataframe
df = pd.read_csv('globalterrorismdb_0718dist.csv',encoding='ansi',low_memory=False)
df=df[df['iyear']>=1980]

df.replace({'country_txt':{"Soviet Union":"Russia"}},inplace=True)
hues = ['attacktype1_txt','targtype1_txt']


# # Section 1 - Glancing at Global Terrorism Events and Identifying the hotspots

# In[ ]:


def plotWorldMap(tempDF,reason=None):
    ##Helper Function to Plot chloromap by different classifications.
    if(reason=='Losses Borne'):
        title="Amount in $"
    else:
        title = "Frequency of Events"
        
    data = dict(type='choropleth',
           locations=tempDF['index'],
           z=tempDF['country_txt'],
           locationmode='country names',
           colorbar = {'title':title},
           colorscale='Viridis',
           reversescale=True)
    
    if(reason!=None):
        reason=reason+" Across Past 40 Years"
    else:
        reason ="Across Past 40 Years"
        
    layout = dict(
        title = reason,
        geo = dict(
            showframe = False,
            projection = {'type':'natural earth'}
        )
    )
    choromap = go.Figure(data = [data],layout = layout)
    iplot(choromap,validate=False)


# In[ ]:


def printByReason(df,col):
     ##Helper Function to find out the 3 most common motivation for attacks
        
    mostCommonReasons = df[col].value_counts().head(3)
    mostCommonReasons = list(mostCommonReasons.index)
    df = df[df[col].isin(mostCommonReasons)]
    
    for reason in mostCommonReasons:
        tempDF = df[df[col]==reason]
        tempDF = tempDF['country_txt'].value_counts()
        tempDF = tempDF.reset_index()
        plotWorldMap(tempDF,reason)


# In[ ]:


tempDF = df['country_txt'].value_counts()
tempDF = tempDF.reset_index()
plotWorldMap(tempDF) #Generic Plot
printByReason(df,'attacktype1_txt')#Plot by Reasons for attacks 


# In[ ]:


#Plot by Loss Value
tempDF = df.groupby('country_txt')
tempDF = tempDF['propvalue'].sum()
tempDF=tempDF.reset_index()
tempDF.columns=['index','country_txt']
plotWorldMap(tempDF,'Losses Borne')


# # Section 2 - Generic Trends and DecadeWise Analysis

# ## Generic Trend of Events from 1980 - 2017

# In[ ]:


##Plots count of events across various years
plt.figure(figsize=(18,9))
df1 = df['iyear'].value_counts()
sns.lineplot(data=df1,palette='coolwarm')
plt.xlabel('Year',fontsize=15)
plt.ylabel('Number of Events',fontsize=15)


# ## Decadewise Analysis 

# ### Helper Functions  

# In[ ]:


def plotEvents(tempDF,hue):
    ##Helper Function to Plot Events
    plt.figure(figsize=(18,6))
    p=sns.countplot(x='iyear',data=tempDF,hue=hue,palette='coolwarm')
    p.set_xlabel("Year",fontsize=15)
    p.set_ylabel("No. of Events",fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #Relocates the legend.
    plt.show();


# In[ ]:


def printData(tempDF,hue):
        #Helper function to Print the Decade and decreasing order of events 
        ##and Selects top 5 motives/victims across each decade.
        
        print(color.BOLD+"Years {}-{}".format(tempDF['iyear'].min(),tempDF['iyear'].max())+color.END)
        x=tempDF[hue].value_counts().head(5)
        indexValues = list(x.index.values)
        s=""
        i=1
        
        for index in indexValues:
            x = " {}. ".format(i)
            i=i+1
            s=s+x+index
        print(color.BOLD+"Decreasing Order- {}".format(s)+color.END)
        print(color.BOLD+ "Most Fatal Year is {}".format(tempDF['iyear'].value_counts().idxmax())+color.END)
        
        tempDF=tempDF[tempDF[hue].isin(indexValues)]
        return tempDF


# In[ ]:


def decadeWise(df,hue):
    #Helper function for decade wise analysis.Display only top 5 reasons.
     x = 1980 
     y = 1990
    
     for i in range(4):
        tempDF = df[(df['iyear']>=x) & (df['iyear']<y)]
        tempDF=printData(tempDF,hue)
        plotEvents(tempDF,hue)
        x=x+10
        y=y+10


# ## a) By motives for Attacks

# In[ ]:


decadeWise(df,hues[0])


# ## b) By Victims of Attacks

# In[ ]:


decadeWise(df,hues[1])


# ### Helper Functions for Damage Analysis

# In[ ]:


##Remove Cells not having data on Property Damage Value and their extent.
propertyDF = df[(df['propextent_txt'].notna()) & (df['propvalue'].notna()) ]
majorEvents = df['propextent_txt'].unique()[3:]


# In[ ]:


#Get A list of Catastrophic and Worst Years in the decade.
#Catastrophic Events - Loss of more than 1 Billion USD.
#Major Property Damage - Loss of more than 1 Million USD but less than 1 Billion USD

def prettyPrint(years):
    
    ##Helper Function to print years
    years=years.sort_values()
    ans =""
    for year in years:
        ans=ans+" "+str(year)+","
        
    ans=ans[:len(ans)-1]
    print(color.BOLD+"{}".format(ans)+color.END)
    
    
def getWorstYearsbyDamage(tempDF,col1='propextent_txt'):
    
    ##Helper Function to determine years of heavy damage.
    df2 = tempDF[tempDF[col1]==majorEvents[1]]
    worstYears=df2['iyear'].value_counts()
    
    if(worstYears.size>0):
        worstYears = worstYears.index
        print(color.BOLD+"Years having catastrophic property damage:"+color.END,end="")
        prettyPrint(worstYears)
    
    df2=tempDF[tempDF[col1]==majorEvents[0]]
    worstYears=df2['iyear'].value_counts()
    
    if(worstYears.size>0):
        worstYears = worstYears.index
        print(color.BOLD+"Years having major property damage:"+color.END,end="")
        prettyPrint(worstYears)


# In[ ]:


def plotLoss(tempDF,col1,col2):
    
    ## Helper Function to plot total loss across the decade and also to bifurcate on the basis
    ## of severity of the events.
    
    #Amount of loss is summation of col1 and 
    #Types of incidents are indicated by col2
    year_max =tempDF['iyear'].max()
    year_min =tempDF['iyear'].min()
        
    tempDF.loc[:,[col1]]/=1e6
    
    totalDamage = tempDF[col1].sum()
    
    print(color.BOLD+"Decade Summary"+color.END)
    print(color.BOLD+"Years: {}-{}".format(year_min,year_max)+color.END)
    print(color.BOLD+"Total Damage is {} million $".format(round(totalDamage,2)))
    
    #Find Catastrophic and Major Events Years.
    
    getWorstYearsbyDamage(tempDF)
    
    plt.figure(figsize=(15,6))
    p=sns.countplot(x='iyear',data=tempDF,hue=col2,palette='coolwarm')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #Relocates the legend.
    p.set_ylabel("Frequency",fontsize=15)
    p.set_xlabel("Years",fontsize=15)
    title ="Classification by Severity"
    p.set_title(title,fontsize=18)
    plt.show()
    
    
    #Plot loss across the decade.
    
    plt.figure(figsize=(12.5,6))
    title ="Loss Across the Decade"
    
        
    tempDF = tempDF[["propextent_txt","propvalue","iyear","country_txt"]]
    tempDF=tempDF[tempDF["propvalue"].notna()]
    
    newDF = tempDF.groupby("iyear")
    newDF = newDF["propvalue"].sum()
    
    p=sns.lineplot(data=newDF,palette='coolwarm')
    
    p.set_title(title,fontsize=18)
    p.set_xlabel("Years",fontsize=15)
    p.set_ylabel("Amount in Million USD",fontsize=15)
    plt.show()


# In[ ]:


def propDamageDecadeWise(tempDF,col1,col2):
    #Helper Function to plot damage decadewise
    x=1980
    y=1990
    
    for i in range(4):
        tempDF1 = tempDF[(tempDF['iyear']>=x) & (tempDF['iyear']<y)]
        plotLoss(tempDF1,col1,col2)
        x=x+10
        y=y+10


# ## c) By Decadewise Loss

# In[ ]:


propDamageDecadeWise(propertyDF,'propvalue','propextent_txt')


# ## d) Comparison of most common motives and victims

# In[ ]:


def getCount(df,attackType,victimType):
    #Helper Function to return count of a certain attack type and victim type
    df = df[(df[hues[0]]==attackType) & 
           (df[hues[1]]==victimType)]
    df.head
    return len(df.axes[0])


# In[ ]:


#Finding the 5 most common victims and motives for attacks
x = df[hues[0]].value_counts()
x=x.head()
rowIndex = list(x.index.values)

y= df[hues[1]].value_counts()
y=y.head()
colIndex = list(y.index.values)


# In[ ]:


#Forming the Dataframe with reasons and victims.
tempDF = pd.DataFrame(index = rowIndex,columns=colIndex)
for i in range(5):
    for j in range(5):
        rowName = rowIndex[i]
        colName = colIndex[j]
        tempDF.iloc[i][colName]=getCount(df,rowName,colName)

tempDF.head()


# In[ ]:


#Plot each column one by one with it's reason.
for col in colIndex:
    plt.figure(figsize=(15,5))
    p=sns.barplot(x=rowIndex,y=col,data=tempDF,palette='coolwarm')
    p.set_title(col,fontsize =18)
    p.set_ylabel("")
    plt.show()


# # Section 3 - Case Study Comparing Data of USA, China and India.

# In[ ]:


#Forming a dataframe having only relevant countries
#DataFrame has data only for USA, Russia, China and India.

countries = ["United States","Russia","India"]
countryWiseDF = df[df["country_txt"].isin(countries)]


# ## a) Comparing Number of Events Across the Decades

# In[ ]:


decadeWise(countryWiseDF,'country_txt')


# ### Helper Functions 

# In[ ]:


def plotReasons(tempDF,col):
    #Helper Function to plot country wise data according to the column passed in.
    
    if(col==hues[0]):
        title ="Motives for Attacks"
    elif col==hues[1]:
        title="Victims"
    else:
        title = "Weapons Used"
    
    newDF = tempDF[col].value_counts()
    newDF = list(newDF.index)[:6]
    tempDF = tempDF[tempDF[col].isin(newDF)]
    
    plt.figure(figsize=(16,8))
    p=sns.countplot(x=col,data=tempDF,hue="country_txt",palette='coolwarm')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #Relocates the legend
    p.set_title(title,fontsize=18)
    p.set_ylabel("Number of Attacks",fontsize=15)
    p.set_xlabel("")
    plt.show()


# ## b) Comparing Motives,Victims and Weapons used across the three countries 

# In[ ]:


plotReasons(countryWiseDF,hues[0])
plotReasons(countryWiseDF,hues[1])
plotReasons(countryWiseDF,"weaptype1_txt")


# ## c) Losses Borne by them.

# In[ ]:


def plotLossTogether(tempDF): 

    ## Helper Function to plot losses of all three nations across the time span
    
    tempDF = tempDF[["propvalue","iyear","country_txt"]]
    tempDF = tempDF.groupby(["country_txt","iyear"])
    tempDF = tempDF['propvalue'].sum()
    
    tempDF = tempDF.reset_index()
    tempDF.columns=['Country','Year','Value']

    plt.figure(figsize=(16,8))
    p=sns.lineplot(x='Year',y='Value',hue='Country',data=tempDF,palette='coolwarm')
    p.set_ylabel("Amount in 100 M USD",fontsize=15)
    p.set_xlabel("Years",fontsize=15)
    p.set_title("Losses Borne",fontsize=18)
    plt.show()


# In[ ]:


plotLossTogether(countryWiseDF)


# ## Individual Losses Across the Decades

# In[ ]:


def plotLossCountry(tempDF,col1,countryName):
    
    #Helper Function for plotting loss country wise
    
    tempDF.loc[:,[col1]]/=1e6    
    totalDamage = tempDF[col1].sum()
   
    print(color.BOLD+"Total Loss borne by {} is {} million $".format(countryName,round(totalDamage,2)))
    
    tempDF = tempDF[[col1,"iyear","country_txt"]]
    tempDF=tempDF[tempDF["propvalue"].notna()]
    
    newDF = tempDF.groupby("iyear")
    newDF = newDF["propvalue"].sum()
    
    plt.figure(figsize=(15,6)) 
    title ="Loss Across the Years for {}".format(countryName)
    
    p=sns.lineplot(data=newDF,palette='coolwarm')
    p.set_title(title,fontsize=18)
    p.set_xlabel("Years",fontsize=15)
    p.set_ylabel("Amount in Million USD",fontsize=15)
    plt.show()


# In[ ]:


for country in countries:
        tempDF=countryWiseDF[countryWiseDF["country_txt"]==country]
        plotLossCountry(tempDF,"propvalue",country)


# 
# # Thank You !!!

# In[ ]:




