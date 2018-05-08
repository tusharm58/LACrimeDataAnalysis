import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import norm
import seaborn as sns
import math
from pylab import *
from decimal import *


def victimPreprocessing(x):
	victimDescent=['X', 'O', 'B', 'W', 'H']
	# Combing all the fields except the above into O
	for a in range(len(x)):
	    if type(x[a]) != float and x[a] not in victimDescent:
	        x[a] = 'O'
	return x

def pie(slices, labels, explode, title):
	fig1, ax1 = plt.subplots()
	ax1.pie(slices, labels=tuple(labels),explode = explode, autopct='%1.1f%%',
	        shadow=True, startangle=90)
	ax1.axis('equal')
	plt.title(title)
	plt.show()

def bar(val, labels, xLabel, yLabel, Title):
	pos = arange(10)+.5
	barh(pos,val, align='center')
	yticks(pos, tuple(labels))
	xlabel(xLabel)
	ylabel(yLabel)
	title(Title)
	show()


def top10(x):
	inverted_dict = getDict(x)
	slices=[]
	labels=[]
	for a in sorted(inverted_dict.keys()):
	    slices.append(a)
	    labels.append(inverted_dict[a])
	slices = slices[-10:]
	labels = labels[-10:]
	bar(slices, labels, 'Number of Crimes', 'Crime Types', 'Top 10 Crime Types')
	

def premise(z,x,p, title):
	dix={}
	for a in range(len(x)):
	    if type(z[a]) != float and type(x[a]) != float and p in x[a]:
	        if z[a] not in dix:
	            dix[z[a]] = 1
	        else:
	            dix[z[a]] += 1
	inverted_dict = dict([[v,k] for k,v in dix.items()])
	slices=[]
	labels=[]
	for a in sorted(inverted_dict.keys()):
	    slices.append(a)
	    labels.append(inverted_dict[a])
	slices = slices[-10:]
	labels = labels[-10:]
	bar(slices, labels, 'Number of Crimes', 'Crime Types', title)


# This will give us a pie graph for Crime Vs Victim Descent
def crimeVsVictimDescent(x):
	x = victimPreprocessing(x)
	inverted_dict = getDict(x)
	slices=[]
	labels=[]
	for a in sorted(inverted_dict.keys()):
	        slices.append(a)
	        labels.append(inverted_dict[a])
	dictionary = {'X': 'Unknown', 'O':'Other', 'B': 'African American', 'W':'Caucasian', 'H':'Hispanic/Latin/Mexican'}
	for a in range(len(labels)):
		labels[a] = dictionary[labels[a]]
	explode = (0,0,0,0,0)
	pie(slices, labels, explode,'Crime Vs Victim Descent')
	

# This will give us a pie graph for Crime Vs Premise Type
def crimeVsPremiseType(x):
	inverted_dict = getDict(x)
	slices=[]
	labels=[]
	for a in sorted(inverted_dict.keys()):
	        slices.append(a)
	        labels.append(inverted_dict[a])
	labels = labels[-5:]
	slices = slices[-5:]
	explode = (0,0,0,0,0.1)  
	pie(slices, labels, explode, 'Crime Vs Premise Type')



def getDict(x):
	dix={}
	for a in x:
		if type(a) != float:
		    if a in dix:
		        dix[a]+=1
		    else:
		        dix[a] = 1
	return dict([[v,k] for k,v in dix.items()])


def getData(column):
	x = dataset[[column]]
	x_array = x.values
	x = x_array.flatten()
	return x

def permutation(X,Y):
	X_mean = float(sum(X)/len(Y))
	Y_mean = float(sum(Y)/len(Y))
	N = len(X)+len(Y)
	T_observed = abs(X_mean - Y_mean)
	sample = np.concatenate((X, Y), axis=0)
	count = 0
	visited = []
	for i in range(100000):
		temp = np.random.permutation(sample) 
		if str(temp) not in visited:
			visited.append(str(temp))
			tempX = temp[0:10]
			tempY = temp[10::]
			tempX_mean = np.mean(tempX)
			tempY_mean = np.mean(tempY)
			tempDiff = round(abs(tempX_mean-tempY_mean),3)
			if tempDiff > T_observed:
				count += 1
		else:
			pass
	print "Number of Ti > T_observed are:",count
	p_value = Decimal(count)/Decimal(10**6)
	print "p-value is: ", p_value,"\n"

def initiative(x,y,t,title):
	tempx = [0,0,0,0,0,0,0,0,0,0,0,0]
	tempy = [0,0,0,0,0,0,0,0,0,0,0,0]
	for a in range(len(x)):
	    if type(x[a]) != float and type(y[a]) != float and t in y[a]:
			temp = int(x[a][0:x[a].index('/')])
			if (x[a][-4:] == '2010' or x[a][-4:] == '2011' or x[a][-4:] == '2012' ):
			    tempy[temp-1] += 1
			elif (  x[a][-4:] == '2013' or x[a][-4:] == '2014' or x[a][-4:] == '2015' or x[a][-4:] == '2016' or x[a][-4:] == '2017' ) :
				tempx[temp-1] += 1
	print ("Permutation Test")
	permutation(tempx,tempy)
	data = [0,0,0,0,0,0,0,0,0]
	for a in range(len(x)):
		if type(x[a]) != float and type(y[a]) != float and t in y[a]:
			data[int(x[a][-1:])] += 1	
	plots=[]
	a, = plt.plot([2010,2011,2012,2013,2014,2015,2016,2017,2018], data,label = 'Crime Rate')
	plots.append(a)
	plt.legend(handles=plots)
	plt.title(title)
	plt.show()

def proposition47(x,title):
	tempx = [0,0,0,0,0,0,0,0,0,0,0,0]
	tempy = [0,0,0,0,0,0,0,0,0,0,0,0]
	for a in range(len(x)):
	    if type(x[a]) != float :
			temp = int(x[a][0:x[a].index('/')])
			if (x[a][-4:] == '2010' or x[a][-4:] == '2011' or x[a][-4:] == '2012' or x[a][-4:] == '2013'):
			    tempy[temp-1] += 1
			elif ( x[a][-4:] == '2014' or x[a][-4:] == '2015' or x[a][-4:] == '2016' or x[a][-4:] == '2017' ) :
				tempx[temp-1] += 1
	print ("Permutation Test")
	permutation(tempx,tempy)
	data = [0,0,0,0,0,0,0,0]
	for a in range(len(x)):
		if type(x[a]) != float and int(x[a][-1:]) != 8:
			data[int(x[a][-1:])] += 1	
	plots=[]
	a, = plt.plot([2010,2011,2012,2013,2014,2015,2016,2017], data,label = 'Crime Rate')
	plots.append(a)
	plt.legend(handles=plots)
	plt.title(title)
	plt.show()




dataset = pd.read_csv('wo.csv')
initiative(getData('Date Occurred'), getData('Status Description'), 'Juv', 'Juvinile Crime Rate Trend')
initiative(getData('Date Occurred'), getData('Crime Code Description'), 'CHILD' ,'Child Abuse Crime Rate Trend')
proposition47(getData('Date Occurred'), 'Crime Rate Trend')
top10(getData('Crime Code Description'))
premise(getData('Crime Code Description'), getData('Premise Description'), 'STREET', 'Top 10 Crime Type on Street')
premise(getData('Crime Code Description'), getData('Premise Description'), 'SINGLE FAMILY', 'Top 10 Crime Type on Single Family Dwelling')
premise(getData('Crime Code Description'), getData('Premise Description'), 'MULTI-UNIT', 'Top 10 Crime Type on Multi-Unit Dwelling')
crimeVsVictimDescent(getData('Victim Descent'))
crimeVsPremiseType(getData('Premise Description'))
