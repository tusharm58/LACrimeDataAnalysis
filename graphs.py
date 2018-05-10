import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
	ax1.pie(slices, labels=tuple(labels),explode = explode, autopct='%1.1f%%',shadow=True, startangle=90)
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
	inverted_dict = getDict(x)
	print inverted_dict
	slices=[]
	labels=[]
	for a in sorted(inverted_dict.keys()):
	        slices.append(a)
	        labels.append(inverted_dict[a])
	print len(labels), len(slices)
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
	plt.ylabel('Number of Crimes')
	plt.xlabel('Year')
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
	plt.ylabel('Number of Crimes')
	plt.xlabel('Year')
	plt.show()

def juv(x,y,z,d):
	fire = ['UNKNOWN WEAPON/OTHER WEAPON', 'REVOLVER', 'HAND GUN', 'UNKNOWN FIREARM', 'SEMI-AUTOMATIC PISTOL',  'AIR PISTOL/REVOLVER/RIFLE/BB GUN','OTHER FIREARM', 'RIFLE', 'SHOTGUN', 'RELIC FIREARM', 'HECKLER & KOCH 93 SEMIAUTOMATIC ASSAULT RIFLE',  'STUN GUN',  'AUTOMATIC WEAPON/SUB-MACHINE GUN',  'UZI SEMIAUTOMATIC ASSAULT RIFLE', 'SAWED OFF RIFLE/SHOTGUN', 'HECKLER & KOCH 91 SEMIAUTOMATIC ASSAULT RIFLE', 'STARTER PISTOL/REVOLVER', 'UNK TYPE SEMIAUTOMATIC ASSAULT RIFLE', 'ASSAULT WEAPON/UZI/AK47/ETC', 'SEMI-AUTOMATIC RIFLE', 'MAC-10 SEMIAUTOMATIC ASSAULT WEAPON', 'ANTIQUE FIREARM', 'MAC-11 SEMIAUTOMATIC ASSAULT WEAPON']
	violent = ['INTIMATE PARTNER - SIMPLE ASSAULT', 'VANDALISM - MISDEAMEANOR ($399 OR UNDER)', 'CRIMINAL HOMICIDE', 'BATTERY - SIMPLE ASSAULT', 'ROBBERY', 'SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH 0007=02', 'ATTEMPTED ROBBERY', 'RESISTING ARREST', 'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT',  'CRIMINAL THREATS - NO WEAPON DISPLAYED',  'BATTERY WITH SEXUAL CONTACT', 'OTHER MISCELLANEOUS CRIME', 'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT', 'CHILD NEGLECT (SEE 300 W.I.C.)', 'OTHER ASSAULT', 'BOMB SCARE', 'EXTORTION', 'RAPE, FORCIBLE', 'INTIMATE PARTNER - AGGRAVATED ASSAULT', 'CHILD ANNOYING (17YRS & UNDER)', 'SHOTS FIRED AT INHABITED DWELLING', 'BATTERY POLICE (SIMPLE)', 'BRANDISH WEAPON', 'CRUELTY TO ANIMALS', 'SEXUAL PENTRATION WITH A FOREIGN OBJECT', 'PURSE SNATCHING', 'RAPE, ATTEMPTED', 'KIDNAPPING', 'BEASTIALITY, CRIME AGAINST NATURE SEXUAL ASSLT WITH ANIM0065', 'SEX, UNLAWFUL', 'ORAL COPULATION', 'DISCHARGE FIREARMS/SHOTS FIRED', 'CHILD STEALING', 'WEAPONS POSSESSION/BOMBING', 'CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT', 'KIDNAPPING - GRAND ATTEMPT', 'PURSE SNATCHING - ATTEMPT', 'BATTERY ON A FIREFIGHTER', 'DEFRAUDING INNKEEPER/THEFT OF SERVICES, OVER $400', 'INCITING A RIOT', 'LYNCHING - ATTEMPTED', 'REPLICA FIREARMS(SALE,DISPLAY,MANUFACTURE OR DISTRIBUTE)0132', 'CHILD ABANDONMENT', 'SHOTS FIRED AT MOVING VEHICLE, TRAIN OR AIRCRAFT', 'LYNCHING', 'MANSLAUGHTER, NEGLIGENT', 'INCEST (SEXUAL ACTS BETWEEN BLOOD RELATIVES)', 'SEX,UNLAWFUL(INC MUTUAL CONSENT, PENETRATION W/ FRGN OBJ0059', 'SEXUAL PENETRATION W/FOREIGN OBJECT', 'LEWD/LASCIVIOUS ACTS WITH CHILD']
	tempx = [0,0,0,0,0,0,0,0,0,0,0,0]
	tempy = [0,0,0,0,0,0,0,0,0,0,0,0]
	for a in range(len(x)):
		if type(x[a]) != float and type(y[a]) != float and type(z[a]) != float and type(d[a]) != float and y[a] in violent and 'Juv' in z[a]:
			temp = int(d[a][0:d[a].index('/')])
			if x[a] not in fire:
				tempx[temp-1] += 1
			tempy[temp-1] += 1
	permutation(tempx,tempy)


dataset = pd.read_csv('output.csv')
initiative(getData('Date Occurred'), getData('Status Description'), 'Juv', 'Juvinile Crime Rate Trend')
initiative(getData('Date Occurred'), getData('Crime Code Description'), 'CHILD' ,'Child Abuse Crime Rate Trend')
proposition47(getData('Date Occurred'), 'Crime Rate Trend')
juv(getData('Weapon Description'),getData('Crime Code Description'),getData('Status Description'),getData('Date Occurred'))
top10(getData('Crime Code Description'))
premise(getData('Crime Code Description'), getData('Premise Description'), 'STREET', 'Top 10 Crime Type on Street')
premise(getData('Crime Code Description'), getData('Premise Description'), 'SINGLE FAMILY', 'Top 10 Crime Type on Single Family Dwelling')
premise(getData('Crime Code Description'), getData('Premise Description'), 'MULTI-UNIT', 'Top 10 Crime Type on Multi-Unit Dwelling')
crimeVsVictimDescent(getData('Victim Descent'))
crimeVsPremiseType(getData('Premise Description'))
