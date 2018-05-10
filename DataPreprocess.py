import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import re
import random
import time


def get_distance(lat1, long1, lat2, long2):
    R = 6373.0
    lat1 = np.radians(lat1)
    long1 = np.radians(long1)
    lat2 = np.radians(lat2)
    long2 = np.radians(long2)
    dlong = long2 - long1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    meters = km * 1000
    return meters


def get_lat_long(location):
    list_loc = str(location).strip('(),')
    loc1 = list_loc.split(',')
    if(len(loc1) is not 2):
        return -1,-1
    lat1 = loc1[0]
    long1 = loc1[1].strip()
    lat1 = float(lat1)
    long1 = float(long1)
    return lat1,long1

def plot_for_area(area_name,popular_area_location):
    df_area = ds2[ds2['Area Name']  == area_name].copy()
    pop_lat,pop_long = get_lat_long(popular_area_location)
    distance = list()
    loc = df_area['Location ']
    for location in loc:
        lat_loc,long_loc = get_lat_long(location)
        if(lat_loc is -1):
            continue
        dist_temp = get_distance(pop_lat,pop_long,lat_loc,long_loc)
        if(dist_temp > 40000):
            continue
        distance.append(dist_temp)
    return distance   

def getData(column):
    x = ds[[column]]
    x_array = x.values
    x = x_array.flatten()
    return x

def victimDescent():
    temp = []
    x = getData('Victim Descent')
    dictionary = {'X': 'Unknown',  'B': 'African American', 'W':'Caucasian', 'H':'Hispanic/Latin/Mexican'}
    for a in x:
        if a in dictionary.keys():
            temp.append(dictionary[a])
        else:
            temp.append('Other')
    return temp

def distance():
    area = ds2["Area Name"]
    loc = ds2["Location "]
    Data = [["77th Street","33.938739, -118.241047"],["Olympic","34.044736, -118.264549"],["Central","34.1, -118.333333"],["Southeast","33.9, -118.166667"],["Topanga","34.045053, -118.563824"],["Northeast","34.11194, -118.19806"],["Foothill","34.140635, -118.044354"],["Mission","34.22472, -118.44889"],["Van Nuys","34.20114, -118.50113"],["Newton","34.037168, -118.256404"],["Hollywood","34.136518, -118.356051"],["Rampart","34.0792, -118.258"],["N Hollywood","34.1919 , -118.4011"],["West Valley","34.179084, -118.413782"],["Pacific","34.042738, -118.55235"],["Devonshire","34.2582, -118.5392"],["Harbor","33.729186, -118.262015"],["Hollenbeck","34.039956, -118.21804"],["West LA","34.0380, -118.4532"],["Wilshire","34.05, -118.2593"]]
    total = list()
    for data in Data:
        distance = plot_for_area(data[0],data[1])
        total = total + distance
    return total

def crimeType():
    temp = []
    x = getData('Crime Code Description')
    violent = ['INTIMATE PARTNER - SIMPLE ASSAULT', 'VANDALISM - MISDEAMEANOR ($399 OR UNDER)', 'CRIMINAL HOMICIDE', 'BATTERY - SIMPLE ASSAULT', 'ROBBERY', 'SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH 0007=02', 'ATTEMPTED ROBBERY', 'RESISTING ARREST', 'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT', 'THROWING OBJECT AT MOVING VEHICLE', 'DEFRAUDING INNKEEPER/THEFT OF SERVICES, $400 & UNDER', 'CRIMINAL THREATS - NO WEAPON DISPLAYED', 'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS) 0114', 'BATTERY WITH SEXUAL CONTACT', 'OTHER MISCELLANEOUS CRIME', 'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT', 'CHILD NEGLECT (SEE 300 W.I.C.)', 'OTHER ASSAULT', 'BOMB SCARE', 'EXTORTION', 'RAPE, FORCIBLE', 'INTIMATE PARTNER - AGGRAVATED ASSAULT', 'CHILD ANNOYING (17YRS & UNDER)', 'SHOTS FIRED AT INHABITED DWELLING', 'BATTERY POLICE (SIMPLE)', 'BRANDISH WEAPON', 'CRUELTY TO ANIMALS', 'SEXUAL PENTRATION WITH A FOREIGN OBJECT', 'PURSE SNATCHING', 'RAPE, ATTEMPTED', 'KIDNAPPING', 'BEASTIALITY, CRIME AGAINST NATURE SEXUAL ASSLT WITH ANIM0065', 'SEX, UNLAWFUL', 'ORAL COPULATION', 'DISCHARGE FIREARMS/SHOTS FIRED', 'CHILD STEALING', 'WEAPONS POSSESSION/BOMBING', 'CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT', 'KIDNAPPING - GRAND ATTEMPT', 'PURSE SNATCHING - ATTEMPT', 'BATTERY ON A FIREFIGHTER', 'DEFRAUDING INNKEEPER/THEFT OF SERVICES, OVER $400', 'INCITING A RIOT', 'LYNCHING - ATTEMPTED', 'REPLICA FIREARMS(SALE,DISPLAY,MANUFACTURE OR DISTRIBUTE)0132', 'CHILD ABANDONMENT', 'SHOTS FIRED AT MOVING VEHICLE, TRAIN OR AIRCRAFT', 'LYNCHING', 'MANSLAUGHTER, NEGLIGENT', 'INCEST (SEXUAL ACTS BETWEEN BLOOD RELATIVES)', 'SEX,UNLAWFUL(INC MUTUAL CONSENT, PENETRATION W/ FRGN OBJ0059', 'SEXUAL PENETRATION W/FOREIGN OBJECT', 'LEWD/LASCIVIOUS ACTS WITH CHILD']
    for a in x:
        if a in violent:
            temp.append('Violent')
        else:
            temp.append('Non-Violent')
    return temp

def date_to_epoch():
    temp = []
    date_format = '%m/%d/%Y %h:%m'
    x = getData('Date Occurred')
    y = getData('Time Occurred')
    for a in range(len(x)):
        date_of_crime = str(x[a])
        time_of_crime = str(y[a])
        print time_of_crime
        time_of_crime = time_of_crime[:2] + ":" + time_of_crime[2:] if len(time_of_crime) > 3 else "0" + time_of_crime[:1] + ":" + time_of_crime[1:]
        print time_of_crime,"\n"
        if len(time_of_crime) < 4:
            time_of_crime += '00'
        epoch = int(time.mktime(time.strptime(date_of_crime + " " + time_of_crime, date_format)))#int(time.mktime(time.strptime(date_of_crime + " " + time_of_crime, date_format)))
        temp.append(epoch)
        print date_of_crime, time_of_crime, epoch, date_of_crime + " " + time_of_crime
    return temp

# Change it with the original csv file name
ds = pd.read_csv('originalData.csv')
ds2 = ds[["DR Number","Area Name","Location "]]
ds['Epoch Occurred'] = pd.Series(date_to_epoch())
ds['distance'] = pd.Series(distance())
ds['Victim Descent'] = pd.Series(victimDescent())
ds['Crime Type'] = pd.Series(crimeType())
ds.to_csv('output.csv', index = True)
