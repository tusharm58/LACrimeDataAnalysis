import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import re
import random
import datetime

#URLs of the three CSVs
properties_url = r'/Users/santwana/Desktop/testing/LACrimeDataAnalysis/Crime_Data_from_2010_to_Present.csv'
ds = pd.read_csv(str(properties_url), low_memory = False)
print(ds.head(3))

print("length of ds :",len(ds))
ds2 = ds[['DR Number','Area Name','Location ']]
print(ds2.head(2))

def get_distance(lat1, long1, lat2, long2):
    # approximate radius of earth in km
    R = 6373.0
#     print(" lat,long : ",lat1, long1, lat2, long2)
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
    #meters = str(meters).split('.')[0]
    #return int(meters)


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

from scipy.interpolate import UnivariateSpline
#from matplotlib import pyplot as plt
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
from matplotlib import pyplot as pls
from numpy import linspace
from scipy.stats.kde import gaussian_kde
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt9

def plot_for_area(area_name,popular_area_location):
    # print("In plot for area")
    df_area = ds2[ds2['Area Name']  == area_name].copy()
    pop_lat,pop_long = get_lat_long(popular_area_location)
#     print(df_area)
#     print(pop_lat,pop_long)
    print()
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
    print("distance")
    # pl.xlabel(area_name)
    # pl.ylabel('crime')
    # pl.hist(distance, bins = 500, facecolor = 'green', alpha = 0.75)
    # pl.axis([400, 20000, 0, 3000])
    # pl.grid(True)
    # print("plot for area finished")
    # filename = "/Users/santwana/Desktop/testing/Plots1/" + area_name + ".png"
    # plt.savefig(filename)
    # plt.show()
    
    # std_dev1 = np.std(distance, ddof = 0)
    # print("std_dev1 is:", std_dev1)

    # mean1 = np.mean(distance)
    # print("mean1 is:", mean1) 

#     kde = gaussian_kde( distance )
#     # these are the values over wich your kernel will be evaluated
#     dist_space = linspace( min(distance), max(distance), 100 )
#     # plot the results
#     plt.plot( dist_space, kde(dist_space) )

# #     n, bins, patches = pl.hist(distance, 500, normed = 10, facecolor ='green', alpha = 0.75)
#     filename1 = "/Users/santwana/Desktop/testing/Plots1/PDF/" + area_name + ".png"
#     plt.savefig(filename1)
#     plt.show()
    
#     sorted_distance = np.sort(distance)
#     p = 1. * np.arange(len(sorted_distance))/(len(sorted_distance) - 1)
#     pls.plot(sorted_distance, p)
#     filename4 = "/Users/santwana/Desktop/testing/Plots1/CDF/" + area_name + ".png"
#     pl.savefig(filename4)
#     pls.show()

    return distance   

# from scipy.stats import ks_2samp
# import pylab 
# def KS_test(area_name,popular_area_location):
#     df_area = ds2[ds2['Area Name']  == area_name].copy()
#     pop_lat,pop_long = get_lat_long(popular_area_location)
# #     print(df_area)
# #     print(pop_lat,pop_long)
#     print()
#     distance = list()
#     loc = df_area['Location ']
   
#     for location in loc:
#         lat_loc,long_loc = get_lat_long(location)
#         if(lat_loc is -1):
#             continue
#         dist_temp = get_distance(pop_lat,pop_long,lat_loc,long_loc)
        
#         if(dist_temp > 40000):
#             continue
#         distance.append(dist_temp)
#     return distance 

# df_Olympic = KS_test("Olympic", "34.044736, -118.264549")
# df_Hollywood = KS_test("Hollywood", "34.136518, -118.356051")
# # print("OLYMPIC:",df_Olympic)
# print("OLYMPIC LENGTH: ", len(df_Olympic))
# # print("HOLLYWOOD",df_Hollywood)
# print("HOLLYWOOD LENGTH: ", len(df_Hollywood))
# print(ks_2samp(df_Olympic, df_Hollywood))

# from matplotlib import pyplot as pls
# sorted_df_Olympic = np.sort(df_Olympic)
# sorted_df_Hollywood = np.sort(df_Hollywood) 
# p1 = 1. * np.arange(len(sorted_df_Olympic))/(len(sorted_df_Olympic) - 1)
# pls.plot(sorted_df_Olympic, p1, 'g', label='True CDF Olympic')
# p2 = 1. * np.arange(len(sorted_df_Hollywood))/(len(sorted_df_Hollywood) - 1)
# pls.plot(sorted_df_Hollywood, p2, 'blue', label='True CDF Hollywood')
# pls.xlabel("Distance")
# pls.ylabel("Cumulative Probability")

# pls.title("CDF for Olympic and Hollyoowd")
# pylab.legend(loc = 'lower right')
# filenamecdfs = "/Users/santwana/Desktop/testing/Plots1/CDF/ComparingHollywood&Olympic.png"
# pls.savefig(filenamecdfs)
# pls.show()



# from scipy.stats import ks_2samp

# def KS_test(area_name,popular_area_location):
#     df_area = ds2[ds2['Area Name']  == area_name].copy()
#     pop_lat,pop_long = get_lat_long(popular_area_location)
# #     print(df_area)
# #     print(pop_lat,pop_long)
#     print()
#     distance = list()
#     loc = df_area['Location ']
   
#     for location in loc:
#         lat_loc,long_loc = get_lat_long(location)
#         if(lat_loc is -1):
#             continue
#         dist_temp = get_distance(pop_lat,pop_long,lat_loc,long_loc)
        
#         if(dist_temp > 40000):
#             continue
#         distance.append(dist_temp)
#     return distance 

# df_Southeast = KS_test("Southeast", "33.9, -118.166667")
# df_Wilshire = KS_test("Wilshire", "34.05, -118.2593")
# # print("Southeast:",df_Southeast)
# print("Southeast LENGTH: ", len(df_Southeast))
# # print("HOLLYWOOD",df_Southeast)
# print("Wilshire LENGTH: ", len(df_Wilshire))
# print(ks_2samp(df_Southeast, df_Wilshire))



# import numpy as np
# from scipy.stats import norm
# import matplotlib.pyplot as plotOLy

# def get_dist(area_name,popular_area_location):
#     df_area = ds2[ds2['Area Name']  == area_name].copy()
#     pop_lat,pop_long = get_lat_long(popular_area_location)
# #     print(df_area)
# #     print(pop_lat,pop_long)
#     print()
#     distance = list()
#     loc = df_area['Location ']
   
#     for location in loc:
#         lat_loc,long_loc = get_lat_long(location)
#         if(lat_loc is -1):
#             continue
#         dist_temp = get_distance(pop_lat,pop_long,lat_loc,long_loc)
        
#         if(dist_temp > 40000):
#             continue
#         distance.append(dist_temp)
#     return distance 

# dist = get_dist("Olympic", "34.044736, -118.264549")

# std_dev_O = np.std(dist, ddof = 0)
# print("std_dev_total is:", std_dev_O)

# mean_O = np.mean(dist)
# print("mean_total is:", mean_O)
# # Fit a normal distribution to the data:
# mu, std = norm.fit(dist)

area = ds2["Area Name"]
# area = str(area)
loc = ds2["Location "]
print("length : ", len(loc))
# print("length of ds2 : ",len(ds2))
# print("area length",len(area))
#print("loaction is dchjsgchsdbjchsdb",loc)
# print(ds2["Area Name"])

Data = [["77th Street","33.938739, -118.241047"],["Olympic","34.044736, -118.264549"],["Central","34.1, -118.333333"],["Southeast","33.9, -118.166667"],["Topanga","34.045053, -118.563824"],["Northeast","34.11194, -118.19806"],["Foothill","34.140635, -118.044354"],["Mission","34.22472, -118.44889"],["Van Nuys","34.20114, -118.50113"],["Newton","34.037168, -118.256404"],["Hollywood","34.136518, -118.356051"],["Rampart","34.0792, -118.258"],["N Hollywood","34.1919 , -118.4011"],["West Valley","34.179084, -118.413782"],["Pacific","34.042738, -118.55235"],["Devonshire","34.2582, -118.5392"],["Harbor","33.729186, -118.262015"],["Hollenbeck","34.039956, -118.21804"],["West LA","34.0380, -118.4532"],["Wilshire","34.05, -118.2593"]]

print("starting process")
total = list()
for data in Data:
    # print("data : ", data)
    distance = plot_for_area(data[0],data[1])
    total = total + distance
    # print(data[0],"finished")

print(total[:5])
# csv_input = pd.read_csv('Crime_Data.csv')

ds['Distance'] = pd.Series(total)
ds.to_csv('output.csv', index = True)

# Plot the histogram.
# plotOLy.hist(total, bins = 300, normed = True, alpha = 0.6, color='g')

# # Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 200)
# p = norm.pdf(x, mu, std)
# plotOLy.plot(x, p, 'k', linewidth = 2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plotOLy.title(title)
# filen = "/Users/santwana/Desktop/testing/New/Olympic.png"
# plotOLy.savefig(filen)
# plotOLy.show()

import csv
import io
import time
import pandas as pd

from pyspark import SparkConf, SparkContext

# Original File Features (With Corresponding Indices)
headers = {'Crime Code': 7, 'Status Code': 17, 'Time Occurred': 3, 'Date Occurred': 2, 'Weapon Used Code': 15,
           'Reporting District': 6, 'Crime Code Description': 8, 'Crime Code 3': 21, 'Victim Age': 10, 'DR Number': 0,
           'Premise Code': 13, 'Weapon Description': 16, 'Premise Description': 14, 'MO Codes': 9, 'Address': 23,
           'Crime Code 1': 19, 'Area Name': 5, 'Victim Descent': 12, 'Crime Code 4': 22, 'Location': 25,
           'Date Reported': 1, 'Victim Sex': 11, 'Cross Street': 24, 'Crime Code 2': 20,
           'Status Description': 18, 'Area ID': 4}

# Selected Features (To add new feature, must add new dict entry with feature name and index)
filtered_headers = {"Date Reported": 0, "Date Occurred": 1, "Time Occurred": 2, "Area ID": 3, "Area Name": 4,
                    "Victim Age": 5, "Victim Sex": 6, "Victim Descent": 7, "Location": 8, "Epoch Occurred": 9}

date_format = '%m/%d/%Y %H:%M'

# IO files
raw_data_file = "sample.csv"
processed_data_file = "sample_new1.csv"


# Filter by selected features
def filter_cols(line):
    input_csv = io.StringIO(line)
    for row in csv.reader(input_csv):
        selected_cols = {}
        for header in filtered_headers.keys():
            if headers.get(header) is not None:
                selected_cols[filtered_headers.get(header)] = row[headers.get(header)]
        return selected_cols


# Convert date to epoch
def date_to_epoch(row):
    date_of_crime = str(row[filtered_headers.get('Date Occurred')]).strip()
    time_of_crime = str(row[filtered_headers.get('Time Occurred')]).strip()
    time_of_crime = time_of_crime[:2] + ":" + time_of_crime[2:] if len(time_of_crime) > 3 else \
        "0" + time_of_crime[:2] + ":" + time_of_crime[2:]
    epoch = int(time.mktime(time.strptime(date_of_crime + " " + time_of_crime, date_format)))
    row[filtered_headers.get('Epoch Occurred')] = epoch

    return row


# Convert data rows to CSV form for export
def transform_for_csv_export(row):
    num_cols = len(row)
    line = []
    for index in range(num_cols):
        line.append(row.get(index))

    output = io.StringIO("")
    csv.writer(output).writerow(line)
    return output.getvalue().strip()


# Get names of selected features for CSV export
def get_filtered_headers():
    indexed_headers = {y: x for x, y in filtered_headers.items()}
    header_list = []
    for index in range(len(indexed_headers)):
        header_list.append(indexed_headers[index])

    return ",".join(header_list)


def main():
    # Spark Config
    conf = SparkConf().setAppName("LACrimePreProcessing").setMaster('local')
    # Init SparkSession
    sc = SparkContext(conf=conf)

    # Read CSV file
    rawFileRdd = sc.textFile(raw_data_file)

    # Pre-processing
    rawFileRdd = rawFileRdd.map(filter_cols).map(date_to_epoch).map(transform_for_csv_export)

    # To export pre-processed data to CSV
    # processed_data = [get_filtered_headers()]
    processed_data = []
    processed_data.extend(rawFileRdd.collect())
    with open(processed_data_file, 'w') as f:
        for line in processed_data:
            f.write(line + '\n')

    # SparkContext Cleanup
    sc.stop()



if __name__ == '__main__':
    main()