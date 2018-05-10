import csv
import io
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkContext, SparkConf

import DataPreprocess as dp

data_file = "sample_new.csv"
sample_year = 2012
days = [31, 54, 62, 95, 121, 152, 200, 240, 265, 350]
sample_size = 5000


# Convert input CSV line to list
def line_to_row(line):
    input_csv = io.StringIO(line)
    for row in csv.reader(input_csv):
        return row


# Sample and map hourly crime incidents by day of the year
def sample_hourly_by_day_and_year(row):
    epoch = float(row[dp.filtered_headers['Epoch Occurred']])
    datetime_occurred = datetime.fromtimestamp(epoch)
    hour = int(datetime_occurred.strftime("%H"))
    year = int(datetime_occurred.strftime("%Y"))
    day = int(datetime_occurred.strftime("%j"))
    if day in days and year == sample_year:
        return (day, hour), 1
    return (1000, 1), 1


# Sample and map hourly crime incidents by year
def sample_hourly_by_day(row):
    epoch = float(row[dp.filtered_headers['Epoch Occurred']])
    datetime_occurred = datetime.fromtimestamp(epoch)
    hour = int(datetime_occurred.strftime("%H"))
    year = int(datetime_occurred.strftime("%Y"))
    if year == sample_year:
        return hour, 1
    return 1000, 1


# Helper function to plot
def plot(values, title):
    plots = []
    for index in range(len(values)):
        value = values[index]
        plot, = plt.plot(value, 'C' + str(index), label=str(days[index]))
        plots.append(plot)
    plt.legend(handles=plots)
    plt.title(title)
    plt.show()


# Helper function to plot CDF with CI
def plot_cdf_with_ci(cdfs):
    ci_neg_plot, = plt.plot(cdfs[0], 'C1', label='CI-')
    cdf_plot, = plt.plot(cdfs[1], 'C2', label='CDF')
    ci_pos_plot, = plt.plot(cdfs[2], 'C1', label='CI+')
    plt.legend(handles=[ci_neg_plot, cdf_plot, ci_pos_plot])
    plt.title("CDF of Daily Crimes with DKW CI")
    plt.show()


if __name__ == '__main__':
    # Spark Config
    conf = SparkConf().setAppName("LACrimeHourlyAnalysis").setMaster('local')
    # Init SparkSession
    sc = SparkContext(conf=conf)

    # Read CSV file
    fileRdd = sc.textFile(data_file)
    fileRdd = fileRdd.map(line_to_row)

    # Count crime incidents by hour of the day
    hourly_agg_rdd = fileRdd.map(sample_hourly_by_day_and_year).reduceByKey(lambda x, y: x + y).sortByKey()
    hourly_agg_data = hourly_agg_rdd.collect()

    # Group crime incidents by hour of day for trends
    freq_map = {}
    for (day, hour), freq in hourly_agg_data:
        freqs = [] if freq_map.get(day) is None else freq_map.get(day)
        freqs.append(freq)
        freq_map[day] = freqs

    # Plot hourly trends
    freq_list = []
    for day in days:
        freq_list.append(freq_map[day])
    plot(freq_list, "Hourly Trend Plots")

    # Calculate PDF and CDF (Sampled by days of a year)
    pdfs = []
    cdfs = []
    for day_freqs in freq_list:
        cdf = []
        pdf = []
        total_crime = np.sum(day_freqs)
        curr_crime_total = 0
        for freq in day_freqs:
            curr_crime_total += freq
            cdf.append(curr_crime_total / total_crime)
            pdf.append(freq / total_crime)
        cdfs.append(cdf)
        pdfs.append(pdf)
    # Plot PDFs
    plot(pdfs, "Hourly PDF Plots")
    # Plot CDFs
    plot(cdfs, "Hourly CDF Plots")

    # Count crime incidents by hour of the day
    hourly_agg_rdd = fileRdd.map(sample_hourly_by_day).filter(lambda x: x[0] != 1000)
    hourly_agg_rdd = sc.parallelize(hourly_agg_rdd.takeSample(False, sample_size, seed=0))
    hourly_agg_rdd = hourly_agg_rdd.reduceByKey(lambda x, y: x + y).sortByKey()
    hourly_agg_data = hourly_agg_rdd.values().collect()

    # Sample and calculate CDFs with DKW CI
    cdfs = []
    total_crime = np.sum(hourly_agg_data)
    curr_crime_total = 0
    # Calculate Epsilon
    epsilon = math.sqrt(math.log(2 / 0.05) / (2 * sample_size))
    ci_pos = []
    ci_neg = []
    cdf = []
    for freq in hourly_agg_data:
        curr_crime_total += freq
        cdf.append(curr_crime_total / total_crime)
        ci_pos.append(curr_crime_total / total_crime + epsilon)
        ci_neg.append(curr_crime_total / total_crime - epsilon)
    cdfs.append(ci_neg)
    cdfs.append(cdf)
    cdfs.append(ci_pos)
    # Plot CDFs with CI
    plot_cdf_with_ci(cdfs)

    # SparkContext Cleanup
    sc.stop()
