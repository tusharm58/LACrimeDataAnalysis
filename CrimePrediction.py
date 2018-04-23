import csv
import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkContext, SparkConf
from sklearn import linear_model

import DataPreprocess as dp

data_file = "sample_new.csv"


# Convert input CSV line to list
def line_to_row(line):
    input_csv = io.StringIO(line)
    for row in csv.reader(input_csv):
        return row


# Plot True Values vs Predictions
def plot(true_values, predictions, title):
    true_values_plot, = plt.plot(true_values, 'C1', label='True Values')
    predictions_plot, = plt.plot(predictions, 'C2', label='Predictions')
    plt.legend(handles=[true_values_plot, predictions_plot])
    plt.title(title)
    plt.show()


# Predictions using AR Time Series Analysis
def get_AR_predictions(p, test_index, data):
    true_values = data[test_index:]
    train_data = []
    target_data = []
    test_data = []
    for end_index in range(p, len(data)):
        start_index = end_index - p
        if end_index < test_index:
            train_data.append(data[start_index:end_index])
            target_data.append(data[end_index])
        else:
            test_data.append(data[start_index:end_index])

    regr = linear_model.LinearRegression()
    regr.fit(train_data, np.array(target_data).reshape(-1, 1))

    return true_values, regr.predict(test_data)


# Map crime occurrence by week of year
def map_by_week(row):
    epoch = float(row[dp.filtered_headers['Epoch Occurred']])
    datetime_occurred = datetime.fromtimestamp(epoch)
    week = str(datetime_occurred.strftime("%U"))
    year = str(datetime_occurred.strftime("%y"))

    return year + "-" + week, 1


# Map crime occurrence by month of year
def map_by_month(row):
    epoch = float(row[dp.filtered_headers['Epoch Occurred']])
    datetime_occurred = datetime.fromtimestamp(epoch)
    month = str(datetime_occurred.strftime("%m"))
    year = str(datetime_occurred.strftime("%y"))

    return year + "-" + month, 1


# Predictions using Seasonal Time Series Analysis
def get_seasonal_predictions(season, start_index, data):
    true_values = data[start_index:]
    predictions = []
    for end_index in range(start_index, len(data)):
        prediction = data[end_index - season]
        predictions.append(prediction)

    return true_values, predictions


if __name__ == '__main__':
    # Spark Config
    conf = SparkConf().setAppName("LACrimeTimeSeries").setMaster('local')
    # Init SparkSession
    sc = SparkContext(conf=conf)

    # Read CSV file
    fileRdd = sc.textFile(data_file)
    fileRdd = fileRdd.map(line_to_row)

    # Monthly time series analysis
    month_agg_rdd = fileRdd.map(map_by_month).reduceByKey(lambda x, y: x + y).sortByKey()
    month_agg_data = month_agg_rdd.values().collect()
    # Monthly AR Prediction
    true_values, predictions = get_AR_predictions(12, 72, month_agg_data)
    plot(true_values, predictions, "MonthlyAR12")
    # Monthly Seasonal Prediction
    true_values, predictions = get_seasonal_predictions(12, 72, month_agg_data)
    plot(true_values, predictions, "MonthlySeasonal")

    # Weekly time series analysis
    week_agg_rdd = fileRdd.map(map_by_week).reduceByKey(lambda x, y: x + y).sortByKey()
    week_agg_data = week_agg_rdd.values().collect()
    # Weekly AR Prediction
    true_values, predictions = get_AR_predictions(3, 250, week_agg_data)
    plot(true_values, predictions, "WeeklyAR3")
    # Weekly Seasonal Prediction
    true_values, predictions = get_seasonal_predictions(53, 250, week_agg_data)
    plot(true_values, predictions, "WeeklySeasonal")

    sc.stop()
