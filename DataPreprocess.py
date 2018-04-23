import csv
import io
import time

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
