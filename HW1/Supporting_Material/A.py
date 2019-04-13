import pandas as pd 
import csv

data = pd.read_csv('data.csv')

print(data.sample(1000))

# csvfile = open('data.csv')
# csv_reader = csv.reader(csvfile, delimiter=',')

# for row in csv_reader:
#     print(row)