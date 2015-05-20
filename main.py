import sys;
import numpy as np;

def read(key_file, train_file, test_file, weather_file):
	store_station_map 	   		  = dict();
	train_X           	   		  = [];
	train_Y     	       		  = [];
	test_X				   		  = [];
	station_dt_weatherDetails_map = dict();

	with open(key_file, "r") as f:
		header = f.readline();

		for line in f:
			 data = line.strip().split(',');
			 if int(data[0]) in store_station_map:
			 	store_station_map[int(data[0])].append(int(data[1]));
			 else :
			 	store_station_map[int(data[0])] = int(data[1])

	with open(train_file, "r") as f:
		header = f.readline();

		for line in f:
			 data = line.strip().split(',');
			 train_X.append([data[0], int(data[1]), int(data[2])]);
			 train_Y.append(float(data[3]))

	with open(test_file, "r") as f:
		header = f.readline();

		for line in f:
			 data = line.strip().split(',');
			 test_X.append([data[0], int(data[1]), int(data[2])]);

	with open(weather_file, "r") as f:
		header = f.readline();

		for line in f:
			 data = line.strip().split(',');
			 station_dt_weatherDetails_map[(data[0], data[1])] = [x  for x in data[2:]] 

	return store_station_map, train_X, train_Y, test_X, station_dt_weatherDetails_map;


	

if __name__ == '__main__':
	read("./data/key.csv", "./data/train.csv", "./data/test.csv", "./data/weather.csv")	

