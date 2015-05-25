import sys;
import numpy as np;
from datetime import datetime, date;
import model
import warnings;

def get_season(dateObj):
	#Order - Winter, Spring,Summer, Autumn, Winter
	seasons = [(0, (date(1,  1,  1),  date(1,  3, 20))),\
				(1, (date(1,  3, 21),  date(1,  6, 20))),\
				(2, (date(1,  6, 21),  date(1,  9, 22))),\
				(3, (date(1,  9, 23),  date(1, 12, 20))),\
				(0, (date(1, 12, 21),  date(1, 12, 31)))];

	try : 
		dateObj = dateObj.date().replace(year=1);
		for season, (start, end) in seasons:
			if start <= dateObj <= end:
				return season;
	except :
		return 0;


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def getEventType(codesums, precipitation, snowfall):
	if ("SN" in codesums or "SG" in codesums) and isfloat(snowfall) and float(snowfall) > 2.0:
	   return 1; #Snow Event
	elif ("RA" in codesums or "SN" not in codesums) and isfloat(precipitation) and float(precipitation) > 1.0:
	   return 2; #Rain Event
	else :
	   return 0; #No Event



def read(key_file, train_file, test_file, weather_file):
	store_station_map 	   		  = dict();
	station_store_map 	   		  = dict();
	train_data            		  = [];
	test_data			  		  = [];
	station_dt_weatherDetails_map = dict();

	with open(key_file, "r") as f:
		header = f.readline();

		for line in f:
			 data = line.strip().split(',');
			 store_station_map[int(data[0])] = data[1];

			 if data[1] not in station_store_map:
			 	station_store_map[data[1]] 	 = [int(data[0])];
			 else:
			 	station_store_map[data[1]].append(int(data[0]));


	with open(train_file, "r") as f:
		header = f.readline();

		for line in f:
			 data = line.strip().split(',');
			 train_data.append([data[0], int(data[1]), int(data[2]), float(data[3])]);
			 

	with open(test_file, "r") as f:
		header = f.readline();

		for line in f:
			 data = line.strip().split(',');
			 test_data.append([data[0], int(data[1]), int(data[2])]);

	with open(weather_file, "r") as f:
		header = f.readline();

		for line in f:
			 data = line.strip().split(',');
			 station_dt_weatherDetails_map[(data[0], data[1])] = [x.replace('"','')  for x in data[2:]] 

	return store_station_map, station_store_map, train_data, test_data, station_dt_weatherDetails_map;



def generateFeatures(store_station_map, station_store_map, data, station_dt_weatherDetails_map, train=True):
	data_X 		 = [];
	data_Y 		 = [];
	test_ids 	 = [];

	unique_code_sum = ['', 'HZ', 'FU', 'BLSN', 'TSSN', 'VCTS', 'DZ', 'BR', 'FG', 'BCFG', 'DU', 'FZRA', 'TS', 'RA', 'PL', 'GS', 'GR', 'FZDZ', 'VCFG', 'PRFG', 'FG+', 'TSRA', 'FZFG', 'BLDU', 'MIFG', 'SQ', 'UP', 'SN', 'SG']

	#Train
	for td in data:
		X = [];

		#Get the station
		station = store_station_map[td[1]];
		#Get the correspoding weather for this station on this date
		weather = station_dt_weatherDetails_map[(station, td[0])];


		for i in range(0, len(weather)):
			if i in [0,1,2,3,4,5,6,7,8,9,13,14,15,16,17]:
				temp = weather[i].strip();
				if temp in ["M", "-"]:
					X.append("NaN");
				else:
					X.append(float(temp)); 


		for i in [11,12]:
			temp = weather[i].strip();
				
			if temp in ["M", "-"]:
				X.append("NaN");
				X.append("NaN");				
			elif temp in ["T"]: 
				#Create a new feature
				X.append(0.0);
				X.append(1); #Categorical varialble to mark a trace
			else:	
				X.append(float(temp)); 
				X.append(0); #Categorical varialble to mark absence of a trace



		#Codesum
		codesums   = weather[10].strip().split(" ");
		codesumInd = [0] * (len(unique_code_sum) + 1);

		for cs in codesums:
			if cs in unique_code_sum :
				codesumInd[unique_code_sum.index(cs)] = 1
			else:
				#unknown
				codesumInd[len(unique_code_sum)] = 1		

		#Add the no of codesums as extrac features
		codesumInd.append(len(codesums));
		X.extend(codesumInd)


		
		##########################################################################
		#How many stores which share the same weather station
		X.append(len(station_store_map[station]));

		#Add the store no, item no as categorical features
		X.append(td[1]);
		X.append(td[2]);

		#Add weather type
		X.append(getEventType(weather[10].strip().split(" "), weather[12], weather[11]));

		#Make date object
		dt = datetime.strptime(td[0],'%Y-%m-%d')
		#Extract date based features
		X.append(dt.month);             	 #Month
		X.append(dt.strftime("%W"));	 	 #WeekNo of the year
		X.append(((dt.day - 1) // 7 + 1));	 #Week of the month
		X.append(dt.weekday());			 #Week day 
		X.append(get_season(dt)); 			 #Season 	

		if dt.weekday() in [5,6]:
			X.append(1);			 		 #Week end
		else:
			X.append(0);

		##########################################################################
		data_X.append(X);
			
		if train:
			data_Y.append(td[3]);
		else:
			test_ids.append( ( str(td[1]) + "_" + str(td[2]) + "_" + str(td[0]) ) );

	if train:
		return data_X, data_Y;
	else:
		return data_X, test_ids;		


if __name__ == '__main__':
	warnings.filterwarnings("ignore");
	store_station_map, station_store_map, train_data, test_data, station_dt_weatherDetails_map = read("./data/key.csv", "./data/train_subset.csv", "./data/test.csv", "./data/weather.csv")	
	print("Files Read")
	train_X, train_Y = generateFeatures(store_station_map, station_store_map, train_data, station_dt_weatherDetails_map, True);
	print("Training Data Loaded")
	test_X, test_ids = generateFeatures(store_station_map, station_store_map, test_data, station_dt_weatherDetails_map, False);
	print("Test Data Loaded")

	del store_station_map, station_store_map, train_data, test_data, station_dt_weatherDetails_map;

	model.do(np.array(train_X), np.array(train_Y), np.array(test_X), test_ids);





