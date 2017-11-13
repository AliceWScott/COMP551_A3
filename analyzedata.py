import csv
import codecs
import numpy as np 

def toList(filename):
	data = csv.reader(codecs.open(filename, 'r'))
	data_list = []
	for row in data:
		if row[1]: data_list.append(row[1])
		else: data_list.append("")
	del data_list[0]
	return data_list

def countOccurances(data):

	counts = {}
	for x in data:
		if x in counts:
			counts[x] += 1
		else:
			counts[x] = 1
	print counts


countOccurances(toList('./predictions.csv'))
countOccurances(toList('./predictions2.csv'))
		