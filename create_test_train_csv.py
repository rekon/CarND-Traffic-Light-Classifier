import csv
import glob
from sklearn import model_selection

def create_csvs():
	train = []
	test = []
	total = []

	for myclass, directory in enumerate(['red', 'not_red', 'not_light']):
		for filename in glob.glob('./dataset_resized/{}/*.jpg'.format(directory)):
			filename = '/'.join(filename.split('\\'))
			total.append([filename, myclass, directory])
			
	train, test = model_selection.train_test_split(total, test_size=0.1)

	with open('traffic_light_train.csv', 'w', newline='') as csvfile:
		mywriter = csv.writer(csvfile)
		mywriter.writerow(['path', 'class', 'color'])
		mywriter.writerows(train)
		print('Training CSV file created successfully')
		
	with open('traffic_light_test.csv', 'w', newline='') as csvfile:
		mywriter = csv.writer(csvfile)
		mywriter.writerow(['path', 'class', 'color'])
		mywriter.writerows(test)
		print('Testing CSV file created successfully')

		
	print('CSV files created successfully')
	
if __name__ == "__main__":
	create_csvs()