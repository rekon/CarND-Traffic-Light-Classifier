import os
import numpy as np
import pandas as pd
import random
import cv2
import csv
import glob

if not os.path.exists('./sim_traffic_light_train.csv'):
	with open('sim_traffic_light_train.csv', 'w', newline='') as csvfile:
		mywriter = csv.writer(csvfile)
		mywriter.writerow(['path', 'class', 'color'])

		for myclass, directory in enumerate(['red', 'green','yellow','not_light']):
			for filename in glob.glob('./my_sim_screenshots/{}/*.jpg'.format(directory)):
				filename = '/'.join(filename.split('\\'))
				mywriter.writerow([filename, myclass, directory])
	print('CSV file created successfully')
else:
	print('CSV already present')