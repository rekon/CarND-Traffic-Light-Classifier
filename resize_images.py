import cv2
import glob
from time import time

t=time()
print('Resizing...')
for myclass, directory in enumerate(['red', 'not_red', 'not_light']):
	print('Converting',directory,'images')
	for filename in glob.glob('./dataset_resized/{}/*.jpg'.format(directory)):
		filename = '/'.join(filename.split('\\'))
		image = cv2.imread(filename)
		if(image.shape[0] != 600):
			image = cv2.resize( image, (800,600))
			cv2.imwrite(filename, image)
print('Total time taken: ', time() - t )