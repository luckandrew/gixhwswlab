# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import tensorflow as tf
from multiprocessing import Process,Queue,Pipe
import OSC
c = OSC.OSCClient()
# c.connect(('10.19.31.129', 57120))   # connect to SuperCollider
# c.connect(('127.0.0.1', 57120))   # connect to SuperCollider IDE Local
c.connect(('192.168.7.2', 57120)) # connect to Bela
import time


class_names = ['cowbell', 'crouch', 'drums', 'guitar', 'handsup', 'headnod', 'tpose']
#with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
		#model = load_model('imdb_mlp_model.h5')
#


# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
	# Windows Import
	if platform == "win32":
		# Change these variables to point to the correct folder (Release/x64 etc.) 
		sys.path.append(dir_path + '/../../python/openpose/Release');
		os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
		import pyopenpose as op
	else:
		# Change these variables to point to the correct folder (Release/x64 etc.) 
		sys.path.append('../../python');
		# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
		# sys.path.append('/usr/local/python')
		from openpose import pyopenpose as op
except ImportError as e:
	print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
	raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params['net_resolution'] = '128x96'
params['camera_resolution'] = '640x480'
params['camera'] = '1'
params['render_pose'] = 0

# Add others in path?
for i in range(0, len(args[1])):
	curr_item = args[1][i]
	if i != len(args[1])-1: next_item = args[1][i+1]
	else: next_item = "1"
	if "--" in curr_item and "--" in next_item:
		key = curr_item.replace('-','')
		if key not in params:  params[key] = "1"
	elif "--" in curr_item and "--" not in next_item:
		key = curr_item.replace('-','')
		if key not in params: params[key] = next_item


stream = cv2.VideoCapture(1)
if (not stream.isOpened()):  # check if succeeded to connect to the camera
   print("Cam open failed");
else:
	stream.set(3,640);
	stream.set(4,480);

loop = 0
windowSize = 6 # just for reference
#list of the coords of the different people 
list1 = []
list2 = []
list3 = []
person1 = {'motion':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
#list21 = []
#list22 = []
tempMovements = []


# Starting OpenPose
opWrapper = op.WrapperPython()
print(params)
opWrapper.configure(params)
opWrapper.start()
oldtime = time.time()
def most_common(lst):
	return max(set(lst), key=lst.count)

#function to send data
def packageAndSend(feature):
	global oldtime
	global tempMovements
	if (time.time() - oldtime > 3):
		tempMovements.append(feature)
		print('*************SENT TO MUSIC SERVER *****************')
		oscmsg = OSC.OSCMessage()
		oscmsg.setAddress("/bang")
		oscmsg.append(most_common(tempMovements))
		c.send(oscmsg)
		oldtime = time.time()
		tempMovements = []
	else:
		tempMovements.append(feature)

def returnNormalizedKeypoints(keypoints):
	temp = []
	for i in range(len(keypoints)):
		if (keypoints[i,0] != 0):
			temp.append(keypoints[i,0]/640) # x value normalized (640 pixels wide)
		else:
			temp.append(-1)
		if (keypoints[i,1] != 0):
			temp.append(keypoints[i,1]/480) # y value normalized (480 pixels high)
		else:
			temp.append(-1)
	#returned the normalized keypoint
	return temp

#def returnLevelOfRandomness(keypointHistory):
	
datum = op.Datum()
while True:
	#need this to happen a bit after the program loads otherwise it throws a memory error
	if loop ==3:
		model=tf.keras.models.load_model('model_size6.h5')
	ret,img = stream.read()
	if (img.any()):
		#print(img)
		datum.cvInputData = img
		opWrapper.emplaceAndPop([datum])

	keypoints = datum.poseKeypoints # chop off the confidence levels
	#print(keypoints.shape)
	
	if (len(list1) == 250):
		person1['motion'].pop(0)
		check = abs(np.sum(np.subtract(list1[150:178],list1[200:228])))
		if (check < 1):
			person1['motion'].append(check)
		print(np.mean(person1['motion']))


	if (keypoints.shape !=()):	
		list1 = list1 + returnNormalizedKeypoints(keypoints[0])
		#two people		
		if (keypoints.shape[0]>1):
			list2 = list2 + returnNormalizedKeypoints(keypoints[1])
		#three people
		if (keypoints.shape[0]>2):
			list3 = list3 + returnNormalizedKeypoints(keypoints[2])
	   	#checking to see if we have enough data on person 1 to make a prediction
		if len(list1)==300: #300 is because 6 seconds of data, 25 (x,y) coords per second...aka  windowSize*50
			b = model.predict_classes(np.expand_dims(list1, axis=0))
			#print ('person 1')
			#print(class_names[b[0]])
			packageAndSend(class_names[b[0]])
			#remove last two frames (50 data points each)
			list1 = list1[100:]

		if len(list2)==300:
			b = model.predict_classes(np.expand_dims(list2, axis=0))
			#print('             Person 2')
			#print("             " + class_names[b[0]])
			packageAndSend(class_names[b[0]])
			#remove last two frames (50 data points each)
			list2 = list2[100:]

		if len(list3)==300:
			b = model.predict_classes(np.expand_dims(list3, axis=0))
			#print('                            Person 3')
			#print('                                    ' + class_names[b[0]])
			packageAndSend(class_names[b[0]])
			#remove last two frames (50 data points each)
			list3 = list3[100:]


		
	loop+=1
	#if loop == 100:
	#	break



