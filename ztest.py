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
import scipy as sp
c = OSC.OSCClient()
c.connect(('10.19.33.79', 57120))   # connect to SuperCollider on surface
# c.connect(('127.0.0.1', 57120))   # connect to SuperCollider IDE Local
#c.connect(('192.168.7.2', 57120)) # connect to Bela
import time


class_names = ['RR', 'guitar-knees', 'bass', 'cowbell', 'piano', 'guitar', 'goats','clap','drums','dab','stand','throatcut','bow']
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
# list of  body keypoints as coordinates
headX = 0 
headY = 1
neckX = 2
neckY = 3
rShoulderX = 4
rShoulderY = 5
rElbowX = 6
rElbowY = 7
rWristX = 8
rWristY = 9
lShoulderX = 10
lShoudlerY = 11
lElbowX = 12
lElbowY = 13
lWristX = 14
lWristY = 15
pelvisX = 16
pelvisY = 17
rHipX = 18
rHipY = 19
rKneeX = 20
rKneeY = 21
rAnkleX = 22
rAnkleY = 23
lHipX = 24
lHipY = 25
lKneeX = 26
lKneeY = 27
lAnkleX = 28
lAnkleY = 29


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
		#oscmsg.append(most_common(tempMovements))
		oscmsg.append('bang')
		c.send(oscmsg)
		#c.send('bang')
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
#returns the percentage that x is between a and b,
#assumes a is lower value and b is higher value
def returnPercentage(x,a,b):
	if (x >=a):
		return 0
	if (x <=b):
		return 1
	else:
		return (x-a)/(b-a)


def get_angle(p0, p1=np.array([0,0]), p2=None):
	''' compute angle (in degrees) for p0p1p2 corner
	Inputs:
	p0,p1,p2 - points in the form of [x,y]
	'''
	if p2 is None:
		p2 = p1 + np.array([1, 0])
	v0 = np.array(p0) - np.array(p1)
	v1 = np.array(p2) - np.array(p1)
	angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
	return np.degrees(angle)


datum = op.Datum()
while True:
	#need this to happen a bit after the program loads otherwise it throws a memory error
	if loop ==3:
		model=tf.keras.models.load_model('model_size6_aug_1.h5')
	ret,img = stream.read()
	if (img.any()):
		#print(img)
		datum.cvInputData = img
		opWrapper.emplaceAndPop([datum])

	keypoints = datum.poseKeypoints # chop off the confidence levels
	#print(keypoints.shape)
	
	if (len(list1) == 250):
		#Checking is there to make sure that we don't get weird valuess above 1 that could mess with our calculations (happens because we set 0 values to -1) 
		check = abs(np.sum(np.subtract(list1[150:178],list1[200:228])))
		if (check < 1):
			person1['motion'].pop(0)
			person1['motion'].append(check)
		#print(np.mean(person1['motion']))
		
		#if left or right foot is above knee
		#high kick
		if (list1[229] < list1[221] or list1[223] < list1[227]):
		#and ):
			print ('highkick')
				
		#delta of hand movement 
		#print("right:" + str(returnPercentage(list1[200+rWristY],list1[200+pelvisY],list1[200+headY])))
		#print("left:" + str(returnPercentage(list1[200+lWristY],list1[200+pelvisY],list1[200+headY])))

		
		
		
		#compares the Y location of the ankle, neck, and pelvis to the previous frame's Y location, and gets hoe much of a percentage increase based on the Y value for their head  
		percentageAnkleGain = returnPercentage(list1[200 + lAnkleY],list1[150+lAnkleY],list1[150+headY])
		percentageNeckGain = returnPercentage(list1[200 + neckY],list1[150+neckY],list1[150+headY])
		percentagePelvisGain = returnPercentage(list1[200 + pelvisY],list1[150+pelvisY],list1[150+headY])
		if ( percentageAnkleGain > .004 and percentageNeckGain > .1 and percentagePelvisGain >.1):
			print('REALLLLLJUMPPP')
		
		#print(percentagePelvisGain)


		#if left hand x pos crosses center line and right arm is reaching out
		#if (list1[214] < list1[216] and (abs(list1[208]-list1[204])>abs(list1[214]-list1[202]))):
		#	print('reaching right')
		#if (list1[208] > list1[216] and (abs(list1[214]-list1[210])>abs(list1[208]-list1[202]))):  
			
		#	print('reaching left')

		# arm bpm checker
		#print("time:%s x:%s  y:%s"%(time.time(),list1[208],list1[209]))
		#leftStrumLastFrame=(list1[214]-list1[208])*(list1[225]-list1[209])-(list1[215]-list1[209])*(list1[224]-list1[208])
		#leftStrumPreviousFrame=(list1[164]-list1[158])*(list1[175]-list1[159])-(list1[165]-list1[159])*(list1[174]-list1[158])
		#if (leftStrumLastFrame > 0 and leftStrumPreviousFrame < 0) or (leftStrumLastFrame < 0 and leftStrumPreviousFrame > 0):
			#print('strum bass')

		
		#pOwer strum
		currentAngle = get_angle([list1[200+rWristX],list1[200+rWristY]],[list1[200+rShoulderX],list1[200+rShoulderY]],[0,list1[200+rShoulderY]])
		previousAngle = get_angle([list1[150+rWristX],list1[150+rWristY]],[list1[150+rShoulderX],list1[150+rShoulderY]],[0,list1[150+rShoulderY]])
		print(currentAngle)
		if (currentAngle>=90 and currentAngle <=170) and previousAngle<=-90:
			print ('powerStrum')
			
		#guitar strum
		#leftStrumLastFrame=(list1[208]-list1[214])*(list1[219]-list1[215])-(list1[209]-list1[215])*(list1[218]-list1[214])
		#leftStrumPreviousFrame=(list1[158]-list1[164])*(list1[169]-list1[165])-(list1[159]-list1[165])*(list1[168]-list1[164])
		#if (leftStrumLastFrame > 0 and leftStrumPreviousFrame < 0) or (leftStrumLastFrame < 0 and leftStrumPreviousFrame > 0):
		#	print('strum guitar')


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
			print ('person 1')
			print(class_names[b[0]])
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



