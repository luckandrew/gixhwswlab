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

# Construct it from system arguments
# op.init_argv(args[1])
#oppython = op.OpenposePython()
stream = cv2.VideoCapture(1)
if (not stream.isOpened()):  # check if succeeded to connect to the camera
   print("Cam open failed");
else:
    stream.set(3,640);
    stream.set(4,480);

loop = 0
dataToSave = []
windowSize = 4
frameCountOne = 0
frameCountTwo = -2

windowCount = 0
list1 = []
list2 = []

# Starting OpenPose
opWrapper = op.WrapperPython()
print(params)
opWrapper.configure(params)
opWrapper.start()


# Process and display images
#for imagePath in imagePaths:

datum = op.Datum()

while True:
	if loop ==3:
	    model=tf.keras.models.load_model('model_size4.h5')
	ret,img = stream.read()
	if (img.any()):
	    #print(img)
	    datum.cvInputData = img
	    opWrapper.emplaceAndPop([datum])
	#opWrapper.waitAndPop([datum])

	temp = []
	count = 0;
	keypoints = datum.poseKeypoints # chop off the confidence levels
	#print(keypoints.shape)

	if (keypoints.shape !=()):
	    for i in range(len(keypoints[0])):
		if (keypoints[0,i,0] != 0):
		    temp.append(keypoints[0,i,0]/640) # x value normalized (640 pixels wide)
		else:
		    temp.append(-1)
		if (keypoints[0,i,0] != 0):
		    temp.append(keypoints[0,i,1]/480) # y value normalized (480 pixels high)
		else:
		    temp.append(-1)
		count += 1
	    #add the normalized keypoints to our windows
	    list1 = list1 + temp
	    if frameCountTwo >= 0:
		list2 = list2 + temp

	    frameCountOne += 1
	    frameCountTwo += 1
	   

	    if frameCountOne >= windowSize:
		b = model.predict_classes(np.expand_dims(list1, axis=0))
		print(class_names[b[0]])
		windowCount += 1
		list1 = []
		frameCountOne = 0

	    if frameCountTwo >= windowSize:
		b = model.predict_classes(np.expand_dims(list2, axis=0))
		print(class_names[b[0]])
		windowCount += 1
		list2 = []
		frameCountTwo = 0

	    # print(datum.poseKeypoints.shape)
	    #keypoints = np.delete(datum.poseKeypoints,2,2)
	    #print(keypoints)
	    #print(keypoints[0,0,0])# x value for first entry
	    #print(keypoints[0,0,1])# y value for first entry
	    #print(keypoints.shape)
	    #dataToSave.append(keypoints) # chop off the confidence values for now
	loop+=1
	#if loop == 100:
	#	break

