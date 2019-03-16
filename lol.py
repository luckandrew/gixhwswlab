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

whatStateAreWeIn = {'mode':'begin'} #play, end
loop = 0
windowSize = 6 # just for reference
#list of the coords of the different people 
list1 = []
list2 = []
list3 = []
motion = {1:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 2:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 3:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
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
frame6 = 250
frame5 = 200
frame4 = 150
frame3 = 100
frame2 = 50
frame1 = 0


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

#sends an OSC message depending on the location and history of the pose. 
def checkDrumsDSP(hand,history,currentFrame,previousFrame):
    yMax = max(history[currentFrame + lShoulderY],history[currentFrame + rShoulderY])
    yMin = min(history[currentFrame + lHipY],history[currentFrame + rHipY])
    xMax = max(history[currentFrame + lShoulderX],history[currentFrame + lHipX])
    xMin = max(history[currentFrame + rShoulderX],history[currentFrame + rHipX])

    #here we check to see if their wrist is out of the safe zone(torso). 
    #if it is out of the box, we send a drum hit depending on the location
    #start with left hand 
    if (history[currentFrame + lWristY] < history[currentFrame + yMax] or history[currentFrame + lWristY] > history[currentFrame + yMin]):
        if (history[currentFrame + lWristY] < history[currentFrame + yMax]):
            if (history[currentFrame + lWristX] < xMin):
                print('left hand outside drum box quadrant 1')
            elif(history[currentFrame + lWristX] > xMax):
                print('left hand outside drum box quadrant 3')
            else:
                print('left hand outside drum box quadrant 2')
        else: #left hand is below waist
            if (history[currentFrame + lWristX] < xMin):
                print('left hand outside drum box quadrant 4')
            elif(history[currentFrame + lWristX] > xMax):
                print('left hand outside drum box quadrant 6')
            else:
                print('left hand outside drum box quadrant 5')

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

def checkForPose(history, person):
    history = np.reshape(history, (-1, 50))
    a = np.expand_dims(history, axis=0)
    print(a.shape)
    #c = np.reshape(a, (0,6,50))
   # print(c.shape)
    b = model.predict(a)
    blist = []
    for i in b[0]:
    blist.append(i)
    #print(class_names[blist.index(max(b[0]))])
    print ('person ' + str(person) +':' +  class_names[blist.index(max(b[0]))] + " confidence: " + str(max(b[0])))
    #packageAndSend(class_names[b[0]])
    #print(type(b))    
    
#def checkDrums():

def checkForDSP(history,person,currentFrame,previousFrame):
    #highkick
    if (history[currentFrame + lAnkleY] < history[currentFrame + rKneeY] or history[currentFrame + rAnkleY] < history[currentFrame + lKneeY]):
        print ('highkick')
    #allow guitar to fret
    #if (person == 1):
        #delta of hand movement 
        #print("right fret level:" + str(returnPercentage(history[currentFrame+rWristY],history[currentFrame+pelvisY],history[currentFrame+headY])))
        #print("left fret level:" + str(returnPercentage(history[currentFrame+lWristY],history[currentFrame+pelvisY],history[currentFrame+headY])))
    
    #find the jump 
    percentageAnkleGain = returnPercentage(history[currentFrame+ lAnkleY],history[previousFrame+lAnkleY],history[previousFrame+headY])
    percentageNeckGain = returnPercentage(history[currentFrame + neckY],history[previousFrame+neckY],history[previousFrame+headY])
    percentagePelvisGain = returnPercentage(history[currentFrame + pelvisY],history[previousFrame+pelvisY],history[previousFrame+headY])
    if ( percentageAnkleGain > .004 and percentageNeckGain > .1 and percentagePelvisGain >.1):
        print('REALLLLLJUMPPP')

    if (person == 1):
        strums = []
        wrist1 = [lWristX,lWristY]
        wrist2 = [rWristX,rWristY]
        hip = [lHipX,lHipY]
        shoulder = [rShoulderX,rShoulderY]
        for i in (currentFrame,previousFrame):
            x = history[i  + wrist1[0]]
            y = history[i  + wrist1[1]]
            x1 = history[i + wrist2[0]]
            y1 = history[i + wrist2[1]]
            x2 = history[i + hip[0]]
            y2 = history[i + hip[0]]
            strums.append((x-x1)*(y2-y1)-(y-y1)*(x2-x1)) #which side of the line are they on?
        
        if (strums[0] > 0 and strums[1] < 0) or (strums[0] < 0 and strums[1] > 0):
            print('strum bass')
        
    currentAngle = get_angle([history[currentFrame+wrist1[0]],history[currentFrame+wrist1[1]]],[history[currentFrame+shoulder[0]],history[currentFrame+shoulder[1]]],[0,history[currentFrame+shoulder[1]]])
    previousAngle = get_angle([history[previousFrame+wrist1[0]],history[previousFrame+wrist1[1]]],[history[previousFrame+shoulder[0]],history[previousFrame+shoulder[1]]],[0,history[previousFrame+shoulder[1]]])
    previousFrame2 = previousFrame-50
    previousAngle2 = get_angle([history[previousFrame2+wrist1[0]],history[previousFrame2+wrist1[1]]],[history[previousFrame2+shoulder[0]],history[previousFrame2+shoulder[1]]],[0,history[previousFrame2+shoulder[1]]])
        #print(currentAngle)
        if (currentAngle>=90 and currentAngle <=170) and (previousAngle<=-90 or previousAngle2 <=-90):
            print ('powerStrum bass ')

    #pOwer strum
    #currentAngle = get_angle([history[currentFrame+rWristX],history[currentFrame+rWristY]],[history[currentFrame+rShoulderX],history[currentFrame+rShoulderY]],[0,history[currentFrame+rShoulderY]])
    #previousAngle = get_angle([history[previousFrame+rWristX],history[previousFrame+rWristY]],[history[previousFrame+rShoulderX],history[previousFrame+rShoulderY]],[0,history[previousFrame+rShoulderY]])
    #print(currentAngle)
    #if (currentAngle>=90 and currentAngle <=170) and previousAngle<=-90:
    #    print ('powerStrum')


    #TODO return 0,1,2
    #update randomness of motion if we got all the major keypoints
    if not (-1 in history[previousFrame:previousFrame+lAnkleX] or -1 in history[currentFrame:currentFrame+lAnkleX]):
        check = np.sum(abs(np.subtract(history[previousFrame:previousFrame+lAnkleX],history[currentFrame:currentFrame+lAnkleX])))
        motion[person].pop(0)
        motion[person].append(check)
    #print("motion: " + str(np.mean(motion[person])))

    #drumbox
    #drum box, limbs outside out this will trigger a drum hit
    #topY, leftX, rightX, bottomY
    if (person==3):#drummer
        yMax = max(history[currentFrame + lShoulderY],history[currentFrame + rShoulderY])
        yMin = min(history[currentFrame + lHipY],history[currentFrame + rHipY])
        xMax = max(history[currentFrame + lShoulderX],history[currentFrame + lHipX])
        xMin = max(history[currentFrame + rShoulderX],history[currentFrame + rHipX])

        #here we check to see if their wrist is out of the safe zone(torso). 
        #if it is out of the box, we send a drum hit depending on the location
        #start with left hand 
        if (history[currentFrame + lWristY] < history[currentFrame + yMax] or history[currentFrame + lWristY] > history[currentFrame + yMin]):
            if (history[currentFrame + lWristY] < history[currentFrame + yMax]):
                if (history[currentFrame + lWristX] < xMin):
                    print('left hand outside drum box quadrant 1')
                elif(history[currentFrame + lWristX] > xMax):
                    print('left hand outside drum box quadrant 3')
                else:
                    print('left hand outside drum box quadrant 2')
            else: #left hand is below waist
                if (history[currentFrame + lWristX] < xMin):
                    print('left hand outside drum box quadrant 4')
                elif(history[currentFrame + lWristX] > xMax):
                    print('left hand outside drum box quadrant 6')
                else:
                    print('left hand outside drum box quadrant 5')
        #right hand dums

        



#begin while loop!
datum = op.Datum()
while True:
    #need this to happen a bit after the program loads otherwise it throws a memory error
    if loop ==3:
        model=tf.keras.models.load_model('model_size6_aug_2.h5')
    ret,img = stream.read()
    if (img.any()):
        #print(img)
        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])

    keypoints = datum.poseKeypoints # chop off the confidence levels


    if (keypoints.shape !=()):  
        list1 = list1 + returnNormalizedKeypoints(keypoints[0])
        #two people     
        if (keypoints.shape[0]>1):
            list2 = list2 + returnNormalizedKeypoints(keypoints[1])
        #three people
        if (keypoints.shape[0]>2):
            list3 = list3 + returnNormalizedKeypoints(keypoints[2])

        #checking to see if we have enough data on person 1 to make a prediction
        if len(list1)==250: 
            checkForDSP(list1,1,frame5,frame4)
        elif len(list1) ==300: #300 is because 6 seconds of data, 25 (x,y) coords per second...aka  windowSize*50
            checkForDSP(list1,1,frame6,frame5)
            checkForPose(list1,1)
            list1 = list1[100:]


        if len(list2)==250:
            checkForDSP(list2,2,frame5,frame4)
        elif len(list2) ==300:
            checkForDSP(list2,2,frame6,frame5)
            checkForPose(list2,2)
            list2 = list2[100:]

        if len(list3)==250:
            checkForDSP(list3,3,frame5,frame4)
        elif len(list3) ==300:
            checkForDSP(list3,3,frame6,frame5)
            checkForPose(list3,3)
            list3 = list3[100:]

    loop+=1
    #if loop == 100:
    #   break


