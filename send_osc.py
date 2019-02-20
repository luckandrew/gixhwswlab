"""Send OSC to Supercollider

This program sends a string to Supercollider to trigger an instrument.

"""
import argparse
import random
import time

from pythonosc import osc_message_builder
from pythonosc import udp_client

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1",
    help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=57120,
    help="The port the OSC server is listening on")
args = parser.parse_args()

client = udp_client.SimpleUDPClient(args.ip, args.port)

def send_osc(gesture):
    client.send_message('/bang', gesture)
  
#   send_osc('cowbell')
#   print("osc sent")

# Unit Test: brute force test gesture assignment
# gestures = ["drums","bass", "guitar", "cowbell", "head nod", "T-Pose", "hands up", "crouch"]
gestures = ["drums","guitar", "cowbell"]

# Unit Test: iterate through gestures array
for gesture in gestures:
    client.send_message("/hello", random.random())
    send_osc(gesture)
    # time.sleep(4)
    