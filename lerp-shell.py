import os

for i in range(1, 505, 2):
    input1 = "/Users/andrewluck/Documents/repos/MMM/midi/" + str(i) + ".mid"
    input2 = "/Users/andrewluck/Documents/repos/MMM/midi/" + str(i + 1) + ".mid"
    folder = str(i)
    os.system("bash lerp.sh "+ input1 + " " + input2 + " " + folder)
    


