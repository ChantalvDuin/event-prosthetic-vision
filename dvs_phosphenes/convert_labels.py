"""
    This script is an adapted script of vision-kit from giacomobartoli to convert CORe50
     labels (.txt) into .pbtxt for Tensorflow Object Detection
"""

# set input as CORe50 label file
input = '/home/chadui/data/dvs_phosphenes/object_detection/core50_category_class_names.txt'

# set output as corresponding pbtxt file
output = '/home/chadui/data/dvs_phosphenes/object_detection/core50_category_labels.pbtxt'

input_file = open(input, 'r').readlines()
output_file = open(output, 'w')
counter = 0

def addApex(s):
    newString = "'" + s + "'"
    return newString

def createItem(s):
    sf='item {\n id:' + str(counter)+' \n name: ' + repr(s) + '\n}\n\n'
    return sf

for i in input_file:
    counter+=1
    output_file.writelines(createItem(i.strip()))
output_file.close()

print('done! Your .pbtxt file is ready!')