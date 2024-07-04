import csv
import math
import numpy as np
from Frames import image
#import random
class nucleus():
    def __init__(self,inputs):
        self.inputs = inputs
        self.into = []
        #self.function = opt ##Look into dictionaries more, for now it's just one function
        for i in range(inputs):
            self.into.append(0)
        self.value = 0
        self.output = 0
        self.end = 0
        self.train = 0
    def func(self,val):
        if (self.end == 0):
            self.value = (1/(1+math.exp(-val))) - 0.5 ##Why 0.5
        else:
            if (val != 0):
                self.value = val/abs(val)
            else:
                self.value = 0
    def pass_out(self):
        if (self.value != 0):
            self.output = self.value
            if (self.train == 0):
                self.value = 0
            elif (self.train > 0):
                self.value = self.value
    def pass_in(self):
        if (type(self.into) == float):
            self.func(self.into)
            self.into = 0
        else:
            self.func(sum(self.into))
            for i in self.into:
                i = 0

    
class wire_in():
    def __init__(self,output):
        self.output = output
        self.value = 0
        if (type(self.output) != nucleus):
            raise TypeError("Must have nuclei as outputs")
    def update(self):
        self.output.into = self.value

class wire_out():
    def __init__(self,input):
        self.input = input
        self.value = 0
        if (type(self.input) != nucleus):
            raise TypeError("Must have nuclei as inputs")
    def update(self):
        self.value = self.input.output



class wire():
    def __init__(self,input,output,num):
        self.input = input
        self.output = output
        self.value = 0
        self.into = num
        self.weight = 1 ###This needs to be changed to change the value 
        if (type(self.input) != nucleus or type(self.output) != nucleus):
            raise TypeError("Must have nuclei as inputs and outputs")
    def update(self):
        self.value = self.input.output * self.weight
        self.output.into[self.into] = self.value ##This needs to be changed, not append
    def update_w(self,weigh):
        self.weight = weigh


class net():
    def __init__(self,input,hidden,width,output):
        if hidden < 3:
            raise ValueError("Must be at least 3 hidden layers")
        self.input = input
        self.hidden = hidden
        self.output = output
        self.width = width
        self.train_data = []
        ###Set Up Inputs###
        self.inputs = []
        for i in range(self.input):
            self.inputs.append(0)
        for i in range(self.input):
            self.inputs[i] = nucleus(1)
        ###################
        self.inwires = []
        for i in range(self.input):
            self.inwires.append(wire_in(self.inputs[i]))
        ###Set Up Hidden Layers###
        self.middle_1 = []
        for i in range(self.width):
            self.middle_1.append(nucleus(self.input))
        self.middle_wire_in = []
        for i in range(self.input):
            for j in range(self.width):
                self.middle_wire_in.append(wire(self.inputs[i],self.middle_1[j],i))
        self.middle_n = []
        for i in range(self.width):
            self.middle_n.append(nucleus(self.width))
        self.middle_wires = []
        if self.hidden == 2:
            for i in range(self.width):
                for j in range(self.width):
                    self.middle_wires.append(wire(self.middle_1[i],self.middle_n[j],i))
        if self.hidden > 2:
            self.middle = []
            for i in range(self.hidden - 2):
                self.middle.append(0)
            for i in range(self.hidden - 2):
                self.middle[i] = []
                for j in range(self.width):
                    self.middle[i].append(0)
                for j in range(self.width):
                    self.middle[i][j] = nucleus(self.width)
            for i in range(self.hidden-1):
                self.middle_wires.append([])
            for i in range(self.hidden-1):
                for j in range(self.width):
                    for k in range(self.width):
                        if (i==0):
                            self.middle_wires[i].append(wire(self.middle_1[j],self.middle[i][k],j))
                        elif (i==self.hidden-2):
                            self.middle_wires[i].append(wire(self.middle[i-2][j],self.middle_n[k],j))
                        else:
                            self.middle_wires[i].append(wire(self.middle[i-1][j],self.middle[i][k],j))
        ########################## 
        ###Set Up Output###
        self.outputs = []
        for i in range(self.output):
            self.outputs.append(nucleus(self.width))
        #for i in self.outputs:
            #i.end = 1
        self.middle_wire_out = []
        for i in range(self.width):
            for j in range(self.output):
                self.middle_wire_out.append(wire(self.middle_n[i],self.outputs[j],i))
        self.wires_out = []
        for i in range(self.output):
            self.wires_out.append(wire_out(self.outputs[i]))
        ###################
    def in_net(self):
        for i in self.inwires:
            i.update()
        for i in self.inputs:
            i.train = 1
            i.pass_in()
            i.pass_out()
        for i in self.middle_wire_in:
            i.update()
    def out_net(self):
        for i in self.middle_wire_out:
            i.update()
        for i in self.outputs:
            i.train = 1
            i.pass_in()
            i.pass_out()
        for i in self.wires_out:
            i.update()
    def mid_net(self):
        for i in self.middle_1:
            i.train = 1
            i.pass_in()
            i.pass_out()
        for i in range(self.hidden-2):
            for j in self.middle_wires[i]:
                j.update()
            for j in self.middle[i]:
                j.train = 1
                j.pass_in()
                j.pass_out()
        for j in self.middle_wires[self.hidden-2]:
            j.update()
        for j in self.middle_n:
            j.train = 1
            j.pass_in()
            j.pass_out()
        for j in self.middle_wire_out:
            j.update()

    def move(self):
        self.in_net()
        for i in self.inwires:
            print("INPUT: ",i.value)
        self.mid_net()
        self.out_net()
        for i in self.wires_out:
            print("OUTPUT: ",i.value)

    def train(self):
        #Start at the outputs - for now just make it work with this set of data, then focus on something generic, maybe a new function that expects a certain array
        self.c = 1
        ##Start at the end -- self.wires_out this will be the predicted value, training data value will be the true
        j = 0
        #if (len(self.train_data.shape) > 2):
        #    if (self.train_data.shape[1] != len(self.wires_out) + 1):
        #        raise TypeError("Training data must have the same shape as the outputs of the net--> ","Number of Wires Out: ",len(self.wires_out)," Shape of Data Array: ",self.train_data.shape[1])
        print(len(self.train_data))
        ##This needs to be changed for inputs and outputs > 1
        while j < len(self.train_data):
            for i in self.inwires:
                i.value = float(self.train_data[j,1])
            self.move()
            diff = []
            for i in self.wires_out:
                diff.append(-(self.train_data[j,0]/i.value)-((1-self.train_data[j,0])/(1-i.value)))
                j+=1
                print("DIFF: ",diff)
            for i in self.outputs:
                print(i.value)
                #u+=1

        ##Calculate difference between true and predicted for the last neuron
        ##Go through this for every layer (dA)
        





data_arr = np.zeros((891,2))
data = open('train.csv',newline = '')
read = csv.reader(data,delimiter=',',quoting=csv.QUOTE_NONE)
length = 0
for row in read:
    if (row[2] == 'Pclass'):
        continue
    else:
        data_arr[length,1] = float(row[2])
        if (float(row[1]) == 1):
            data_arr[length,0] = float(row[1])
        else:
            data_arr[length,0] = -1
    length += 1
print(data_arr)

ne = net(1,10,10,1)
#ne.inwires[0].value = -2.0
#ne.move()
ne.train_data = data_arr
ne.train()
#n.inwires[0].value = data_arr[0,0]
#n.move()
#n.train_data = data_arr[1,:]
#n.train()


#im = image()
#print(im.frame.shape)
#import matplotlib.pyplot as plt
#plt.imshow(im.frame)
#plt.show()



####THINGS THAT NEED TO BE ADDED####

#  Read data continuously not just input-output-input
#  Find way to read off text file or csv and automate this as a function of net
#  Add in functions on nucleus and weights on wires
#  Need to write something to write the data to a file or something