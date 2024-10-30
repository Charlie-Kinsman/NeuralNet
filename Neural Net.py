import csv
import math
import numpy as np
from Frames import image
import random as r
import matplotlib.pyplot as plt
class nucleus():
    def __init__(self,inputs,fun):
        self.inputs = inputs
        self.into = []
        #self.function = opt ##Look into dictionaries more, for now it's just one function
        for f in range(inputs):
            self.into.append(0)
        self.value = 0
        self.output = 0
        self.train = 0
        self.training_val = 0
        self.active = fun
    def func(self,val):
        if (self.active == "sig"):
            if (type(val) == float):
                self.value = (1/(1+math.exp(-val)))
            else:
                self.value = (1/(1+math.exp(-sum(val))))
        if (self.active == "rel"):
            if (type(val) == float):
                self.value = max(0,val)
            else:
                val.append(0)
                self.value = max(val)
    def pass_out(self):
        if (self.value != 0):
            self.output = self.value
            if (self.train == 0):
                self.value = 0
            else:
                self.value *=1
    def pass_in(self):
        if (type(self.into) == float):
            self.func(self.into)
            self.into = 0
        else:
            self.func(self.into)
            for f in self.into:
                f = 0

    
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
        if (self.value >= 0.5):
            self.value = self.value/abs(self.value)
        else:
            self.value = 0



class wire():
    def __init__(self,input,output,num):
        self.input = input
        self.output = output
        self.value = 0
        self.into = num
        n = r.random()
        self.weight = n
        self.bias = 0
        if (type(self.input) != nucleus or type(self.output) != nucleus):
            raise TypeError("Must have nuclei as inputs and outputs")
    def update(self):
        self.value = (self.input.output * self.weight) + self.bias
        self.output.into[self.into] = self.value ##This needs to be changed, not append


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
            self.inputs[i] = nucleus(1,"rel")
        ###################
        self.inwires = []
        for i in range(self.input):
            self.inwires.append(wire_in(self.inputs[i]))
        ###Set Up Hidden Layers###
        self.middle_1 = []
        for i in range(self.width):
            self.middle_1.append(nucleus(self.input,"rel"))
        self.middle_wire_in = []
        for i in range(self.input):
            for j in range(self.width):
                self.middle_wire_in.append(wire(self.inputs[i],self.middle_1[j],i))
        self.middle_n = []
        for i in range(self.width):
            self.middle_n.append(nucleus(self.width,"rel"))
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
                    self.middle[i][j] = nucleus(self.width,"rel")
            for i in range(self.hidden-1):
                self.middle_wires.append([])
            for i in range(self.hidden-1):
                for j in range(self.width):
                    for k in range(self.width):
                        if (i==0):
                            self.middle_wires[i].append(wire(self.middle_1[j],self.middle[i][k],j))
                        elif (i==self.hidden-2):
                            self.middle_wires[i].append(wire(self.middle[i-1][j],self.middle_n[k],j))
                        else:
                            self.middle_wires[i].append(wire(self.middle[i-1][j],self.middle[i][k],j))
        ########################## 
        ###Set Up Output###
        self.outputs = []
        for i in range(self.output):
            self.outputs.append(nucleus(self.width,"sig"))
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
            i.pass_in()
            i.pass_out()
        for i in self.middle_wire_in:
            i.update()
    def out_net(self):
        for i in self.middle_wire_out:
            i.update()
        for i in self.outputs:
            i.pass_in()
            i.pass_out()
        for i in self.wires_out:
            i.update()
    def mid_net(self):
        for i in self.middle_1:
            i.pass_in()
            i.pass_out()
        for i in range(self.hidden-2):
            for j in self.middle_wires[i]:
                j.update()
            for j in self.middle[i]:
                j.pass_in()
                j.pass_out()
        for j in self.middle_wires[self.hidden-2]:
            j.update()
        for j in self.middle_n:
            j.pass_in()
            j.pass_out()

    def move(self):
        self.in_net()
        self.mid_net()
        self.out_net()

    def train(self):
        for j in self.inputs:
            j.train = 1
        for j in self.middle_1:
            j.train = 1
        for u in range(self.hidden-2):
            for j in self.middle[u]:
                j.train = 1
        for j in self.middle_n:
            j.train = 1
        for j in self.outputs:
            j.train = 1
        self.c = 1
        j = 0
        ##This needs to be changed for inputs and outputs > 1
        while j < len(self.train_data):
            if (j == 10):
                break
            for i in self.inwires:
                i.value = float(self.train_data[j,1])
            print("\nINPUT: ",self.inwires[0].value)
            self.move()
            print("\nNEURON OUTPUT: ",self.wires_out[0].input.output," WIRE OUTPUT: ",self.wires_out[0].value," TRAINING VALUE: ",self.train_data[j,0],"\n")
            b = 0
            for i in range(len(self.outputs)):
                diff = -((self.train_data[j,0]/self.outputs[i].value) + ((1-self.train_data[j,0])/(1-self.outputs[i].value)))
                dz = 0
                b = 0
                if (self.outputs[i].active == "sig"):
                    dz = self.outputs[i].value * (1 - self.outputs[i].value)
                if (self.outputs[i].active == "rel"):
                    if (diff > 0):
                        dz = 1
                    else:
                        dz = 0
                diff = diff * dz
                b += diff
                for t in range(len(self.middle_wire_out)):
                    if (self.middle_wire_out[t].output == self.outputs[i]):
                        self.middle_wire_out[t].input.training_val += self.middle_wire_out[t].weight * diff
                        self.middle_wire_out[t].weight -= (1/self.width) * self.middle_wire_out[i].input.value * diff * self.c
            print(self.middle_wire_out[0].input.training_val)
            for i in range(len(self.middle_wire_out)):
                self.middle_wire_out[t].bias -= (1/self.width) * b * self.c
            i = self.hidden - 2
            while (i >= 0):
                b = 0
                for t in self.middle_wires[i]:
                    diff = 0
                    if (t.output.active == "sig"):
                        diff = t.output.training_val * t.output.value * (1 - t.output.value)
                    if (t.output.active == "rel"):
                        if (t.output.value > 0):
                            diff = t.output.training_val
                        else:
                            diff = 0
                    b+=diff
                    t.input.training_val += t.weight * diff
                    t.weight -= (1/self.width) * t.input.value * diff * self.c
                print(b)
                for t in self.middle_wires[i]:
                    t.output.training_val = 0
                    t.bias -= (1/self.width) * b * self.c
                i-=1
            b = 0
            for t in self.middle_wire_in:
                diff = 0
                if (t.output.active == "sig"):
                    diff = t.output.training_val * t.output.value * (1 - t.output.value)
                if (t.output.active == "rel"):
                    if (t.output.value > 0):
                        diff = t.output.training_val
                    else:
                        diff = 0
                b+=diff
                t.input.training_val += t.weight * diff
                t.weight -= (1/len(self.inputs)) * t.input.value * diff * self.c
            for t in self.middle_wire_in:
                t.bias -= (1/len(self.inputs)) * b * self.c
            j+=1
            print(100*float(j)/float(len(self.train_data)),"%")
        for u in self.inputs:
            u.train = 0
        for u in self.middle_1:
            u.train = 0
        for u in range(self.hidden-2):
            for t in self.middle[u]:
                t.train = 0
        for t in self.middle_n:
            t.train = 0
        for t in self.outputs:
            t.train = 0

    def draw(self):
        max = 0.00001
        for i in self.middle_wire_in:
            if abs(i.weight) > max:
                max = abs(i.weight)
        for i in range(self.hidden - 1):
            for j in self.middle_wires[i]:
                if abs(j.weight) > max:
                    max = abs(j.weight)
        for i in self.middle_wire_out:
            if abs(i.weight) > max:
                max = abs(i.weight)
        x = []
        y = []
        for i in range(len(self.inputs)):
            x.append(0)
            y.append((i+1)*(self.width)/(len(self.inputs)+1))
        plt.scatter(x,y,c='green')
        x = []
        y = []
        for i in range(len(self.middle_1)):
            x.append(1)
            y.append(i)
        for i in range(self.hidden - 2):
            for j in range(self.width):
                x.append(i+2)
                y.append(j)
        for i in range(len(self.middle_n)):
            x.append(self.hidden)
            y.append(i)
        plt.scatter(x,y,c='blue')
        x = []
        y = []
        for i in range(len(self.outputs)):
            x.append(self.hidden+1)
            y.append((i+1)*(self.width)/(len(self.outputs)+1))
        plt.scatter(x,y,c='red')
        w_x = []
        w_y = []
        for i in range(len(self.inputs)):
            w_x.append(-1.5)
            w_x.append(0)
            w_y.append((i+1)*(self.width)/(len(self.inputs)+1))
            w_y.append((i+1)*(self.width)/(len(self.inputs)+1))
            plt.plot(w_x,w_y,c='green',alpha=1.0)
            w_x = []
            w_y = []
        for i in range(len(self.inputs)):
            for j in range(len(self.middle_1)):
                w_x.append(0)
                w_x.append(1)
                w_y.append((i+1)*(self.width)/(len(self.inputs)+1))
                w_y.append(j)
                tra = 0
                for u in self.middle_wire_in:
                    if (u.input == self.inputs[i] and u.output == self.middle_1[j]):
                        tra = u.weight/max
                if (tra > 0.0):
                    plt.plot(w_x,w_y,c='blue',alpha=abs(tra))
                elif (tra <= 0.0):
                    plt.plot(w_x,w_y,c='red',alpha=abs(tra))
                w_x = []
                w_y = []
        for i in range(len(self.middle_1)):
            for j in range(len(self.middle[0])):
                w_x.append(1)
                w_x.append(2)
                w_y.append(i)
                w_y.append(j)
                tra = 0
                for u in self.middle_wires[0]:
                    if (u.input == self.middle_1[i] and u.output == self.middle[0][j]):
                        tra = u.weight/max
                if (tra > 0.0):
                    plt.plot(w_x,w_y,c='blue',alpha=abs(tra))
                elif (tra <= 0.0):
                    plt.plot(w_x,w_y,c='red',alpha=abs(tra))
                w_x = []
                w_y = []
        for t in range(self.hidden-3):
            for i in range(self.width):
                for j in range(self.width):
                    w_x.append(t+2)
                    w_x.append(t+3)
                    w_y.append(i)
                    w_y.append(j)
                    tra = 0
                    for u in self.middle_wires[t+1]:
                        if (u.input == self.middle[t][i] and u.output == self.middle[t+1][j]):
                            tra = u.weight/max
                    if (tra > 0.0):
                        plt.plot(w_x,w_y,c='blue',alpha=abs(tra))
                    elif (tra <= 0.0):
                        plt.plot(w_x,w_y,c='red',alpha=abs(tra))
                    w_x = []
                    w_y = []
        for i in range(len(self.middle[-1])):
            for j in range(len(self.middle_n)):
                w_x.append(self.hidden-1)
                w_x.append(self.hidden)
                w_y.append(i)
                w_y.append(j)
                tra = 0
                for u in self.middle_wires[-1]:
                    if (u.input == self.middle[-1][i] and u.output == self.middle_n[j]):
                        tra = u.weight/max
                if (tra > 0.0):
                    plt.plot(w_x,w_y,c='blue',alpha=abs(tra))
                elif (tra <= 0.0):
                    plt.plot(w_x,w_y,c='red',alpha=abs(tra))
                w_x = []
                w_y = []
        for i in range(len(self.middle_n)):
            for j in range(len(self.outputs)):
                w_x.append(self.hidden)
                w_x.append(self.hidden+1)
                w_y.append(i)
                w_y.append((j+1)*(self.width)/(len(self.outputs)+1))
                tra = 0
                for u in self.middle_wire_out:
                    if (u.input == self.middle_n[i] and u.output == self.outputs[j]):
                        tra = u.weight/max
                if (tra > 0.0):
                    plt.plot(w_x,w_y,c='blue',alpha=abs(tra))
                elif (tra <= 0.0):
                    plt.plot(w_x,w_y,c='red',alpha=abs(tra))
                w_x = []
                w_y = []
        for i in range(len(self.outputs)):
            w_x.append(self.hidden+1)
            w_x.append(self.hidden+2.5)
            w_y.append((i+1)*(self.width)/(len(self.outputs)+1))
            w_y.append((i+1)*(self.width)/(len(self.outputs)+1))
            plt.plot(w_x,w_y,c='red',alpha=1.0)
            w_x = []
            w_y = []
        plt.show()





data_arr = np.zeros((891,2))
data = open('train.csv',newline = '')
read = csv.reader(data,delimiter=',',quoting=csv.QUOTE_NONE)
length = 0
for row in read:
    if (row[2] == 'Pclass'):
        continue
    else:
        data_arr[length,1] = float(row[10])
        if (float(row[1]) == 1):
            data_arr[length,0] = 0.99
        else:
            data_arr[length,0] = 0.001
    length += 1
m = 0
for i in data_arr[:,1]:
    if (i > m):
        m = i
data_arr[:,1] = data_arr[:,1]/m
ne = net(1,6,5,1)
#ne.inwires[0].value = -2.0
#ne.move()
ne.draw()
ne.train_data = data_arr
ne.train()
ne.draw()
####THINGS THAT NEED TO BE ADDED####

#  Read data continuously not just input-output-input
#  Find way to read off text file or csv and automate this as a function of net
#  Add in functions on nucleus and weights on wires
#  Need to write something to write the data to a file or something