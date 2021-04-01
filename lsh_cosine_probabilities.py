import math
import numpy
import matplotlib.pyplot as plt
# Import math Library
import math
import numpy as np

# Return the arc cosine of numbers

sim= numpy.linspace(0, 1, 1000)

fig = plt.figure(figsize=(6,6))

p=[]

for s in sim:
    p.append(1-(math.acos(s)/math.pi))
    
v1=[1, 2, 3, 4 , 5, 6 , 7,  8 , 9, 10, 17 ]
v2=[1, 1, 1, 2 , 3, 4 , 5 , 6 , 7, 9,  40 ]

for i,j in zip(v1,v2):
    plt.plot(sim , 1-(1-np.array(p)**i)**j, label="k=" + str(i) + ", L=" + str(j))
    plt.legend()
plt.show()



p=[]

for s in sim:
    p.append(1-(math.acos(s)/math.pi))
    
v1=[1, 2, 3, 4 , 5, 6 , 7 , 8 , 9, 10, 17 ]    
v2=[1, 1, 1, 1 , 1, 1 , 1,  1 , 1, 1,  40]

for i,j in zip(v1,v2):
    plt.plot(sim , 1-(1-np.array(p)**i)**j, label="k=" + str(i) + ", L=" + str(j))
    plt.legend()
plt.plot(sim,sim)
plt.show()
