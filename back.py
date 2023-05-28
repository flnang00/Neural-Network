import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

numIter=10000

w1=0.1
w2=0.2
w3=.3
w4=.4
w5=.5
w6=.7
w7=.3
w8=.6
w9=.7
w10=.8
w11=.9
w10=.1

w_list=[w1,w2,w3,w4,w5,w6,w7,w8,w9,w10]

b1=.5
b2=.5
b_list=[b1,b2]
#Input and TargetValues
x1=1
x2=2
x3=3
x4=4
x5=5
x_list=[x1,x2,x3]
t1=1
t2=2
t3=4
t4=16
t5=25
t_list=[t1,t2,t3,t4,t5]
#set learningrate
alpha=.01

def sigmoid(x): #do sigmoid  
    exp_Value=np.exp(-x)
      
    return 1/1+exp_Value


def forwardProp(x_list,w_list,b_list):
    zh1=w_list[0]*x_list[0] + w_list[2]*x_list[1] + w_list[4]*x_list[2] +b_list[0]
    zh2=w_list[1]*x_list[0] + w_list[3]*x_list[1] + w_list[5]*x_list[2] +b_list[0]
    zh3=
    h1=sigmoid(zh1)
    h2=sigmoid(zh2)
    zo1=w_list[6]*h1+w_list[8]*h2+b_list[1]
    zo2=w_list[7]*h1+w_list[9]*h2+b_list[1]
    o1=sigmoid(zo1)
    o2=sigmoid(zo2)
    return h1,h2,o1,o2


def error(oList,t_list):
    return .5*(np.power(oList[0]-t_list[0],2)+np.power(oList[1]-t_list[1],2))



errList=[]
for i in range(numIter):

    #forward propogation


    h1,h2,o1,o2 = forwardProp(x_list,w_list,b_list)

    #compute Error
    sse=error([o1,o2], t_list)
    errList.append(sse)


    print('Running'+ str(i+1)+'of'+str(numIter))
    print('o1'+ str(o1))
    print('t1:'+str(t1))
    print('o2:'+str(o2))
    print('t2:'+str(t2))
    print('o1:'+str(t1))
    print('errror:'+str(sse))
    print('')


    #Error derivative calculations


    #compute dE_dw7
    dE_do1=o1-t1
    do1_dzo1=o1*(1-o1)
    dzo1_dw7=h1
    dE_dw7=(o1-t1)*o1*(1-o1)*h1
    #compute dE_dw8
    dE_do2=o2-t2
    do2_dzo2=o2*(1-o2)
    dzo2_dw8=h1
    dE_dw8=dE_do2*do2_dzo2*dzo2_dw8

    #compute dE_dw9
    dE_do2=o1-t1
    do1_dzo1=o1*(1-o1)
    dzo1_dw9=h2
    dE_dw9=(o1-t2)*o2*(1-o2)*dzo1_dw9

    #compute dE_w10
    dE_do2=o2-t2 #look w8
    do2_zo2=o2*(1-o2)#look w8
    dzo2_dw10=h2
    dE_dw10=dE_do2*do2_zo2*dzo2_dw10

    #compute dE_db2
    dzo1_db2=1
    dzo2_db2=1

    dE_db2=dE_do1*do1_dzo1*dzo1_db2+dE_do2*do2_dzo2*dzo2_db2


    #now dE_w1:
    #first compute dE_dh1
    dzo1_dh1=w7
    dzo2_dh1=w8


    dE_dh1=dE_do1*do1_dzo1*dzo1_dh1 + dE_do2*do2_dzo2*dzo2_dh1

    #compute dE_w1


    dh1_dzh1=h1 *(1-h1)
    dzh1_dw1=x1

    dE_dw1= dE_dh1*dh1_dzh1*dzh1_dw1


    #compute dE_dw3

    dzh1_dw3=x2
    dE_dw3=dE_dh1*dh1_dzh1*dzh1_dw3

    #compute dE_dw5

    dzh1_dw5=x3
    dE_dw5=dE_dh1*dh1_dzh1*dzh1_dw5
    
    #compute dE_dw2
    dzo2_dh2=w10
    dzo1_dh2=w9
    dE_dh2=dE_do1*do1_dzo1*dzo1_dh2 + dE_do2*do2_dzo2*dzo2_dh2

    dh2_dzh2= h2*(1-h2)
    dzh2_dw2=x1
    dE_dw2=dE_dh2*dh2_dzh2*dzh2_dw2

    #compute dE_dw4
    dzh2_dw4=x2
    dE_dw4=dE_dh2*dh2_dzh2*dzh2_dw4

    #compute dE_w6

    dzh2_dw6=x3
    dE_dw6=dE_dh2*dh2_dzh2*dzh2_dw6


    #compute dE_db1
    dzh1_db1=1
    dzh2_db1=1

    term1=dE_do1*do1_dzo1*dzo1_dh1*dh1_dzh1*dh1_dzh1*dzh1_db1
    term2=dE_do2*do2_dzo2*dzo1_dh1*dh1_dzh1*dh1_dzh1*dzh1_db1

    dE_db1= term1 + term2

    #Update all parameters

    w1 = w1-alpha * dE_dw1
    w2= w2-alpha * dE_dw2
    w3= w3-alpha * dE_dw3
    w4= w4-alpha * dE_dw4
    w5= w5-alpha * dE_dw5
    w6= w6-alpha * dE_dw6
    w7= w7-alpha * dE_dw7
    w8= w8-alpha * dE_dw8
    w9= w9-alpha * dE_dw9
    w10= w10-alpha * dE_dw10

    b1=b1-alpha*dE_db1
    b2=b2-alpha*dE_db2

    w_list=[w1,w2,w3,w4,w5,w6,w7,w8,w9,w10]
    b_list=[b1,b2]




df=pd.DataFrame(errList,columns=['SSE'])
df.plot()

plt.show()

    




    











