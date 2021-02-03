#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def linfit(distances, recvel):
    """                                                                   
    A function to find needed value for a linear fit                                                                                                 
    Variables                                                           
    ---------------                                                    
    Distances:list, distances or x axis of line                                 
    Recvel: list, the recessional velocities of the data                        
    """
    x=np.array(distances)#convert distances and velocities to arrays     
    y=np.array(recvel)
    dim=np.vstack((np.ones(len(x)),x))#add a row of ones to the matrix       
    a=np.dot((np.linalg.inv(np.dot(dim,dim.T))),(np.dot(y,dim.T)))#make equation for line(find slope and y intercept                                        
    sigmay=np.sqrt(((1/(float(len(y))-2)*np.sum((y-(a[0]+(a[1]*x)))**2))))#findsigma y value
    delta=(len(y)*sum(x**2))-((sum(x))**2)
    deltaA=sigmay*(np.sqrt((float(sum(x**2)))/delta))#error in a              
    deltaB=sigmay*(np.sqrt(float(len(y))/delta))#error in b                   
    aerrmax=[(a[0]+deltaB), (a[1]+deltaA)]#find maximim error                 
    aerrmin=[(a[0]-deltaB), (a[1]-deltaA)]#find minimum error                 
    exp=(a[1]*x)+a[0]#make line equation                                      
    chisq=np.sum((y-exp)**2/(np.abs(exp)))#find how fit the data is to best fit line                                                                          
    p=1-stats.chi2.cdf(chisq, len(distances))#find p value                    
    return a[1], a[0]#return these values

def save(filename, varnames, data):
    """                                                                         
    Filename=string specifying name of file to be saved                         
    varnames=tuple of variable names as strings                                 
    data=tuple of arrays                                                        
    """

    fyl=open(filename, 'wb')
    for i, name in enumerate(varnames):
        fyl.write(name + '\n')#.\n is a linebreak                               

        var=data[i]
        shape=var.shape
        shape=','.join(np.array(shape, dtype=str))
        fyl.write(shape+'\n')

        dtype=str(var.dtype)
        fyl.write(dtype+'\n')

        var_str=var.flatten().tobytes()
        fyl.write(var_str+'\n\n')

    fyl.close()

def restore(filename):
    """                                                                            Filename=string specifying name of file to be saved                        
    """
    fyl=open(filename, 'rb')

    print("restoring variables: \n")
    data=[]
    while True:
        var_name=fyl.readline()
        if var_name=="":break
        print(var_name)

        shape=fyl.readline().replace('\n','')
        shape=shape.split(',')
        shape=np.array(shape, dtype=int)

        dtype=fyl.readline().replace("\n","")

        data_str=""
        line=""
        while line!="\n":
            data_str+=line
            line=fyl.readline()
        array=np.fromstring(data_str[:-1], dtype=dtype)
        array=array.reshape(shape)
        data.append(array)
    fyl.close()
    return data

