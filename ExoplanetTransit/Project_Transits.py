#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from functions import save

#ai
data=np.genfromtxt("wasp_4b.tsv", comments="#", delimiter=";", dtype="float64")
#get stuff
date=data[:, 0]#put in arrays
flux=data[:,1]
error=data[:,2]
date=(date-date[0])*24

plt.plot(date, flux)#plot it
plt.xlabel("Time (days)")
plt.ylabel("Flux (watts/m^s)")
plt.savefig("hill_project6ai1.png")
plt.show()

#aii

#aiii
transits=[]
flux_tran=[]
error_tran=[]
timelist=[]
fluxlist=[]
errorlist=[]
for i in range(len(date)-1):
    timelist.append(date[i])
    int=date[i+1]-date[i]
    fluxlist.append(flux[i])
    errorlist.append(error[i])
#loop to start at zero and make into arrays
    if int>2.4:
        timelist=np.array(timelist)
        transits.append(timelist-timelist[0])
        flux_tran.append(np.array(fluxlist))
        error_tran.append(np.array(errorlist))
        timelist=[]
        errorlist=[]
        fluxlist=[]


timelist=np.array(timelist)
transits.append(timelist-timelist[0])
flux_tran.append(np.array(fluxlist))
error_tran.append(np.array(errorlist))
#make arrays

transits=np.array(transits)
error_tran=np.array(error_tran)
flux_tran=np.array(flux_tran)

plt.plot(transits[0], flux_tran[0])
plt.savefig("hill_project6ai.png")
plt.show()
#plot

fig,axis=plt.subplots(2,3,figsize=(12,9))#make figure 
axis[0,0].plot(transits[0],flux_tran[0])
axis[0,0].set_ylabel("Flux(watts/m^2) ")
axis[0,0].set_xlabel("Time(days)")
axis[0,0].set_title("Transit 1")
axis[0,1].plot(transits[1],flux_tran[1])
axis[0,1].set_ylabel("Flux(watts/m^2)")
axis[0,1].set_xlabel("Time(days)")
axis[0,1].set_title("Transit 2")
axis[0,2].plot(transits[2],flux_tran[2])
axis[0,2].set_ylabel("Flux(watts/m^2)")
axis[0,2].set_xlabel("Time(days)")
axis[0,2].set_title("Transit 3")
axis[1,0].plot(transits[3],flux_tran[3])
axis[1,0].set_ylabel("Flux(watts/m^2)")
axis[1,0].set_xlabel("Time(days)")
axis[1,0].set_title("Transit 4")
axis[1,1].plot(transits[4],flux_tran[4])
axis[1,1].set_ylabel("Flux(watts/m^2)")
axis[1,1].set_xlabel("Time(days)")
axis[1,1].set_title("Transit 5")
axis[1,2].plot(transits[5],flux_tran[5])
axis[1,2].set_ylabel("Flux(Watts/m^2)")
axis[1,2].set_xlabel("Time(days)")
axis[1,2].set_title("Transit 6")
for label in axis[0,1].get_xticklabels()[::2]:
    label.set_visible(False)#make less tick marks so looks better                      
fig.tight_layout()
plt.savefig("hill_project6aii.png")
plt.show()
plt.close()

#aiv
plt.plot(transits[0], flux_tran[0], label="1")
plt.plot(transits[1], flux_tran[1], label="2")
plt.plot(transits[2], flux_tran[2], label="3")
plt.plot(transits[3], flux_tran[3], label="4")
plt.plot(transits[4], flux_tran[4], label="5")
plt.plot(transits[5], flux_tran[5], label="6")
plt.legend()
plt.savefig("hill_project6aiv.png")
plt.show()


save("hill_project6a.txt", ("transits", "1", "2", "3", "4", "5"), transits)
save("hill_project6aflux.txt",("1","2", "3","4","5","6"), flux_tran)
save("hill_project6aerror.txt", ("1","2","3","4","5","6"),error_tran)
#basiaclly lots of plots and save it


# In[ ]:





# In[ ]:




