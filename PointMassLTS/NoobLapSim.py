# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:31:54 2024

@author: Ananth
Man is trying something stupid and silly, but why even do anything else?
"""

import os
os.chdir('C:/Users/admin/Desktop/GitHub/BarelyOptimalLapSim')

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from setup.createTrack import createTrack
from setup.createCar import createCar

#%% Load Track

trackFile = "trackFiles/TrackFile.txt"

sLap, kt = createTrack(trackFile, 3000)


#%% Load Car

setupFile = "setupFiles/SetupFile.json"
data = createCar(setupFile)

mCar = data["mcar"]
muy  = data["gripy"]
mux  = data["gripx"]
nEngine = np.array(data["EngRpm"])
MEngine = np.array(data["EngNm"]) * data["reff"]
rGearRatio = np.array(data["rGearRat"])
radWheel = data["rtyre"]
Cx = data["cx"]

vMax = 300/3.6
g = 9.81

#%% Powertrain Model

# Calculate vWheel for each gear, along with its corresponding torque output 

v_wheel = np.zeros((len(nEngine), len(rGearRatio)))
M_wheel = np.zeros((len(nEngine), len(rGearRatio)))
PowerOut = np.zeros((len(nEngine), len(rGearRatio)))

for i in range(len(rGearRatio)):
    v_wheel[:,i] = np.multiply(radWheel, nEngine / (rGearRatio[i] * 9.55) )
    M_wheel[:,i] = MEngine * rGearRatio[i]

# extend vWheel_range

vWheel_range = np.linspace(0, vMax, 100)

MWheelOut = np.zeros((len(vWheel_range), len(rGearRatio)))

for i in range(len(rGearRatio)):
    MWheel_interp = interp.interp1d(v_wheel[:,i], M_wheel[:,i],bounds_error=False,fill_value='0')
    MWheelOut[:,i] = MWheel_interp(vWheel_range)

MWheelMax = np.amax(MWheelOut, axis=1)
MWheelMax_interp = interp.interp1d(vWheel_range, MWheelMax)

# Calculate Limit Speed and Find Apexes

limit_speed = np.minimum( abs( np.sqrt( muy * g / kt ) ), vMax)

index_min = np.argmin(limit_speed)

# Moving Forward Through Time and Space and Everything in Between
sector_distance = np.mean(np.diff(sLap))

vCar_forward = np.zeros(len(sLap))
vCar_forward[index_min] = limit_speed[index_min]

for i in range(index_min, len(sLap)-1):
    
    MWheel_delivered = MWheelMax_interp(vCar_forward[i])
    FWheel_delivered = MWheel_delivered / radWheel
    delivered_ax = (FWheel_delivered - 0.5*1.225*Cx*vCar_forward[i]**2)/ mCar
    
    accy = ( np.square(vCar_forward[i]) * kt[i] ) 
    max_ay = muy * g
    max_ax = mux * g
    tyre_potential_ax =  np.sqrt ( np.square(max_ax) * (1 - np.square( np.round( accy/max_ay ,2 ))) ) 
    
    combine_ax = min(delivered_ax, tyre_potential_ax)
    
    vCar_forward[i+1] = np.sqrt ( np.square(vCar_forward[i]) + 2 * combine_ax * sector_distance )
    if vCar_forward[i+1] > limit_speed[i+1]:
        vCar_forward[i+1] = limit_speed[i+1]

vCar_forward[0] = vCar_forward[-1]

for i in range(0, index_min):
    MWheel_delivered = MWheelMax_interp(vCar_forward[i])
    FWheel_delivered = MWheel_delivered / radWheel
    delivered_ax = (FWheel_delivered - 0.5*1.225*Cx*vCar_forward[i]**2)/ mCar
    
    accy = ( np.square(vCar_forward[i]) * kt[i] ) 
    max_ay = muy * g
    max_ax = mux * g
    tyre_potential_ax =  np.sqrt ( np.square(max_ax) * (1 - np.square( np.round( accy/max_ay ,2 ))) ) 
    
    combine_ax = min(delivered_ax, tyre_potential_ax)
    
    vCar_forward[i+1] = np.sqrt ( np.square(vCar_forward[i]) + 2 * combine_ax * sector_distance )
    if vCar_forward[i+1] > limit_speed[i+1]:
        vCar_forward[i+1] = limit_speed[i+1]
  
    
  # Now going backwards
  
vCar_reverse = np.zeros(len(sLap))
vCar_reverse[-1] = limit_speed[-1]
for i in range(len(sLap)-1,0,-1):
    accy = ( np.square(vCar_reverse[i]) * kt[i] ) 
    max_ay = muy * g
    max_ax = mux * g
    combine_ax =  np.sqrt ( np.square(max_ax) * (1 - np.square( np.round( accy/max_ay ,2 ))) ) 
    vCar_reverse[i-1] = np.sqrt ( np.square(vCar_reverse[i]) + 2 * combine_ax * sector_distance )
    if vCar_reverse[i-1] > limit_speed[i-1]:
        vCar_reverse[i-1] = limit_speed[i-1]

# Final Car Speed
vCar = np.zeros(len(sLap))
for i in range(len(sLap)):
   vCar[i] =  min(limit_speed[i], vCar_forward[i], vCar_reverse[i])

plt.figure(1)
plt.title("Velocity Profiles")
plt.plot(sLap , limit_speed, 'r-', linewidth=1, label="limit-speed")
plt.plot(sLap, vCar_forward, 'g-', linewidth=1, label="forward-speed")
plt.plot(sLap, vCar_reverse, 'b-', linewidth=1, label="reverse-speed")
plt.plot(sLap, vCar, 'k-', linewidth=2, label="final-vCar")
plt.xlabel('distance [m]')
plt.ylabel('limit speed [m/s]')
plt.legend()
plt.grid( visible='true' , which='major', linestyle=':')




