import numpy as np

# Velocidad por defecto del vehiculo en km/h
VD = 40

# Flete en dolars
f = 15 

# Consumo de combustible del 
#vehiculo colector en litros por km (l/km). 
CC = 0.35 

#patients
N = np.array([
    0, #0 
    1, #1 
    2, #2
    3, #3
    4, #4
    5, #5
    6, #6
    7, #7
    8, #8
    9, #9
    10, #10
    11, #11
    12, #12
    13, #13
])


#vehicles
K = np.array([
    0, 
    1, 
    2,
])


coords = np.array([
        [90,  85], #1
        [75,  76], #2
        [96,  44], #3
        [50,  5],  #4
        [49,  8],  #5
        [13,  7],  #6
        [29,  89], #7
        [58,  30], #8
        [70,  39], #9
        [14,  24], #10
        [25,  39], #11
        [40,  25], #12
        [90,  85], #13
])


demands = np.array([
    0,  #1
    10, #2
    30, #3
    10, #4
    10, #5
    10, #6
    20, #7
    20, #8
    20, #9
    10, #10
    10, #11
    10, #12
    0   #13
])

#time windows    
a = np.array([
    4,
    7,
    0,
    0,
    7,
    7,
    0,
    0,
    0,
    0,
    7,
    5, 
    4, 
])

b = np.array([
    15,
    24,
    24,
    7,
    24,
    24,
    7,
    24,
    24,
    24,
    24,
    14,
    15,
])

service_time = np.array([
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
])

# Capacidad máxima del tanque de combustible 
# del vehiculo colector.
T = np.array([
    15,
    15,
    15,   
])  

# Emisión de CO2 del vehiculo colector en 
# gr/l de combustible del vehiculo.
EM = np.array([
    2.7,
    2.7,
    2.7,   
])  

# capacidad de los vehiculos k = v1, v2, v3.
Q_MAX = np.array([
	80,
	80,
	80,
])
