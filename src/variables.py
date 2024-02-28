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
    0, 
    1, 
    2, 
    3, 
    4, 
    5,  
    6, 
    7, 
    8, 
    9, 
    10,
    11,
    12
])


#vehicles
K = np.array([
    0, 
    1, 
    2,
])


coords = np.array([
        [90,  85],
        [75,  76],
        [96,  44],
        [50,  5],
        [49,  8],
        [13,  7],
        [29,  89],
        [58,  30],
        [70,  39],
        [14,  24],
        [25,  39],
        [40,  25], 
        [90,  85],
])


demands = np.array([
    0,
    10,
    30,
    10,
    10,
    10,
    20,
    20,
    20,
    10,
    10,
    10,
    0
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
