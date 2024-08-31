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
    14, #14
    15, #15
    16, #16
    17, #17
])


#vehicles
K = np.array([
    0, 
    1, 
    2,
])


coords = np.array([
    [90,  85], #0
    [75,  76], #1
    [96,  44], #2
    [50,  5],  #3
    [49,  8],  #4
    [13,  7],  #5
    [29,  89], #6
    [58,  30], #7
    [70,  39], #8
    [14,  24], #9
    [25,  39], #10
    [40,  25], #11
    [30,  15], #12
    [25,  65], #13
    [20,  80], #14
    [25,  70], #15
    [34,  75], #16
    [90,  85], #17
])


demands = np.array([
    0, #0
    2, #1
    1, #2
    1, #3
    1, #4
    2, #5
    2, #6
    1, #7
    1, #8
    1, #9
    1, #10
    2, #11
    1, #12
    1, #13
    1, #14
    1, #15
    1, #16
    0, #17
])

#time windows    
a = np.array([
    4, #0
    7, #1
    0, #2
    0, #3
    7, #4
    7, #5
    0, #6
    0, #7
    0, #8
    0, #9
    7, #10
    5, #11
    9, #12
    9, #13
    8, #14
    8, #15
    4, #16
    4, #16
])

b = np.array([
    15, #0
    24, #1
    24, #2
    7, #3
    24, #4
    24, #5
    7, #6
    24, #7
    24, #8
    24, #9
    24, #10
    14, #11
    18, #12
    13, #13
    11, #14
    11, #15
    18, #16
    15, #17
])

service_time = np.array([
    0, #0
    0, #2
    0, #3
    0, #4
    0, #5
    0, #6
    0, #7
    0, #8
    0, #9
    0, #10
    0, #11
    0, #12
    0, #13
    0, #14
    0, #15
    0, #16
    0 #17
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
	20,
	20,
	20,
])
