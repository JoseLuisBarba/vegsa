from copy import deepcopy
from queue import PriorityQueue
from typing import Any, List, Tuple
from collections import deque
import numpy as np
import pandas as pd
from math import exp, log, sqrt
import matplotlib.pyplot as plt
import plotly.express as px
from random import random
from abc import abstractmethod, ABC

from src.variables import N, K, coords, a, b, service_time, demands, Q_MAX, VD, T 
from src.functions import Distance, Time, Co2Generation, TransportCost, PddrffValue


class sa_cooling_operator(ABC):
    @abstractmethod
    def __call__(self, step: int) -> float:
        pass

class linear_multiplicative_cooling(sa_cooling_operator):
    def __init__(self, t_max: float, alpha: float) -> None:
        super().__init__()
        self.t_max = t_max
        self.alpha = alpha
    
    def __call__(self, step: int) -> float:
        return self.t_max / (1 + self.alpha * step)

class linear_additive_cooling(sa_cooling_operator):
    def __init__(self, t_max: float, t_min: float, alpha: float, step_max: int) -> None:
        super().__init__()
        self.t_max = t_max
        self.t_min = t_min
        self.alpha = alpha
        self.step_max = step_max
    
    def __call__(self, step: int) -> float:
        return self.t_min + (self.t_max - self.t_min) * (self.step_max - step) / self.step_max


class cuadratic_multiplicative_cooling(sa_cooling_operator):
    def __init__(self, t_max: float, alpha: float ) -> None:
        super().__init__()
        self.t_max = t_max
        self.alpha = alpha

    
    def __call__(self, step: int) -> float:
        return self.t_min + (1 + self.alpha * step**2)
    

class cuadratic_additive_cooling(sa_cooling_operator):
    def __init__(self, t_max: float, t_min: float, alpha: float, step_max: int) -> None:
        super().__init__()
        self.t_max = t_max
        self.t_min = t_min
        self.alpha = alpha
        self.step_max = step_max

    
    def __call__(self, step: int) -> float:
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step) / self.step_max)**2
    
class exponential_multiplicative_cooling(sa_cooling_operator):
    def __init__(self, t_max: float, alpha: float) -> None:
        super().__init__()
        self.t_max = t_max
        self.alpha = alpha
    
    def __call__(self, step: int) -> float:
        return self.t_max * (self.alpha ** step)
    
class logarithmic_multiplicative_cooling(sa_cooling_operator):
    def __init__(self, t_max: float, alpha: float) -> None:
        super().__init__()
        self.t_max = t_max
        self.alpha = alpha
    
    def __call__(self, step: int) -> float:
        return self.t_max / (self.alpha * log(step + 1))



class BaseModel:

    def __init__(self,) -> None:
        print('created')

    def proximity(
            self, 
            waitTime: float, 
            urgencyTime: float,  
            timeRemaining: float, 
            alpha: float=0.33, 
            beta: float= 0.33, 
            gamma: float=0.34
        ) -> float:
        return alpha * waitTime + beta * urgencyTime + gamma * timeRemaining
    
    def nearestNeighborHeuristic(self, chromSize) -> np.ndarray:

        chromList = list()
        noSelected = set(N[1:-1])
        pq = PriorityQueue(maxsize= chromSize)
        vehicle_n: int =  0
        change: bool = True

        while len(chromList) < chromSize:

            # de wharehouse al vecino mas proximo
            if change:
                q_n: float =  0
                fuel_expen_t: float = 0
                leave_t: float = a[0] + service_time[0]
                q_n +=  demands[0]
                lastSelected = 0
                found = False



            # encontrar al vecino más proximo que cumpla con las condiciones
            for i in noSelected:
                # ultima informacion
                leave_test: float  = leave_t
                vehicle: int = vehicle_n
                fuel_test: float = fuel_expen_t
                q_test: float = q_n

                if lastSelected == 0 or change:
                    dist_test: float = Distance()(coords[0], coords[i])
                    change = False
                else:
                    dist_test: float = Distance()(coords[lastSelected], coords[i])


                time_test: float = Time()(dist_test)
                arrival_test: float = leave_test + time_test
                co2_test: float = Co2Generation()(vehicle, dist_test)
                fuel_test += co2_test
                q_test += demands[i]


                dist_to_wharehouse: float = Distance()(coords[i], coords[0])
                fuel_expen_wh: float = Co2Generation()(vehicle, dist_to_wharehouse) 
                return_time_t1: float = a[i] + service_time[i] + Time()(dist_to_wharehouse)
                return_time_t2: float = arrival_test + service_time[i] + Time()(dist_to_wharehouse)
                fuel_exp = fuel_test + fuel_expen_wh

                # restricciones
                earlyArrivalCond = (
                    arrival_test < a[i], 
                    q_test <= Q_MAX[vehicle], 
                    fuel_exp <= T[vehicle], 
                    return_time_t1 <= b[0]  
                )

                twArrivalCond = (
                    a[i] <= arrival_test <= b[i], 
                    q_test <= Q_MAX[vehicle], 
                    fuel_exp <= T[vehicle], 
                    return_time_t2 <= b[0]  
                )

                noArrivalCond = (
                    arrival_test > b[i], 
                    q_test > Q_MAX[vehicle],  
                    return_time_t1 > b[0], 
                    return_time_t2 > b[0], 
                    fuel_exp  > T[vehicle] 
                )

                if all(earlyArrivalCond):
                    found = True
                    leave_test = a[i] + service_time[i]
                    waitTime: float = a[i] 
                    urgencyTime: float = b[i]
                    timeRemaining: float =  a[i] + service_time[i]
                    pq.put((
                        self.proximity(waitTime, urgencyTime, timeRemaining),
                        i,
                        {
                            'leave_t':leave_test,
                            'q_n':q_test,
                            'fuel_expen_t':fuel_test,
                        }
                    ))

                if all(twArrivalCond):
                    found = True
                    leave_test = arrival_test + service_time[i]
                    waitTime: float = arrival_test
                    urgencyTime: float = b[i]
                    timeRemaining: float =  leave_test
                    pq.put((
                        self.proximity(waitTime, urgencyTime, timeRemaining),
                        i,
                        {
                            'leave_t':leave_test,
                            'q_n':q_test,
                            'fuel_expen_t':fuel_test,
                        }
                    ))
                elif any(noArrivalCond):
                    pass                


            if not found: 
                pq = PriorityQueue(maxsize=chromSize)
                vehicle_n += 1
                change = True
                continue

            else:
                change = False
                found = False

                _, proximityNode, summary = pq.get()
                pq = PriorityQueue(maxsize=chromSize)
                noSelected.discard(proximityNode)
                chromList.append(proximityNode)

                #update
                leave_t = summary['leave_t']
                q_n = summary['q_n']
                fuel_expen_t = summary['fuel_expen_t']
                lastSelected = proximityNode
        return np.array(chromList)

    def objective_function(self, chromosome: np.ndarray) -> float:
        r , df = self.decoding(chromosome.astype(int), summary=True) 
        if len(r) == 0:
            return float('inf')
        co2: float = df['CO2'].sum()
        tcost: float = TransportCost()(df['Distancia'].sum())
        return  co2 + tcost    
    
    def encoding(self, route: dict) -> np.ndarray:
        values = [item[0] for sublist in route.values() for item in sublist if item[0] != 0]
        result_array = np.array(values)
        return result_array

    #check
    def initial_population(self, PopSize: int, ChroSize: int) -> np.ndarray:
        """Generar poblacion
        Args:
            PopSize (int): tamaño de la poblacion
            ChroSize (int): numero de alelos
        Returns:
            np.ndarray: Poblacion de cromosomas
        """
        population = np.zeros(shape=(PopSize, ChroSize))
        for i in range(0, PopSize):
            population[i] = self.create_chromosome(n=ChroSize)
        return population
    # check
    def create_chromosome(self, n: int) -> np.ndarray:
        """ generar un cromosoma permutando una secuencia de alelos
        Args:
            n (int): numero de alelos
        Returns:
            np.ndarray: permutacion de alelos
        """
        chromosome = np.arange(1, n+1) #secuencia 
        np.random.shuffle(chromosome)
        return chromosome
    
    def stopping_restrictions(self, step):
        return step < self.step_max 
    
    def population_objs_fun(self, chroPopulation: np.ndarray) -> np.ndarray:
        popSize = chroPopulation.shape[0]
        eval_array = np.zeros(shape=(popSize, )) 
        for i in range(0, popSize):
            eval_array[i] = self.objective_function(chroPopulation[i])
        return eval_array
    

    #check
    def mutation_operator(self, chromosome_i: np.ndarray):
        """_summary_
        Args:
            chromosome_i (np.ndarray): _description_
        Returns:
            _type_: _description_
        """
        chromosome = chromosome_i.copy()
        a = np.random.randint(0, chromosome.shape[0] - 1)
        b = np.random.randint(0, chromosome.shape[0] - 1)
        while True:
            b = np.random.randint(0, chromosome.shape[0] - 1)
            if (a != b) and (a+1 != b) and (a != b+1) and (a+1 != b+1):
                break
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
        chromosome[a+1], chromosome[b+1] = chromosome[b+1], chromosome[a+1]

        if self.objective_function(chromosome_i) <= self.objective_function(chromosome):
            return chromosome_i
        return chromosome
    
    def mutation(self, population: np.ndarray, PopSize: int, ElitePopSize: int, MutPro: float):
        j: int = ElitePopSize 
        while j < PopSize:
            if j == PopSize:
                break
            sigma: float = np.random.uniform(0, 1, size=1)[0]
            if sigma <=  MutPro:
                population[j] = self.mutation_operator(population[j])
            j += 1
        return population
    

    def crossover_operator(self, chromosome1: np.ndarray, chromosome2: np.ndarray):
        c1 = chromosome1.copy()
        c2 = chromosome2.copy()
        a = np.random.randint(0, chromosome1.shape[0] - 1)
        b = np.random.randint(0, chromosome1.shape[0])
        while True:
            b = np.random.randint(0, chromosome1.shape[0])
            if b > a:
                break
        
        offspring1 = np.zeros(shape=chromosome1.shape)
        offspring2 = np.zeros(shape=chromosome1.shape)

        # offspring 1
        swath = chromosome1[a: b + 1]
        offspring1[a: b + 1] = swath
        idx_swath = np.isin(chromosome2, swath)
        alleles_c2 = deque()

        for select, allele in zip(idx_swath, chromosome2):
            if not select:
                alleles_c2.append(allele)
        offspring = offspring1.copy()

        for i, allele in enumerate(offspring):
            if allele == 0:
                offspring1[i] = alleles_c2.popleft()

        # offspring 2
        swath = chromosome2[a: b + 1]
        offspring2[a: b + 1] = swath
        idx_swath = np.isin(chromosome1, swath)
        alleles_c1 = deque()

        for select, allele in zip(idx_swath, chromosome1):
            if not select:
                alleles_c1.append(allele)
        offspring = offspring2.copy()

        for i, allele in enumerate(offspring):
            if allele == 0:
                offspring2[i] = alleles_c1.popleft()

        candidates = [
            (self.objective_function(c1), c1),
            (self.objective_function(c2), c2),
            (self.objective_function(offspring1),offspring1),
            (self.objective_function(offspring2),offspring1)
        ]
        sorted_candidates = sorted(candidates, key=lambda x: x[0])
        return sorted_candidates[0][1], sorted_candidates[1][1]
    
    def crossover(self, matingPool: np.ndarray , PopSize: int, ElitePopSize: int, CrossPro: float):
        offspring = np.zeros(shape=(PopSize, matingPool.shape[1]))
        offspring[0: ElitePopSize]  = matingPool[0:ElitePopSize] # mantenemos poblacion elite
        j: int = ElitePopSize # cuantos van seleccionandose 
        while j < PopSize:
            if j == PopSize:
                break
            a = np.random.randint(0, matingPool.shape[0])
            b = np.random.randint(0, matingPool.shape[0])
            mu: float = np.random.uniform(0, 1, size=1)[0]
            if mu <=  CrossPro:
                child1, child2 = self.crossover_operator(
                    chromosome1=matingPool[a],
                    chromosome2=matingPool[b]
                )
                offspring[j] = child1
                offspring[j+1] = child2
            else:
                continue
            j += 2
        return offspring
    
    def decoding(self, chromosome: np.ndarray, summary: bool= False):
        #summary
        infoData = {
            'Vehiculo':[],
            'Nodoi-1':[],
            'Nodoi':[],
            'Distancia':[],
            'TiempoArco':[],
            'TW_A':[],
            'TW_B':[],
            'Tanque':[],
            'CO2':[],
        }
        # colocar restricciones a n-vehiculos
        vehicle_n: int =  0 # numero del vehiculo actual
        q_n: float =  0  #actual carga
        fuel_expen_t: float = 0
        fuel_exp: float = 0 #actual consumo de combustible
        arrival_t: float = a[0] # el tiempo de llegada al nodo t
        leave_t: float = a[0] + service_time[0] # tiempo de salida del nodo t


        r: dict = dict()  #ruta
        r[0] = [[0, vehicle_n], ]
        q_n +=  demands[0] #demanda del almacen siempre es 0

        idx: int = 0
        change: bool = True
        excessVehicle: int = K.shape[0]
        while idx < len(chromosome):
            #distancia del nodo i-1 al nodo i
            
            
            if idx == 0 or change:
                dist_t: float = Distance()(coords[0], coords[chromosome[idx]])
            else:
                dist_t: float = Distance()(coords[chromosome[idx-1]], coords[chromosome[idx]])
            #Tiempo entre arco
            time_t: float = Time()(dist_t)
            #Tiempo de llegada del nodo i-1 al nodo i
            arrival_t = leave_t + time_t

            
            #consumo de combustible del nodo i-1 al nodo i
            co2_gen = Co2Generation()(vehicle_n, dist_t)
            fuel_expen_t += co2_gen

            #carga que queda saliendo del nodo j
            q_n += demands[chromosome[idx]]

            # si existe un retorno a warehouse del nodo i
            dist_to_wharehouse: float = Distance()(coords[chromosome[idx]], coords[0])
            fuel_expen_wh: float = Co2Generation()(vehicle_n, dist_to_wharehouse)
    
            #debe tener suficiente tiempo para retornar a warehouse
            #Si llega antes de tiempo del nodo i-1 al nodo i. el retorno a wharehouse es
            return_time_t1: float = a[chromosome[idx]] + service_time[chromosome[idx]] + Time()(dist_to_wharehouse)
            #Si llega dentor de la ventana de tiempo del nodo i-1 al nodo i. el retorno a wharehouse es
            return_time_t2: float = arrival_t + service_time[chromosome[idx]] + Time()(dist_to_wharehouse)
            # tiene que tener suficiente combustible para retornar a wharehouse
            fuel_exp = fuel_expen_t + fuel_expen_wh


            # posibles condiciones

            earlyArrivalCond = (
                arrival_t < a[chromosome[idx]], # la llega debe ser antes de la ventana de tiempo
                q_n <= Q_MAX[vehicle_n], # cubrir la demanda total no debe superar la capacidad máxima
                fuel_exp <= T[vehicle_n], # el uso de combustible total más el de regreso no debe superar la capacidad del tanque
                return_time_t1 <= b[0] # el tiempo total más del regreso no debe superar la ventana de tiempo
            )

            twArrivalCond = (
                a[chromosome[idx]] <= arrival_t <= b[chromosome[idx]], # la llega debe ser dentro de la ventana de tiempo
                q_n <= Q_MAX[vehicle_n], # cubrir la demanda total no debe superar la capacidad máxima
                fuel_exp <= T[vehicle_n], # el uso de combustible total más el de regreso no debe superar la capacidad del tanque
                return_time_t2 <= b[0]  # el tiempo total más del regreso no debe superar la ventana de tiempo
            )

            noArrivalCond = (
                arrival_t > b[chromosome[idx]], # supera la ventana de tiempo
                q_n > Q_MAX[vehicle_n], # cubrir supera la capacidad de carga 
                return_time_t1 > b[0], #no puede regresar a tiempo si llega temprano
                return_time_t2 > b[0], #no puede regresar a tiempo si llega tarde
                fuel_exp  > T[vehicle_n] #no alcanza combustible para atender y regresar
            )

            if all(earlyArrivalCond):
                # almacenamos viaje
                r[vehicle_n].append([chromosome[idx], vehicle_n])
                leave_t = a[chromosome[idx]] + service_time[chromosome[idx]]
     
                if idx == 0 or change:
                    infoData['Nodoi-1'].append(0)
                    infoData['Nodoi'].append(chromosome[idx])
                    change = False
                else:
                    infoData['Nodoi-1'].append(chromosome[idx-1])
                    infoData['Nodoi'].append(chromosome[idx])

                infoData['Vehiculo'].append(vehicle_n)
                infoData['Distancia'].append(dist_t) 
                infoData['TiempoArco'].append(time_t)   
                infoData['TW_A'].append(a[chromosome[idx]])
                infoData['TW_B'].append(b[chromosome[idx]])
                infoData['Tanque'].append(T[vehicle_n] -  fuel_expen_t)
                infoData['CO2'].append(co2_gen)

            # arrival in tw
            elif all(twArrivalCond):
                r[vehicle_n].append([chromosome[idx], vehicle_n])
                leave_t = arrival_t + service_time[chromosome[idx]]

                if idx == 0 or change:
                    infoData['Nodoi-1'].append(0)
                    infoData['Nodoi'].append(chromosome[idx])
                    change = False
                else:
                    infoData['Nodoi-1'].append(chromosome[idx-1])
                    infoData['Nodoi'].append(chromosome[idx])

                infoData['Vehiculo'].append(vehicle_n)
                infoData['Distancia'].append(dist_t) 
                infoData['TiempoArco'].append(time_t)   
                infoData['TW_A'].append(a[chromosome[idx]])
                infoData['TW_B'].append(b[chromosome[idx]])
                infoData['Tanque'].append(T[vehicle_n] - fuel_expen_t)
                infoData['CO2'].append(co2_gen)
            

            elif any(noArrivalCond):
                # el nodo no puede ser servido por el vehiculo entonces debe ir a warehouse
                r[vehicle_n].append([0, vehicle_n]) 
                dist_t: float = Distance()(coords[chromosome[idx-1]], coords[0])
                time_t: float = Time()(dist_t)
                co2_gen_wh = Co2Generation()(vehicle_n, dist_t)
                fuel_expen_t -= co2_gen
                fuel_expen_t += co2_gen_wh


                #### guardar información #######################
                infoData['Nodoi-1'].append(chromosome[idx-1])
                infoData['Nodoi'].append(0)

                infoData['Vehiculo'].append(vehicle_n)
                infoData['Distancia'].append(dist_t) 
                infoData['TiempoArco'].append(time_t)  

                infoData['TW_A'].append(a[0])
                infoData['TW_B'].append(b[0])

                infoData['Tanque'].append(T[vehicle_n] - fuel_expen_t)
                infoData['CO2'].append(co2_gen_wh)


                # correguir
                # lo debe atender otro vehiculo
                vehicle_n += 1 

                if vehicle_n >= excessVehicle:
                    r = dict()
                    return r , pd.DataFrame(infoData)

                r[vehicle_n] = [[0, vehicle_n], ] 
                dist_t: float = 0
                time_t: float = Time()(dist_t)
                co2_gen = Co2Generation()(vehicle_n, dist_t)
                fuel_expen_t = 0
                q_n = 0
                leave_t = a[0]
                change = True

                continue
     
            idx += 1

        if r[vehicle_n][-1][0] != 0:
            r[vehicle_n].append([0, vehicle_n])

            dist_t: float = Distance()(coords[chromosome[idx-1]], coords[0])
            time_t: float = Time()(dist_t)
            co2_gen_wh = Co2Generation()(vehicle_n, dist_t)
            #fuel_expen_t -= co2_gen
            fuel_expen_t += co2_gen_wh


            #### guardar información #######################
            infoData['Nodoi-1'].append(chromosome[idx-1])
            infoData['Nodoi'].append(0)

            infoData['Vehiculo'].append(vehicle_n)
            infoData['Distancia'].append(dist_t) 
            infoData['TiempoArco'].append(time_t)  

            infoData['TW_A'].append(a[0])
            infoData['TW_B'].append(b[0])

            infoData['Tanque'].append(T[vehicle_n] - fuel_expen_t)
            infoData['CO2'].append(co2_gen_wh)

        if summary:
            summdf= pd.DataFrame(infoData)
            return r, summdf
        return r


    def fitness_eval(self, pop_objs_fun: np.ndarray):
        evals_arr = np.zeros(shape=(pop_objs_fun.shape[0], 4))
        performances = np.argsort(pop_objs_fun)
        for i , chrom in enumerate(performances):
            q: int =  i # posicion en ranking
            p: int = (len(performances) - 1) - i # numero de individuos que domina
            evals_arr[i][0] = i  # hierarchy / rank
            evals_arr[i][1] = chrom # reference chromosome
            evals_arr[i][2] = PddrffValue()(q, p) # fitness value
            evals_arr[i][3] = 1 - (i/evals_arr.shape[0]) # rank distance
        return evals_arr
    

    def selection_rank_elite(self, pop_fit_evals: np.ndarray,  ElitePopSize: int,  subPopSize: int) -> np.ndarray:
        mating_pool = np.zeros(shape=(ElitePopSize + subPopSize, ))
        #initializing elite population At
        mating_pool[0:ElitePopSize] = pop_fit_evals[0:ElitePopSize, 1]  #selecciona lo indices que ref al  chromosoma en la poblacion    

        shuffled_pop_fit_evals = np.copy(pop_fit_evals[ElitePopSize:])
        #np.random.shuffle(shuffled_pop_fit_evals)

        i: int = ElitePopSize # compiten a partir del ultimo seleccionado como elite
        j: int = 0 # cuantos van seleccionandose del torneo binario

        
        while j < subPopSize:
            if j == subPopSize:
                break
            #a = np.random.randint(0, subPopSize)
            #b = np.random.randint(0, subPopSize)
            a = np.random.randint(0, shuffled_pop_fit_evals)
            b = np.random.randint(0, shuffled_pop_fit_evals)
            opponent1, opponent2 = shuffled_pop_fit_evals[a], shuffled_pop_fit_evals[b]
            mating_pool[ElitePopSize + j] = self.bin_tournament_selection(x1=opponent1, x2=opponent2)[1]
            j += 1 # agregó 1 ganador
        return mating_pool.astype(int)
    
    def selection_rank_elite2(self, population: np.ndarray, pop_fit_evals: np.ndarray,  ElitePopSize: int,  subPopSize: int) -> np.ndarray:

        mating_pool = np.zeros(shape=(ElitePopSize + subPopSize, population.shape[1]))
        #initializing elite population At
        
        mating_pool[0:ElitePopSize] = np.take(population, pop_fit_evals[0:ElitePopSize, 1].astype(int), axis=0)


        shuffled_pop_fit_evals = np.take(population, pop_fit_evals[ElitePopSize:,1].astype(int), axis=0)
        #shuffled_pop_fit_evals = np.copy(pop_fit_evals[ElitePopSize:])
        #np.random.shuffle(shuffled_pop_fit_evals)

        i: int = ElitePopSize # compiten a partir del ultimo seleccionado como elite
        j: int = 0 # cuantos van seleccionandose del torneo binario

        while j < subPopSize:
            if j == subPopSize:
                break
            a = np.random.randint(0, shuffled_pop_fit_evals.shape[0])
            b = np.random.randint(0, shuffled_pop_fit_evals.shape[0])
            opponent1, opponent2 = shuffled_pop_fit_evals[a], shuffled_pop_fit_evals[b]

            mating_pool[ElitePopSize + j] = self.bin_tournament_selection(x1=opponent1, x2=opponent2)
            j += 1 
        return mating_pool

    def bin_tournament_selection(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.objective_function(x1) < self.objective_function(x2):
            return x1
        return x2
        
    def differentialExchangeSequence(self,  s1: np.ndarray, s2: np.ndarray) -> List[Tuple[int]]:
        s = s2.copy()
        exchangeSequence: List[Tuple[int]] = []
        for i in range(0, s1.shape[0]):
            if s1[i] == s2[i]:
                continue
            for j in range(i+1, s1.shape[0]):
                if s1[i] != s2[j]:
                    continue
                exchangeSequence.append((i,j))
                s2[j], s2[i] = s2[i], s2[j] 
                break
        max_L: int =  len(exchangeSequence)
        u =  np.random.uniform(0,1, size=1)
        L: int =  int(np.floor( u *  max_L)[0])
        s2 = s.copy()
        for k, (i,j) in enumerate(exchangeSequence):
            if k == L:
                break
            s2[j], s2[i] = s2[i], s2[j] 
        return s2
    
    def diffPopulationImprovement(self, best: np.ndarray, population: np.ndarray) -> np.ndarray:
        popCopy = population.copy()
        for i in range(0, population.shape[0]):
            popCopy[i] = self.differentialExchangeSequence(best,popCopy[i])
            if self.objective_function(popCopy[i]) < self.objective_function(population[i]):
                population[i] = popCopy[i]
        return population
            

    def bestIndividual(self, population: np.ndarray) -> np.ndarray:
        objects_values = [self.objective_function(individual) for individual in population]
        best_idx = np.argmin(objects_values)
        best = population[best_idx]
        return best.copy()

    def lambdaInterchange(self, chrom: np.ndarray, lambdaValue: int=2) -> np.ndarray:
        chrom = chrom.astype(int)
        best: float = self.objective_function(chrom)
        decode: dict = self.decoding(chrom) 
        n_routes: int = len(decode)

        x, y = np.random.choice(range(n_routes), size=2, replace=False)

        i_index = np.random.choice(range(0,len(decode[x])-1), size=1)[0]
        j_index = np.random.choice(range(0,len(decode[y])-1), size=1)[0]

        decode[x][i_index], decode[y][j_index] = decode[y][j_index], decode[x][i_index]
        
        new_chrom = self.encoding(decode)
        new_obj = self.objective_function(new_chrom)

        if new_obj < best:
            return new_chrom
        return chrom
    
    def swapOperator(self, chromosome_i: np.ndarray):
        chromosome = chromosome_i.copy()
        a = np.random.randint(0, chromosome.shape[0] - 1)
        b = np.random.randint(0, chromosome.shape[0] - 1)

        while True:
            b = np.random.randint(0, chromosome.shape[0] - 1)
            if a != b:
                break

        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]

        return chromosome

    def swap2Operator(self, chrom: np.ndarray):
        chrom = chrom.astype(int)
        decode: dict = self.decoding(chrom) 
        n_routes: int = len(decode)

        x, y = np.random.choice(range(n_routes), size=2, replace=False)

        i_index = np.random.choice(range(0,len(decode[x])-1), size=1)[0]
        j_index = np.random.choice(range(0,len(decode[y])-1), size=1)[0]

        decode[x][i_index], decode[y][j_index] = decode[y][j_index], decode[x][i_index]
        
        new_chrom = self.encoding(decode)
        return new_chrom

    def swapOperator(self, chromosome_i: np.ndarray):
        chromosome = chromosome_i.copy()
        a = np.random.randint(0, chromosome.shape[0])
        b = np.random.randint(0, chromosome.shape[0])

        while True:
            b = np.random.randint(0, chromosome.shape[0])
            if a != b:
                break

        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]

        return chromosome
    
    def scrambledSubstring(self, chromosome_i: np.ndarray):
        chromosome = chromosome_i.copy()
        a = np.random.randint(0, chromosome.shape[0])
        b = np.random.randint(0, chromosome.shape[0])

        while True:
            b = np.random.randint(0, chromosome.shape[0])
            if a != b:
                break
        x = chromosome[a:b].copy()        
        np.random.shuffle(x)
        chromosome[a:b] = x
        return chromosome
    
    def insertion(self, chromosome_i: np.ndarray):
        x = chromosome_i.copy()
        n = np.random.randint(1, x.shape[0]-2)
        m = np.random.randint(1, x.shape[0])

        while True:
            m = np.random.randint(0, x.shape[0])
            if  m > n:
                break
        return np.concatenate((x[:n],x[n+1:m],np.array([x[n]]),x[m:]))
    
    def reverseSubstring(self, chromosome_i: np.ndarray):
        x = chromosome_i.copy()
        n = np.random.randint(1, x.shape[0]-2)
        m = np.random.randint(1, x.shape[0])

        while True:
            m = np.random.randint(n+1, x.shape[0])
            if  m > n:
                break
        return np.concatenate((x[:n],x[m:n-1:-1] ,x[m+1:]))

    def localSearch(self, Chrom: np.ndarray, x_best: np.ndarray = np.array([])):
        x1 = self.swap2Operator(Chrom)
        x2 = self.swapOperator(Chrom)
        x3 = self.scrambledSubstring(Chrom)
        x4 = self.insertion(Chrom)
        x5 = self.reverseSubstring(Chrom)
        x6 = self.mutation_operator(Chrom)

        if x_best.any():
            x7 =self.differentialExchangeSequence(x_best, Chrom)
            operations = [
                x1,
                x2,
                x3,
                x4,
                x5,
                x6,
                self.differentialExchangeSequence(x_best, x1),
                self.differentialExchangeSequence(x_best, x2),
                self.differentialExchangeSequence(x_best, x3),
                self.differentialExchangeSequence(x_best, x4),
                self.differentialExchangeSequence(x_best, x5),
                self.differentialExchangeSequence(x_best, x6),
                self.differentialExchangeSequence(x_best, x7),
            ]
        else:
            operations = [
                x1,
                x2,
                x3,
                x4,
                x5,
                x6,
            ]
        neighborhood = np.zeros(shape=(len(operations), Chrom.shape[0]))
        for i in range(0, len(operations)):
            neighborhood[i] = operations[i]
        return self.bestIndividual(neighborhood)


    def initialGeneration(self, PopSize: int, Chrom: np.ndarray):   
        population = np.zeros(shape=(PopSize, Chrom.shape[0])) 
        for i in range(0, PopSize):
            population[i] = self.create_chromosome(Chrom.shape[0])
            population[i] = self.lambdaInterchange(Chrom)
            if self.objective_function(population[i]) > self.objective_function(Chrom):
                population[i] = self.differentialExchangeSequence(Chrom, population[i])
        return population
    

    #algorithms
    def ga(
            self, 
            x0: np.ndarray, # solucion inicial
            PopSize: int, 
            ElitePopSize: int=0, 
            subPopSize: int=0, 
            MaxGenerations: int=0,
            localOptimumTime: int = 0
        ) -> dict:
        hist: list[tuple] = []
        t: int = 0
        x_best = x0
        e_best = self.objective_function(x_best)
        ############### poblacion inicial ###############
        population_t = self.initialGeneration(PopSize, x0)
        ############### poblacion inicial ###############
        k_converge = 0

        while t <= MaxGenerations and  k_converge < localOptimumTime:

            pop_objs_fun = self.population_objs_fun(population_t) #obtener fitness array
            pop_fit_evals = self.fitness_eval(pop_objs_fun)

            mating_pool = self.selection_rank_elite2(
                population= population_t,
                pop_fit_evals= pop_fit_evals, 
                ElitePopSize= ElitePopSize,
                subPopSize= subPopSize
            )

            offspring1 = self.crossover(mating_pool, PopSize, ElitePopSize, CrossPro=0.9)
            offspring1 = self.mutation(offspring1, PopSize, ElitePopSize, MutPro=0.2)
            x_current = self.bestIndividual(offspring1)

            if self.objective_function(x_current) < self.objective_function(x_best):
                x_best = x_current.copy()
                e_best = self.objective_function(x_current)
                k_converge = 0
            
            x_neighbor = self.localSearch(x_current, x_best)

            if self.objective_function(x_neighbor) < self.objective_function(x_best):
                x_best = x_neighbor.copy()
                e_best = self.objective_function(x_neighbor)
                k_converge = 0

            population_t = offspring1.copy()
            population_t = self.diffPopulationImprovement(x_best, population_t)
            t += 1
            k_converge += 1
            hist.append((t, e_best))
            print(f'generation: {t}')

        return {
            'x_best': x_best.astype(int),
            'e_best': e_best,
            'time': t,
            'hist': hist

        }
    def stopping_restrictions(
            self, 
            step: int,
            t: float,  
            StepMax: int= 100,
            TMin: float= 0, 
            TMax: float= 100
        ):
        return step <= StepMax and  TMax >= t >= TMin


    def vegsa(
            self, 
            x0: np.ndarray, # solucion inicial
            PopSize: int, 
            ElitePopSize: int=0, 
            subPopSize: int=0, 
            MaxGenerations: int=0,
            localOptimumTime: int = 0,
            StepMax: int = 100, 
            Tmin: float = 0, 
            Tmax: float = 100,
            SAlocalEntropyTime: int = 0,
            cooling_operator: sa_cooling_operator = linear_additive_cooling
        ) -> dict:
        
        print('GA init: ')
        ################ GA ######################
        information = self.ga(
            x0=x0, # solucion inicial
            PopSize=PopSize, 
            ElitePopSize=ElitePopSize, 
            subPopSize=subPopSize, 
            MaxGenerations= MaxGenerations,
            localOptimumTime= localOptimumTime
        )
        ##########################################
        print('GA result: ')
        print(information)
        print('SA init: ')
        x_best: np.ndarray = information['x_best']
        e_best: float = information['e_best']
        x_current: np.ndarray = x_best.copy()
        e_current: float = e_best

        step = 1
        t = Tmax
        hist: list[tuple] = []
        k_converge: int = 0

        while self.stopping_restrictions(
                step= step,
                t= t,
                StepMax= StepMax,
                TMin=Tmin,
                TMax=Tmax
            ) and k_converge <= SAlocalEntropyTime:
            x_neighbor = self.localSearch(Chrom=x_current)
            e_neighbor = self.objective_function(chromosome=x_neighbor)
            e_delta = e_neighbor - e_current
            if random() < self.safe_exp(- e_delta / t):
                x_current = x_neighbor.copy()
                e_current = e_neighbor
                k_converge = 0
            if e_current <= e_best:
                x_best = x_current.copy()
                e_best = e_current
                k_converge = 0
                
            
            hist.append((step, e_best, t)) #UPDATE
            print(f'step: {step}, temperature: {t}, energy: {e_best}')
            #add cooling operator
            t = cooling_operator(step=step)
            k_converge += 1
            step += 1
            

        sa_information = {
            'x_best': x_best.astype(int),
            'e_best': e_best,
            'temperature': t,
            'time': step,
            'hist': hist

        }
        print('SA result: ')
        print(sa_information)
        return sa_information
    
    def safe_exp(self, x: int | float):
        try:
            return exp(x)
        except:
            return 0

    def draw_energy_plot(self, hist: list[tuple]):
        steps = [entry[0] for entry in hist]
        energy_values = [entry[1] for entry in hist]
        fig = px.line(
            x=steps, 
            y=energy_values, labels={'x': 'Step', 'y': 'Best Energy Value'},
            title='Optimization Progress', markers=True, line_shape='linear'
        )
        fig.show()





    