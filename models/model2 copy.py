from copy import deepcopy
from min_max_heap import MinMaxHeap
from queue import PriorityQueue
from typing import Any, List, Tuple
from collections import deque
import numpy as np
import pandas as pd
from math import exp, log, sqrt
import matplotlib.pyplot as plt
import plotly.express as px
from abc import abstractmethod, ABC
from random import random
from colorama import init, Fore, Style


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
    

class x_sa(ABC):
    def __init__(self, ) -> None:
        super().__init__()
    
    @abstractmethod
    def cost_function(self,) -> float:
        pass

    @abstractmethod
    def generate_neighbor(self, ) -> 'x_sa':
        pass

    @abstractmethod
    def check_restrictions(self,) -> bool:
        pass
    @abstractmethod   
    def __str__(self) -> str:
        pass

    @abstractmethod
    def get_information() -> Any:
        pass


class CES:

    def __init__(self):
        pass

    def proximity(self, waitTime: float, urgencyTime: float,  timeRemaining: float, alpha: float=0.33, beta: float= 0.34, gamma: float=0.33) -> float:
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






    # def __init__(self, 
    #         x0: np.ndarray,  
    #         cooling_operator: sa_cooling_operator,
    #         step_max: int = 100, 
    #         t_min: float = 0, 
    #         t_max: float = 100,
    #     ) -> None:

    #     self.t: float = t_max
    #     self.t_max: float = t_max
    #     self.t_min : float  = t_min
    #     self.step_max: int = step_max
    #     self.hist: list[Any] = []
    #     self.cooling_operator: sa_cooling_operator = cooling_operator
    #     self._x_current: x_sa = x0
    #     self._e_current: float = self.objective_function(x0)
    #     self._x_best: x_sa = self._x_current
    #     self._e_best: float = self._e_current
    #     self.step: int = 0
    #     self.accept: int = 0
    #     self._band: bool = False

    
    # def stopping_restrictions(self, step):
    #     return step < self.step_max and self.t >= self.t_min and self.t > 0
    
    # def generate_neighbor(self, x: np.ndarray):
    #     pass
    
    # def solve(self,) -> None:
    #     self.step = 1 
    #     self.accept = 0
    #     while self.stopping_restrictions(self.step):
            
    #         x_neighbor = self.generate_neighbor(self._x_current) # conseguimos un vecino
    #         e_neighbor = self.objective_function(x_neighbor)
    #         e_delta =  e_neighbor - self._e_current # calculamos la diff de energia del vecino con la actual

    #         if random() < self.safe_exp(- e_delta / self.t): 
    #             self._x_current = x_neighbor 
    #             self._e_current = e_neighbor 
    #             self.accept += 1 

    #         if self._e_current < self._e_best: 
    #             self._e_best = self.e_current 
    #             self._x_best = self.x_current

    #         self.update_history()
    #         self.t = self.cooling_operator(self.step) #enfriamos
    #         self.step += 1
    #     self._band = True


    # @abstractmethod
    # def results(self):
    #     if self._band:
    #         init(autoreset=True)
    #         print(Fore.YELLOW + 'results: ')
    #         print(Fore.CYAN + '{')
    #         print(f'{Fore.MAGENTA}\tcost:{Style.RESET_ALL} {Fore.GREEN} {self._e_best}')
    #         print(f'{Fore.YELLOW}\tinitial_temp:{Style.RESET_ALL} {Fore.GREEN}{self.t_max}')
    #         print(f'{Fore.YELLOW}\tfinal_temp:{Style.RESET_ALL} {Fore.GREEN}{self.t}')
    #         print(f'{Fore.LIGHTRED_EX}\tmax_steps:{Style.RESET_ALL} {Fore.GREEN}{self.step_max}')
    #         print(f'{Fore.LIGHTGREEN_EX}\tfinal_step:{Style.RESET_ALL} {Fore.GREEN}{self.step}')
    #         print(f'{Fore.YELLOW}\tfinal_energy:{Style.RESET_ALL} {Fore.GREEN}{self._e_best}')
    #         print(Fore.CYAN + '}')
    #     else:
    #         print('musts first execute the solve method.')


    # @property
    # def x_current(self):
    #     return self._x_current

    # @property
    # def e_current(self):
    #     return self._e_current

    # @property
    # def x_best(self):
    #     return self._x_best

    # @x_best.setter
    # def x_best(self, value):
    #     self._x_best = value

    # @property
    # def e_best(self):
    #     return self._e_best

    # @e_best.setter
    # def e_best(self, value):
    #     self._e_best = value



    # def update_history(self,):
    #     self.hist.append(
    #         {
    #             'step': self.step,
    #             'temperature': self.t,
    #             'e_best': self._e_best,
    #             'x_best': self._x_best
    #         }
    #     )

    # def draw_energy_plot(self):
        
    #     steps = [entry['step'] for entry in self.hist]
    #     energy_values = [entry['e_best'] for entry in self.hist]

    #     fig = px.line(x=steps, y=energy_values, labels={'x': 'Step', 'y': 'Best Energy Value'},
    #                   title='Optimization Progress', markers=True, line_shape='linear')

    #     fig.show()
        
            
    # def safe_exp(self, x):
    #     try: 
    #         return exp(x)
    #     except: 
    #         return 0





    def initial_population(self, PopSize: int, ChroSize: int) -> np.ndarray:
        population = np.zeros(shape=(PopSize, ChroSize))
        for i in range(0, PopSize):
            population[i] = self.create_chromosome(n=ChroSize)
        return population

    def create_chromosome(self, n: int) -> np.ndarray:
        chromosome = np.arange(1, n+1)
        np.random.shuffle(chromosome)
        return chromosome

    def encoding(self):
        return N[1:-1].copy() #only vertex, no wharehouse
    

    def mutation_operator(self, chromosome_i: np.ndarray):
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
        
        j: int = ElitePopSize #mantenemos la poblacion elite
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
        #initializing elite population At
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
                #print(child1)
                offspring[j] = child1
                offspring[j+1] = child2
            else:
                #print(matingPool[a])
                # offspring[j] = matingPool[a]
                # offspring[j+1] = matingPool[b]
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
                infoData['Tanque'].append(fuel_expen_t)
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
                infoData['Tanque'].append(fuel_expen_t)
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

                infoData['Tanque'].append(fuel_expen_t)
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

            infoData['Tanque'].append(fuel_expen_t)
            infoData['CO2'].append(co2_gen_wh)

        if summary:
            summdf= pd.DataFrame(infoData)
            return r, summdf
        return r

    def objective_function(self, chromosome: np.ndarray) -> float:
        
        r , df = self.decoding(chromosome.astype(int), summary=True) 
        if len(r) == 0:
            return float('inf')
        co2: float = df['CO2'].sum()
        tcost: float = TransportCost()(df['Distancia'].sum())
        return  co2 + tcost
    
    def population_objs_fun(self, chroPopulation: np.ndarray) -> np.ndarray:
        popSize = chroPopulation.shape[0]
        eval_array = np.zeros(shape=(popSize, )) #MULTI-OBJETIVE
        for i in range(0, popSize):
            eval_array[i] = self.objective_function(chroPopulation[i])
        return eval_array
    
    def fitness_eval(self, pop_objs_fun: np.ndarray):
        evals_arr = np.zeros(shape=(pop_objs_fun.shape[0], 4))
        performances = np.argsort(pop_objs_fun)
        for i , chrom in enumerate(performances):
            q: int = 0 + i
            p: int = (len(performances) - 1) - i # 
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

        if self.objective_function(x1) > self.objective_function(x2):
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
                s2[j], s2[i] = s2[i], s2[j] #swap
                break

        max_L: int =  len(exchangeSequence)
        u =  np.random.uniform(0,1, size=1)
        L: int =  int(np.floor( u *  max_L)[0])
        s2 = s.copy()
        for k, (i,j) in enumerate(exchangeSequence):
            if k == L:
                break
            #print(i,j)
            s2[j], s2[i] = s2[i], s2[j] #swap
        return s2
            
    def simpleInsertionSearch(self, chromosome: np.ndarray):
        pass


    # def evolutionary_search(self, PopSize: int, ChroSize: int, ElitePopSize: int=0, subPopSize: int=0, MaxGenerations: int=0):
     
    #     t: int = 0
        
    #     population_t = self.initial_population(PopSize, ChroSize)
    #     # chooseIndex = np.random.randint(0, population_t.shape[0])
    #     # bestChrom = population_t[chooseIndex].copy()

    #     while t <= MaxGenerations:

    #         pop_objs_fun = self.population_objs_fun(population_t)
    #         pop_fit_evals = self.fitness_eval(pop_objs_fun)
    #         selected_mating_pool = self.selection_rank_elite2(
    #             pop_fit_evals= pop_fit_evals, 
    #             ElitePopSize= ElitePopSize,
    #             subPopSize= subPopSize
    #         )

    #         mating_pool = np.take( population_t,selected_mating_pool,axis=0)
    #         offspring1 = self.crossover(mating_pool, PopSize, ElitePopSize, CrossPro=0.9)
    #         offspring1 = self.mutation(offspring1, PopSize, ElitePopSize, MutPro=0.2)
    #         population_t = offspring1.copy()

    #         # minIndex = np.argmin([self.objective_function(chro) for chro in population_t])

    #         # if self.objective_function(population_t[minIndex]) < self.objective_function(bestChrom):
    #         #     bestChrom = population_t[minIndex].copy()
    #         print(f'tiempo {t}')
    #         print(population_t[0:20])
    #         t += 1

    #     #print(bestChrom)
    #     print(population_t[0:30])

    def climatic_evolutionary_search(self, PopSize: int, ChroSize: int, ElitePopSize: int=0, subPopSize: int=0, MaxGenerations: int=0):
     
        t: int = 0
        
        population_t = self.initial_population(PopSize, ChroSize)
        # chooseIndex = np.random.randint(0, population_t.shape[0])
        # bestChrom = population_t[chooseIndex].copy()

        while t <= MaxGenerations:

            pop_objs_fun = self.population_objs_fun(population_t)
            pop_fit_evals = self.fitness_eval(pop_objs_fun)
            mating_pool = self.selection_rank_elite2(
                population= population_t,
                pop_fit_evals= pop_fit_evals, 
                ElitePopSize= ElitePopSize,
                subPopSize= subPopSize
            )
            print('time: ', t)
            print(mating_pool)
            offspring1 = self.crossover(mating_pool, PopSize, ElitePopSize, CrossPro=0.9)
            offspring1 = self.mutation(offspring1, PopSize, ElitePopSize, MutPro=0.2)
            population_t = offspring1.copy()


            t += 1





    

    def generate_neighbor(self):
        return deepcopy(self.patient_exchange())
    
    #@no_test
    def check_restrictions(self) -> bool:
        print('')
        return True
    
    def get_information(self, ) -> Any:
        return self.get_output()


    def __getitem__(self, key: int):
        return  
       

    def __str__(self) -> str:
        pass


    def sort_by_ready_time(self,):
        pass

    def find_inital_solution(self):
       pass

    def patient_exchange(self):
       pass
        

    def get_neighbour(self): #listo
        pass
    

    def get_output(self, ) -> dict:
        pass


    def all_attended(self, ) -> bool: 
        pass


    
    def get_total_cost_and_vehicles(self, ):
        pass

