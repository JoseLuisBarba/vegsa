from copy import deepcopy
from min_max_heap import MinMaxHeap
from typing import Any, List, Tuple
from collections import deque
import numpy as np
import pandas as pd

from src.variables import N, K, coords, a, b, service_time, demands, Q_MAX, VD, T 
from src.functions import Distance, Time, Co2Generation, TransportCost, PddrffValue




class Model():
    def __init__(self, ) -> None:
        self.n = self.encoding()



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
    

    def mutation_operator(self, chromosome: np.ndarray):
        a = np.random.randint(0, chromosome.shape[0] - 1)
        b = np.random.randint(0, chromosome.shape[0] - 1)
        while True:
            b = np.random.randint(0, chromosome.shape[0] - 1)
            if (a != b) and (a+1 != b) and (a != b+1) and (a+1 != b+1):
                break
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
        chromosome[a+1], chromosome[b+1] = chromosome[b+1], chromosome[a+1]
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

        return offspring1, offspring2
    

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
                offspring[j] = matingPool[a]
                offspring[j+1] = matingPool[b]

            j += 2
        return offspring


    def decoding2(self, chromosome: np.ndarray):
        # colocar restriccion a n-vehiculos
        vehicle_n: int =  0 # current vehicle number
        q_n: float =  0  #current load of vehicle
        arrival_t: float = a[0] # is the arrival time of vertex t
        leave_t: float = a[0] + service_time[0] # is the leave time of vertex t
        #return_time_t: float # time from vertex t back to depot 0
        # add the depot as the beginning of route
        #r = [[vertex, route] ...]
        r: dict = dict()
        r[0] = [[0, vehicle_n], ]

        # add the first vertex in chromosome into first route 0
        q_n +=  demands[0]


        for i in chromosome:
            dist_t: float = Distance()(coords[i-1], coords[i])
            time_t: float = Time()(dist_t)
            arrival_t = leave_t + time_t
            q_n += demands[i]

            dist_to_wharehouse: float = Distance()(coords[i], coords[0])
            #for early arrival
            return_time_t1: float = a[i] + service_time[i] + Time()(dist_to_wharehouse)
            # for no early arrival
            return_time_t2: float = arrival_t + service_time[i] + Time()(dist_to_wharehouse)
            # early arrival
            if arrival_t < a[i] and q_n <= Q_MAX[vehicle_n] and return_time_t1 <= b[0]:
                # belongs to the current vehicle
                r[vehicle_n].append([i, vehicle_n])
                leave_t = a[i] + service_time[i]
            # arrival in tw
            elif a[i] <= arrival_t <= b[i] and q_n <= Q_MAX[vehicle_n] and  return_time_t2 <= b[0]:
                r[vehicle_n].append([i, vehicle_n])
                leave_t = arrival_t + service_time[i]

            elif arrival_t > b[i] or q_n > Q_MAX[vehicle_n] or  return_time_t1 > b[0] or return_time_t2 > b[0]:
                # i-vertex isn't served by the current vehicle
                r[vehicle_n].append([0, vehicle_n]) # add wharehouse as the ending of route
                vehicle_n += 1 # is served by the new vehicle
                r[vehicle_n] = [[0, vehicle_n], ] # add the depot as the beginning of route
                r[vehicle_n].append([i, vehicle_n])
                leave_t = a[i] + service_time[i]
                q_n = demands[i]
        if r[vehicle_n][-1][0] != 0:
            r[vehicle_n].append([0, vehicle_n])
        return r
    
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
        
        routes = self.decoding(chromosome.astype(int)) 
        J: float = 0
        #print(routes)
        for vehicle, route in routes.items(): 
    
            for i in range(0, len(route)):
                prev_vertex: int = route[i - 1][0]
                vertex: int = route[i][0]
                if i == 0:
                    continue
                dist: float = Distance()(i=coords[prev_vertex], j=coords[vertex])
                co2_measure: float =  Co2Generation()(vehicle=vehicle, dist=dist)
                trans_cost: float = TransportCost()(dist)
                J += (trans_cost + co2_measure)

                #print(prev_vertex, ': ', vertex, 'cost: ',J)
        #print(J)
        return  J
    
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
            a = np.random.randint(0, subPopSize)
            b = np.random.randint(0, subPopSize)
            opponent1, opponent2 = shuffled_pop_fit_evals[a], shuffled_pop_fit_evals[b]
            mating_pool[ElitePopSize + j] = self.bin_tournament_selection(x1=opponent1, x2=opponent2)[1]
            j += 1 # agregó 1 ganador
        return mating_pool.astype(int)

    def bin_tournament_selection(self, x1: np.ndarray, x2: np.ndarray):
        r1 = np.random.uniform(0,1,size=1)[0]
        r2 = np.random.uniform(0,1,size=1)[0]
        if x1[2] >= x2[2]:
            if r1 <= x1[3]:
                return x1
            if r2 <= x2[3]:
                return x2
            return x1
        else:
            if r2 <= x2[3]:
                return x2
            if r1 <= x1[3]:
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
            


    def evolutionary_search(self, PopSize: int, ChroSize: int, ElitePopSize: int=0, subPopSize: int=0, MaxGenerations: int=0):
     
        t: int = 0
        
        population_t = self.initial_population(PopSize, ChroSize)

        while t <= MaxGenerations:
            pop_objs_fun = self.population_objs_fun(population_t)
  
            pop_fit_evals = self.fitness_eval(pop_objs_fun)

            selected_mating_pool = self.selection_rank_elite(
                pop_fit_evals= pop_fit_evals, 
                ElitePopSize= ElitePopSize,
                subPopSize= subPopSize
            )
            # Obtener los valores de population_t cuyos índices están en mating_pool
            mating_pool = np.take( population_t,selected_mating_pool,axis=0)
            offspring1 = self.crossover(mating_pool, PopSize, ElitePopSize, CrossPro=0.9)
            #print('###### Crossover #######')
            #print(offspring1)
            offspring1 = self.mutation(offspring1, PopSize, ElitePopSize, MutPro=0.1)
            #print('###### Mutation #######')
            #print(offspring1)
            t += 1
        print(offspring1[0:ElitePopSize])




    

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

