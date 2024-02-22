from models.model import Model
from src.variables import a, coords
from src.functions import Distance, Time, Co2Generation, TransportCost
from collections import deque
import numpy as np
from typing import List, Tuple

# x = np.array([1,2,3,4,5,6])
# for i, j in zip(x[::2], x[1::2]):
#     print(i,',',j)


# # print(x)

# # a = np.random.randint(0, x.shape[0] - 1)
# # b = np.random.randint(0, x.shape[0] - 1)
# # while True:
# #     b = np.random.randint(0, x.shape[0])
# #     if (a != b) and (a+1 != b) and (a != b+1) and (a+1 != b+1):
# #         break
# # x[a], x[b] = x[b], x[a]
# # x[a+1], x[b+1] = x[b+1], x[a+1]

# # print(a)
# # print(b)
# # print(x)



x = Model()
x.evolutionary_search(
    PopSize=1000,
    ChroSize= 11,
    ElitePopSize= 10,
    subPopSize= 20,
    MaxGenerations= 100
)


# print('##################TEST###############')

# c1 = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9],
# ])

# c2 = np.array([
#     [1,1,1],
#     [2,2,2],
#     [3,3,3],
# ])

# c1[0:2] = c2[0:2]
# print(c1)









# c1 = np.array([1,2,3,4,5,6,7,8,9])
# c2 = np.array([5,4,6,9,2,1,7,8,3])

# def crossover_operator(chromosome1: np.ndarray, chromosome2: np.ndarray):
#     a = np.random.randint(0, chromosome1.shape[0] - 1)
#     b = np.random.randint(0, chromosome1.shape[0])
#     while True:
#         b = np.random.randint(0, chromosome1.shape[0])
#         if b > a:
#             break
    
#     offspring1 = np.zeros(shape=chromosome1.shape)
#     offspring2 = np.zeros(shape=chromosome1.shape)

#     # offspring 1
#     swath = chromosome1[a: b + 1]
#     print(swath)
#     offspring1[a: b + 1] = swath
#     idx_swath = np.isin(chromosome2, swath)
#     alleles_c2 = deque()

#     for select, allele in zip(idx_swath, chromosome2):
#         if not select:
#             alleles_c2.append(allele)
#     offspring = offspring1.copy()

#     for i, allele in enumerate(offspring):
#         if allele == 0:
#             offspring1[i] = alleles_c2.popleft()

#     # offspring 2
#     swath = chromosome2[a: b + 1]
#     offspring2[a: b + 1] = swath
#     idx_swath = np.isin(chromosome1, swath)
#     alleles_c1 = deque()

#     for select, allele in zip(idx_swath, chromosome1):
#         if not select:
#             alleles_c1.append(allele)
#     offspring = offspring2.copy()

#     for i, allele in enumerate(offspring):
#         if allele == 0:
#             offspring2[i] = alleles_c1.popleft()

#     return offspring1, offspring2


# print(crossover_operator(c1,c2))
# n: int = 5
# chromosome = np.arange(0,n)
# print(chromosome)
# np.random.shuffle(chromosome)
# print(chromosome)


# x: dict = {0: [[0, 0], [1, 0], [2, 0], [0, 0]], 1: [[0, 1], [3, 1], [4, 1], [5, 1], [0, 1]], 2: [[0, 2], [6, 2], [7, 2], [8, 2], [9, 2], [10, 2], [0, 2]]}

# J: float = 0
# for vehicle, route in x.items(): 
#     for node in route:
#         vertex: int = node[0]
#         if (vertex - 1) == -1:
#             continue
#         dist: float = Distance()(i=coords[vertex-1], j=coords[vertex])
#         co2_measure: float =  Co2Generation()(vehicle=vehicle, dist=dist)
#         trans_cost: float = TransportCost()(dist)
#         J += (trans_cost + co2_measure)
#     print(f'cost {vehicle} = {J}')
# print(f'Total cost = {J}')

# n = 5
# chromosome = np.arange(1, n+1)
# np.random.shuffle(chromosome)
# print(chromosome)



# def differentialExchangeSequence( s1: np.ndarray, s2: np.ndarray) -> List[Tuple[int]]:
#     s = s2.copy()
#     exchangeSequence: List[Tuple[int]] = []
#     for i in range(0, s1.shape[0]):
#         if s1[i] == s2[i]:
#             continue
#         for j in range(i+1, s1.shape[0]):
#             if s1[i] != s2[j]:
#                 continue
#             exchangeSequence.append((i,j))
#             s2[j], s2[i] = s2[i], s2[j] #swap
#             break

#     max_L: int =  len(exchangeSequence)
#     u =  np.random.uniform(0,1, size=1)
#     L: int =  int(np.floor( u *  max_L)[0])
#     s2 = s.copy()
#     for k, (i,j) in enumerate(exchangeSequence):
#         if k == L:
#             break
#         print(i,j)
#         s2[j], s2[i] = s2[i], s2[j] #swap
#     return s2


# print(differentialExchangeSequence(s1=np.array([1,2,3,4,5,6]), s2=np.array([3,2,4,5,1,6])))




# def selection_rank_with_elite(self, pop_fit_evals: np.ndarray, ElitePopSize: int=0, subPopSize: int=0) -> np.ndarray:
#     pass

