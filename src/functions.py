from typing import Any
import numpy as np
from src.variables import CC, f, EM, Q_MAX, N, K, VD


class Distance:
    def __call__(self, i: np.ndarray, j: np.ndarray) -> float:
        if i.shape != j.shape or len(i.shape) != 1:
            raise ValueError("Ambos vectores deben ser unidimensionales y tener la misma longitud")

        return np.linalg.norm(i - j)
    

class Co2Generation:
    def __call__(self, vehicle: int , dist: float) -> float:
        if vehicle > K.shape[0]:
            raise ValueError("el vehiculo ingresado debe estar en el conjunto de vehiculos")
        return CC * (EM[vehicle] / Q_MAX[vehicle]) * dist



class TransportCost:
    def __call__(self, dist: float) -> float:
        return f * dist / 500
    

    
class Time:
    def __call__(self, dist: float) -> float:
        return dist / VD
    

class PddrffValue:
    def __call__(self, q_x: int, p_x: int) -> float:
        """_summary_

        Args:
            q_x (int): is the number of individuals dominating individual X
            p_x (int):  is the number of individuals dominated by individual X

        Returns:
            float: _description_
        """
        return  q_x + (1 / (p_x + 1))