from random import random
from math import exp
from math import log
from typing import Any
from abc import abstractmethod, ABC
from colorama import init, Fore, Style
from typing import Any
import matplotlib.pyplot as plt
import plotly.express as px

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



class simulated_annealing:
    def __init__(
            self, 
            x0: x_sa,  
            cooling_operator: sa_cooling_operator,
            step_max: int = 1000, 
            t_min: float = 0, 
            t_max: float = 100, 
        ) -> None:


        self.t: float = t_max
        self.t_max: float = t_max
        self.t_min : float  = t_min
        self.step_max: int = step_max
        self.hist: list[Any] = []
        self.cooling_operator: sa_cooling_operator = cooling_operator
        self._x_current: x_sa = x0
        self._e_current: float = x0.cost_function()
        self._x_best: x_sa = self._x_current
        self._e_best: float = self._e_current
        self.step: int = 0
        self.accept: int = 0
        self._band: bool = False
    


    def move(self,):
        pass

    def stopping_restrictions(self, step):
        return step < self.step_max and self.t >= self.t_min and self.t > 0



    def solve(self,) -> None:
        self.step = 1 
        self.accept = 0
        while self.stopping_restrictions(self.step):
            #print(f'step={self.step}, x_current {self.x_current} y x_best {self.x_best},  t= {self.t}')
            x_neighbor = self._x_current.generate_neighbor() # conseguimos un vecino
            e_neighbor = x_neighbor.cost_function() # medimos su energia
            e_delta =  e_neighbor - self._e_current # calculamos la diff de energia del vecino con la actual
            #print(f't={self.step}, x_neighbor {x_neighbor}')
            if random() < self.safe_exp(- e_delta / self.t): #  determinamos si aceptamos al vecino
                self._x_current = x_neighbor # si es aceptado entonces ahora es la solucion actual
                self._e_current = e_neighbor # tambien la energía
                self.accept += 1 

            if self._e_current < self._e_best: # si la energía es menor que la mejor solución
                self._e_best = self.e_current # asignamos los valores del x actual como el mejor
                self._x_best = self.x_current

            self.update_history()
            self.t = self.cooling_operator(self.step) #enfriamos
            #print(f'nueva t = {self.t}')
            self.step += 1
        self.acceptance_rate = self.accept / self.step
        self._band = True
            

    @abstractmethod
    def results(self):
        if self._band:
            init(autoreset=True)
            print(Fore.YELLOW + 'results: ')
            print(Fore.CYAN + '{')
            print(f'{Fore.MAGENTA}\tcost:{Style.RESET_ALL} {Fore.GREEN} {self._e_best}')
            print(f'{Fore.YELLOW}\tinitial_temp:{Style.RESET_ALL} {Fore.GREEN}{self.t_max}')
            print(f'{Fore.YELLOW}\tfinal_temp:{Style.RESET_ALL} {Fore.GREEN}{self.t}')
            print(f'{Fore.LIGHTRED_EX}\tmax_steps:{Style.RESET_ALL} {Fore.GREEN}{self.step_max}')
            print(f'{Fore.LIGHTGREEN_EX}\tfinal_step:{Style.RESET_ALL} {Fore.GREEN}{self.step}')
            print(f'{Fore.YELLOW}\tfinal_energy:{Style.RESET_ALL} {Fore.GREEN}{self._e_best}')
            print(Fore.CYAN + '}')
        else:
            print('musts first execute the solve method.')

    @property
    def x_current(self):
        return self._x_current

    @property
    def e_current(self):
        return self._e_current

    @property
    def x_best(self):
        return self._x_best

    @x_best.setter
    def x_best(self, value):
        self._x_best = value

    @property
    def e_best(self):
        return self._e_best

    @e_best.setter
    def e_best(self, value):
        self._e_best = value



    def update_history(self,):
        self.hist.append(
            {
                'step': self.step,
                'temperature': self.t,
                'e_best': self._e_best,
                'x_best': self._x_best
            }
        )

    def draw_energy_plot(self):
        
        steps = [entry['step'] for entry in self.hist]
        energy_values = [entry['e_best'] for entry in self.hist]

        fig = px.line(x=steps, y=energy_values, labels={'x': 'Step', 'y': 'Best Energy Value'},
                      title='Optimization Progress', markers=True, line_shape='linear')

        fig.show()
        
            




    def safe_exp(self, x):
        """_summary_

        Args:
            x (_type_): valor flotante

        Returns:
            _type_: retorna valor exponencial de x, sino no se puede resolver devuelve 0.
        """
        try: 
            return exp(x)
        except: 
            return 0







