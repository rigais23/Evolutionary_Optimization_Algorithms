from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import math
from numpy import asarray, argsort, exp, sqrt, pi, cos
from numpy.random import rand, randn, seed
from copy import deepcopy

##################
## FIT FUNCTION ##
##################

def rastrigin(*X, A=10):
        # *X --> point coordinates
        # A --> hardness (how deep)
    return A*len(X) + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X]) # A*len(X) --> ensure the minimum is at [0,0] in 2D.


######################
## INDIVIDUAL CLASS ##
######################

class Individual:
    """
    Representa a un individuo en la población, conteniendo su solución ('x') y sus parámetros de estrategia ('sigma').
    """
    def __init__(self, n_dims, bounds, init_step_size=0.15):
        self.n_dims = n_dims
        self.bounds = bounds
        
        # CREATE RANDOM INITIAL INDIVIDUAL --> x(solution)
        self.x = bounds[:, 0] + rand(n_dims) * (bounds[:, 1] - bounds[:, 0])
        
        # CREATE A MATRIX OF SIGMA VALUES; PREDEIFNED init_step_size FOR ALL THE INDIVIDUAL ANS THE SAME DIMENSIONS AS THEM
        self.sigma = np.full(n_dims, init_step_size)
        
        # INITIAL FITNESS VALUE
        self.fitness = np.inf 


    # Check if each of the point variables are inside the bounds
    def in_bounds(self):
        return np.all(self.x >= self.bounds[:, 0]) and np.all(self.x <= self.bounds[:, 1])
    


    def mutate(self, tau, tau_prime, sigma_floor=1e-5):
        """
        Muta al individuo (self) IN-PLACE usando mutación auto-adaptable no correlacionada (n sigmas).
        """
        # MUTATE Sigma: σ_i' := σ_i * exp(N(0, τ') + N_i(0, τ))
        global_factor = np.random.randn() * tau_prime # common gaussian to all individuals --> one same single random value
        local_factors = np.random.randn(self.n_dims) * tau # individual gaussian --> vector of random numbers (one for each dimension)
        self.sigma = self.sigma * np.exp(global_factor + local_factors)
        
        # Avoid sigma becomes 0
        self.sigma = np.maximum(self.sigma, sigma_floor)


        # Mutate x
        self.x = self.x + self.sigma * np.random.randn(self.n_dims)

        # Force bounds 
        self.x = np.clip(self.x, self.bounds[:, 0], self.bounds[:, 1])



##################
## ES ALGORITHM ##
##################

class ES_Algorithm:
    """
    Implementa el algoritmo de Estrategia Evolutiva (μ, λ).
    """
    def __init__(self, objective, bounds, mu, lam, n_iter, init_step_size=0.15, target_fitness=None):
        self.objective = objective # Fit function --> rastrigin
        self.bounds = np.asarray(bounds) # ensure are arrays
        self.mu = mu # padres
        self.lam = lam # size population (children)
        self.n_iter = n_iter
        self.init_step_size = init_step_size # first sigma values
        self.target_fitness = target_fitness
        
        self.n_dims = len(self.bounds)
        self.n_children = int(self.lam / self.mu) # number of children per parent
        
        # Diagonal self-adaptive Mutation Parameters
        self.tau = 1.0 / (2 * np.sqrt(self.n_dims))
        self.tau_prime = 1.0 / (2 * self.n_dims)

        # Initialize population
        self.population = []
        self.best_ind = None
        self.best_eval = np.inf


    # Evaluate fitness function
    def fitness_function(self, individual):
        return self.objective(*individual.x) # * for rastrigin compatibility


    # Create initial population with 'lam' individuos
    def init_population(self):
        self.population = [
            Individual(self.n_dims, self.bounds, self.init_step_size) 
            for _ in range(self.lam) # Create 'lam' individuals
        ]

        # Evaluate initial population 
        for ind in self.population: # instance Individual
            ind.fitness = self.fitness_function(ind)
        
        print(f"Initial population of {self.lam} individuals.")


    # Select best 'mu' individuas
    def select(self):
        # Ordenar la población por fitness (de menor a mayor)
        self.population.sort(key=lambda ind: ind.fitness) # Ordenar la población por fitness (de menor a mayor)
        # Devolver los 'mu' mejores
        return self.population[:self.mu]


    # Run the entire algorithm
    def run(self):
        """
        Pseudocódigo del algoritmo (µ, λ)-ES aplicado:
        
        1. Inicializar población P de 'λ' individuos.
        2. Evaluar P.
        3. Repetir durante 'n_iter' generaciones:
        4.   Seleccionar 'µ' mejores individuos (Padres) de P.
        5.   Guardar el mejor individuo global encontrado.
        6.   Crear una nueva población P' (Hijos) vacía.
        7.   Para cada Padre:
        8.     Crear 'n_children' copias.
        9.     Mutar cada copia (x y σ).
        10.    Añadir copias mutadas a P'.
        11.  Evaluar P'.
        12.  Reemplazar P con P'.
        13. Devolver el mejor individuo global.
        """
        
        # 1. & 2.
        self.init_population()
        self.population.sort(key=lambda ind: ind.fitness)
        self.best_ind = deepcopy(self.population[0])
        self.best_eval = self.best_ind.fitness

        # 3.
        for epoch in range(self.n_iter): # GENERATION
            # 4.
            parents = self.select()
            
            # 5.
            if parents[0].fitness < self.best_eval:
                # update the parameters
                self.best_ind = deepcopy(parents[0]) 
                self.best_eval = self.best_ind.fitness 
                print(f"{epoch}, Best: f({self.best_ind.x}) = {self.best_eval:.5f}")
            
            if self.target_fitness is not None and self.best_eval <= self.target_fitness:
                print(f"\n--- Target fitness reached at epoch {epoch}! ---")
                print(f"Target: {self.target_fitness}, Achieved: {self.best_eval:.5f}")
                break # Exit the main loop
            
            # 6.
            children = []
            # 7.
            for parent in parents:
                # 8.
                for _ in range(self.n_children):
                    child = deepcopy(parent)
                    # 9.
                    child.mutate(self.tau, self.tau_prime)
                    # 10.
                    children.append(child)
            
            # 11.
            for child in children:
                child.fitness = self.fitness_function(child)
                
            # 12.
            self.population = children
            
        return [self.best_ind.x, self.best_eval]




# --- Main execution ---
if __name__ == '__main__':
    # seed the pseudorandom number generator
    # seed(23)
    
    # define range for input
    bounds = asarray([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    
    # define the total iterations
    n_iter = 5000 # 5000 es mucho para un test, 1000 es suficiente
    
    # define the *initial* step size
    step_size = 0.5 # Un valor inicial más grande es común
    
    # number of parents selected
    mu = 100
    
    # the number of children generated by parents
    lam = 500
    
    # define the target fitness to stop at
    target_stop_fitness = 2 # Stop when fitness is 2 or less
    
    # 1. Crear la instancia del algoritmo
    optimizer = ES_Algorithm(
        objective=rastrigin,
        bounds=bounds,
        mu=mu,
        lam=lam,
        n_iter=n_iter,
        init_step_size=step_size,
        target_fitness=target_stop_fitness
    )
    
    # 2. Ejecutar la optimización
    best, score = optimizer.run()
    
    print('Done!')
    print('f(%s) = %f' % (best, score))