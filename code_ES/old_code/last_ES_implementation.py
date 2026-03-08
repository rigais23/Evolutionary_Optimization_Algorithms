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
import time # For tracking execution time
import pandas as pd # For data aggregation
import seaborn as sns # <-- MODIFICATION: Added for better boxplots

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
        global_factor = np.random.randn() * tau_prime # common gaussian to all individuals --> one same single random value
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
        
        # Statistics & History Tracking
        self.fitness_calls = 0
        self.generations_run = 0
        self.total_time = 0.0
        self.history = [] # List to store results


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
        self.fitness_calls += len(self.population) # Track initial evaluations
        
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
        
        start_time = time.time() # Start timer
        
        # 1. & 2.
        self.init_population()
        self.population.sort(key=lambda ind: ind.fitness)
        self.best_ind = deepcopy(self.population[0])
        self.best_eval = self.best_ind.fitness
        
        # Store initial state (Generation 0)
        pop_fitnesses_init = [ind.fitness for ind in self.population]
        self.history.append({
            'epoch': 0, 
            'best_fitness': self.best_eval, 
            'avg_fitness': np.mean(pop_fitnesses_init),
            'std_fitness': np.std(pop_fitnesses_init),
            'avg_sigma': np.mean([np.mean(ind.sigma) for ind in self.population]),
            'fitness_calls': self.fitness_calls,
            'time_elapsed': time.time() - start_time
        })

        # 3.
        epoch = 0 # In case n_iter = 0
        run_completed = False # To track if loop runs at all
        for epoch in range(self.n_iter): # GENERATION
            run_completed = True
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
            self.fitness_calls += len(children) # Track new evaluations
                
            # 12.
            self.population = children
            
            # Store history for this generation (epoch 0 -> gen 1)
            pop_fitnesses = [ind.fitness for ind in self.population]
            self.history.append({
                'epoch': epoch + 1,
                'best_fitness': self.best_eval, # Best fitness *so far*
                'avg_fitness': np.mean(pop_fitnesses),
                'std_fitness': np.std(pop_fitnesses),
                'avg_sigma': np.mean([np.mean(ind.sigma) for ind in self.population]),
                'fitness_calls': self.fitness_calls,
                'time_elapsed': time.time() - start_time
            })
        
        # After loop finishes (normally or by break)
        self.total_time = time.time() - start_time
        self.generations_run = epoch + 1 if run_completed else 0
            
        return [self.best_ind.x, self.best_eval]




# --- Main execution ---
if __name__ == '__main__':
    # seed the pseudorandom number generator
    # seed(23)
    
    # define range for input
    bounds = asarray([[-2, 2], [-2, 2]])
    
    # define the total iterations
    n_iter = 1000 # 5000 es mucho para un test, 1000 es suficiente
    
    # define the *initial* step size
    step_size = 1 # Un valor inicial más grande es común
    
    # number of parents selected
    mu = 50
    
    # the number of children generated by parents
    lam = 1000
    
    # define the target fitness to stop at
    target_stop_fitness = 1e-5 # This is 0.00001
    
    # --- Define number of runs and storage lists ---
    N_RUNS = 15 # Number of times to run the experiment
    all_times = []
    all_generations = []
    all_fitness_calls = []
    all_final_best_fitness = [] # <-- MODIFICATION: Added to plot final fitness
    success_count = 0
    all_run_histories = [] # Store full histories

    print(f"--- Starting {N_RUNS} ES Algorithm Runs ---")

    # --- Loop N_RUNS times ---
    for i in range(N_RUNS):
        print(f"\n--- Starting Run {i+1}/{N_RUNS} ---") # Added newline for clarity
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
        
        # --- Store results ---
        all_times.append(optimizer.total_time)
        all_generations.append(optimizer.generations_run)
        all_fitness_calls.append(optimizer.fitness_calls)
        all_final_best_fitness.append(score) # <-- MODIFICATION: Store final score
        all_run_histories.append(optimizer.history) 
        
        # Check if the desired performance was reached
        if score <= target_stop_fitness:
            success_count += 1
            
        print(f"--- Run {i+1}/{N_RUNS} Complete. Best Score: {score:.5f} ---")

    print('Done!')

    # --- Print Average Statistics ---
    print(f"\n--- Average Statistics Over {N_RUNS} Runs ---")
    print(f"Average Execution Time: {np.mean(all_times):.4f} seconds")
    print(f"Average Generations Run:  {np.mean(all_generations):.2f}")
    print(f"Average Fitness Calls:  {np.mean(all_fitness_calls):.2f}")
    print(f"Success Proportion:     {success_count / N_RUNS:.2%} ({success_count}/{N_RUNS})")
    print("------------------------------------------")


    # --- MODIFICATION: Process and Plot Data with Boxplots ---
    print("\n--- Processing data for plots... ---")
    
    # --- Plot 1: Boxplots of Final Summary Metrics ---
    # This shows the distribution of your summary statistics
    
    fig_summary, axes_summary = plt.subplots(1, 4, figsize=(20, 6))
    
    sns.boxplot(data=all_times, ax=axes_summary[0], orient='v')
    axes_summary[0].set_title('Distribution of Execution Time')
    axes_summary[0].set_ylabel('Time (s)')

    sns.boxplot(data=all_generations, ax=axes_summary[1], orient='v')
    axes_summary[1].set_title('Distribution of Generations Run')
    axes_summary[1].set_ylabel('Generations')
    
    sns.boxplot(data=all_fitness_calls, ax=axes_summary[2], orient='v')
    axes_summary[2].set_title('Distribution of Fitness Calls')
    axes_summary[2].set_ylabel('Count')
    
    sns.boxplot(data=all_final_best_fitness, ax=axes_summary[3], orient='v')
    axes_summary[3].set_title('Distribution of Final Best Fitness')
    axes_summary[3].set_ylabel('Fitness Score')
    
    fig_summary.suptitle(f'Distribution of Final Metrics Over {N_RUNS} Runs', fontsize=16, y=1.03)
    plt.tight_layout()
    
    
    # --- Plot 2: Boxplots of Convergence Over Time ---
    # This shows how the distribution of 'best_fitness' changes at
    # key milestones (e.g., 0%, 25%, 50%, 75%, 100% of the way through)
    
    # Combine all histories into a single DataFrame
    all_dfs = []
    for i, history in enumerate(all_run_histories):
        run_df = pd.DataFrame.from_records(history)
        run_df['run'] = i
        all_dfs.append(run_df)
    full_history_df = pd.concat(all_dfs)

    # Find the maximum epoch *actually* reached in the data
    # (in case some runs stopped early)
    max_epoch_run = full_history_df['epoch'].max()
    
    # Define epoch milestones for the boxplots
    milestones = [
        0, 
        int(max_epoch_run * 0.25), 
        int(max_epoch_run * 0.5), 
        int(max_epoch_run * 0.75), 
        max_epoch_run
    ]
    # Ensure milestones are unique (in case of very short runs)
    milestones = sorted(list(set(milestones))) 

    print(f"Plotting convergence milestones at epochs: {milestones}")
    
    # Filter the DataFrame to only include these milestone epochs
    milestone_df = full_history_df[full_history_df['epoch'].isin(milestones)]
    
    # Create the convergence plots
    fig_conv, axes_conv = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 2a: Best Fitness at milestones
    sns.boxplot(
        x='epoch', 
        y='best_fitness', 
        data=milestone_df, 
        ax=axes_conv[0]
    )
    axes_conv[0].set_title(f'Distribution of Best Fitness at Milestones (over {N_RUNS} runs)')
    axes_conv[0].set_ylabel('Best Fitness')
    axes_conv[0].set_xlabel('Generation (Epoch)')
    # Use log scale if fitness varies wildly, optional
    axes_conv[0].set_yscale('log') 
    
    # Plot 2b: Average Sigma at milestones
    sns.boxplot(
        x='epoch', 
        y='avg_sigma', 
        data=milestone_df, 
        ax=axes_conv[1]
    )
    axes_conv[1].set_title(f'Distribution of Avg. Sigma at Milestones (over {N_RUNS} runs)')
    axes_conv[1].set_ylabel('Average Sigma')
    axes_conv[1].set_xlabel('Generation (Epoch)')

    fig_conv.suptitle('Algorithm Convergence & Adaptation Over Time', fontsize=16, y=1.03)
    plt.tight_layout()

    print("Showing plots...")
    plt.show()
    # --- End Modification ---

    # --- MODIFICATION: Process and Plot Data (Line Plots) ---
    print("\n--- Processing data for plots... ---")
    
    # Combine all histories into a single DataFrame for easier processing
    all_dfs = []
    for i, history in enumerate(all_run_histories):
        run_df = pd.DataFrame.from_records(history)
        run_df['run'] = i
        all_dfs.append(run_df)
    
    # This single DataFrame has all data from all runs
    full_history_df = pd.concat(all_dfs)

    # Calculate mean and std dev *across runs* for each epoch
    # This gives us a new DataFrame indexed by 'epoch'
    metrics_df = full_history_df.groupby('epoch').agg(
        # Best fitness (across runs)
        avg_best_fitness=('best_fitness', 'mean'),
        std_best_fitness=('best_fitness', 'std'),
        
        # Avg fitness of the population (averaged across runs)
        avg_pop_fitness=('avg_fitness', 'mean'),
        
        # Avg std dev of the population (averaged across runs)
        # This shows population convergence
        avg_pop_std=('std_fitness', 'mean'), 
        
        # Avg sigma (averaged across runs)
        avg_sigma=('avg_sigma', 'mean')
    )
    
    # Create the plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Average Algorithm Performance Over {N_RUNS} Runs', fontsize=16, y=1.03)
    
    # --- Plot 1: Best Fitness Convergence (with std dev across runs) ---
    ax = axes[0]
    # Plot the average line
    ax.plot(metrics_df.index, metrics_df['avg_best_fitness'], label='Mean Best Fitness')
    # Plot the shaded standard deviation area
    ax.fill_between(
        metrics_df.index,
        metrics_df['avg_best_fitness'] - metrics_df['std_best_fitness'],
        metrics_df['avg_best_fitness'] + metrics_df['std_best_fitness'],
        color='blue',
        alpha=0.2,
        label='Std. Dev. (across runs)'
    )
    ax.set_yscale('log') # Log scale is useful for fitness
    ax.set_title(f'Mean Best Fitness per Generation')
    ax.set_ylabel('Best Fitness (log scale)')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # --- Plot 2: Population Fitness (Average and Std. Dev.) ---
    ax = axes[1]
    ax.plot(metrics_df.index, metrics_df['avg_pop_fitness'], label='Mean of Population Average Fitness')
    ax.plot(metrics_df.index, metrics_df['avg_pop_std'], label='Mean of Population Std. Dev.', linestyle='--')
    ax.set_yscale('log')
    ax.set_title('Average Population Dynamics (Diversity)')
    ax.set_ylabel('Fitness (log scale)')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # --- Plot 3: Average Sigma (Strategy Parameter) Adaptation ---
    ax = axes[2]
    ax.plot(metrics_df.index, metrics_df['avg_sigma'], label='Mean of Avg. Sigmas')
    ax.set_title('Average Strategy Parameter (Sigma) Adaptation')
    ax.set_xlabel('Generation (Epoch)')
    ax.set_ylabel('Average Sigma Value')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make room for suptitle
    print("Showing plots...")
    plt.show()
    # --- End Modification ---