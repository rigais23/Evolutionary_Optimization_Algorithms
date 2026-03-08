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
import time  # For tracking execution time
import pandas as pd  # For data aggregation
import seaborn as sns  # <-- MODIFICATION: Added for better boxplots
import os  # <-- NEW: Import os for creating directories
import contextlib  # <-- NEW: Import for redirecting stdout

# --- Imports for L-BFGS-B ---
import scipy.optimize
from scipy.stats import qmc
from numpy.linalg import norm  # For vector distance calculation

# --- NEW: Imports for Statistical Testing ---
from scipy.stats import kruskal, mannwhitneyu
import scikit_posthocs as sp  # You may need to run: pip install scikit-posthocs


##################
## FIT FUNCTION ##
##################

# --- MODIFIED: To accept a single vector X for scipy.optimize compatibility ---
def rastrigin(X, A=10):
    # *X --> point coordinates
    # A --> hardness (how deep)
    return A * len(X) + sum(
        [(x ** 2 - A * np.cos(2 * math.pi * x)) for x in X])  # A*len(X) --> ensure the minimum is at [0,0] in 2D.


# --- NEW: Gradient (Jacobian) of Rastrigin for L-BFGS-B ---
def rastrigin_grad(X, A=10):
    X = np.asarray(X)  # Ensure it's a numpy array
    return 2 * X + A * np.sin(2 * math.pi * X) * (2 * math.pi)


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
        global_factor = np.random.randn() * tau_prime  # common gaussian to all individuals --> one same single random value
        local_factors = np.random.randn(
            self.n_dims) * tau  # individual gaussian --> vector of random numbers (one for each dimension)
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
        self.objective = objective  # Fit function --> rastrigin
        self.bounds = np.asarray(bounds)  # ensure are arrays
        self.mu = mu  # padres
        self.lam = lam  # size population (children)
        self.n_iter = n_iter
        self.init_step_size = init_step_size  # first sigma values
        self.target_fitness = target_fitness

        self.n_dims = len(self.bounds)
        self.n_children = int(self.lam / self.mu)  # number of children per parent

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
        self.history = []  # List to store results
        self.target_reached = False  # To track success

    # Evaluate fitness function
    def fitness_function(self, individual):
        # --- MODIFIED: Pass vector 'x' directly, not with * ---
        return self.objective(individual.x)  # * for rastrigin compatibility

    # Create initial population with 'lam' individuos
    def init_population(self):
        self.population = [
            Individual(self.n_dims, self.bounds, self.init_step_size)
            for _ in range(self.lam)  # Create 'lam' individuals
        ]

        # Evaluate initial population
        for ind in self.population:  # instance Individual
            ind.fitness = self.fitness_function(ind)
        self.fitness_calls += len(self.population)  # Track initial evaluations

        # --- MODIFIED: Quieted this print statement for tuning experiment ---
        # print(f"Initial population of {self.lam} individuals.")

    # Select best 'mu' individuas
    def select(self):
        # Ordenar la población por fitness (de menor a mayor)
        self.population.sort(key=lambda ind: ind.fitness)  # Ordenar la población por fitness (de menor a mayor)
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

        start_time = time.time()  # Start timer

        # --- NEW: Parameters for stagnation check ---
        STAGNATION_WINDOW = 10  # The number of generations to look back
        IMPROVEMENT_THRESHOLD = 1e-6  # The minimum required fitness improvement

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
        epoch = 0  # In case n_iter = 0
        run_completed = False  # To track if loop runs at all
        for epoch in range(self.n_iter):  # GENERATION
            run_completed = True
            # 4.
            parents = self.select()

            # 5.
            if parents[0].fitness < self.best_eval:
                # update the parameters
                self.best_ind = deepcopy(parents[0])
                self.best_eval = self.best_ind.fitness
                # --- MODIFIED: Quieted this print statement for tuning experiment ---
                # print(f"{epoch}, Best: f({self.best_ind.x}) = {self.best_eval:.5f}")

            if self.target_fitness is not None and self.best_eval <= self.target_fitness:
                # print(f"\n--- Target fitness reached at epoch {epoch}! ---")
                # print(f"Target: {self.target_fitness}, Achieved: {self.best_eval:.5f}")
                self.target_reached = True  # Set success flag
                break  # Exit the main loop

            # --- NEW: Stagnation Check ---
            # Stop if fitness hasn't improved by at least IMPROVEMENT_THRESHOLD
            # over the last STAGNATION_WINDOW generations.
            if epoch >= STAGNATION_WINDOW:
                # self.history[0] is from epoch 0.
                # At epoch 10, we check self.history[0] (which is epoch 0).
                # The index is (epoch - STAGNATION_WINDOW)
                fitness_then = self.history[epoch - STAGNATION_WINDOW]['best_fitness']
                improvement = fitness_then - self.best_eval

                if improvement < IMPROVEMENT_THRESHOLD:
                    print(f"\n--- Stagnation detected at epoch {epoch}! ---")
                    print(
                        f"Best fitness {STAGNATION_WINDOW} gens ago (epoch {epoch - STAGNATION_WINDOW}): {fitness_then:.5f}")
                    print(f"Current best fitness (epoch {epoch}):     {self.best_eval:.5f}")
                    print(f"Improvement ({improvement:.7f}) is below threshold ({IMPROVEMENT_THRESHOLD})")
                    break  # Exit the main loop
            # --- End of Stagnation Check ---

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
            self.fitness_calls += len(children)  # Track new evaluations

            # 12.
            self.population = children

            # Store history for this generation (epoch 0 -> gen 1)
            pop_fitnesses = [ind.fitness for ind in self.population]
            self.history.append({
                'epoch': epoch + 1,
                'best_fitness': self.best_eval,  # Best fitness *so far*
                'avg_fitness': np.mean(pop_fitnesses),
                'std_fitness': np.std(pop_fitnesses),
                'avg_sigma': np.mean([np.mean(ind.sigma) for ind in self.population]),
                'fitness_calls': self.fitness_calls,
                'time_elapsed': time.time() - start_time
            })

        # After loop finishes (normally or by break)
        self.total_time = time.time() - start_time
        # --- MODIFIED: Ensure generations_run is correct even if loop breaks ---
        # We add 1 because 'epoch' is 0-indexed and we want a count.
        # If loop breaks at epoch 10, it has *run* 11 generations (0-10).
        self.generations_run = epoch + 1 if run_completed else 0

        # --- MODIFIED: Return vector and score ---
        return [self.best_ind.x, self.best_eval]


# --- NEW: Wrapper for L-BFGS-B Multi-start ---
def run_lbfgsb_multistart(objective, gradient, bounds, n_starts, target_fitness):
    """
    Runs a multi-start L-BFGS-B optimization.
    Returns a dictionary of statistics for comparison.
    """
    start_time = time.time()

    n_dims = len(bounds)
    best_score = np.inf
    best_x = None
    total_fitness_calls = 0
    target_reached = False

    # Create the Scipy bounds object
    scipy_bounds = scipy.optimize.Bounds(bounds[:, 0], bounds[:, 1])

    # --- Generate good starting points ---
    sampler = qmc.LatinHypercube(d=n_dims, seed=int(time.time() * 1000) % 2 ** 32)  # Add seed for variety
    start_points = sampler.random(n=n_starts)
    start_points = qmc.scale(start_points, bounds[:, 0], bounds[:, 1])

    for i in range(n_starts):
        x0 = start_points[i]

        # Run the local optimizer
        res = scipy.optimize.minimize(
            fun=objective,  # Objective function
            x0=x0,  # Starting point
            method='L-BFGS-B',
            jac=gradient,  # Gradient function
            bounds=scipy_bounds
        )

        total_fitness_calls += res.nfev

        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x

            if not target_reached and best_score <= target_fitness:
                target_reached = True

    end_time = time.time()

    # --- MODIFIED: Return best_x as well ---
    return {
        "final_fitness": best_score,
        "best_vector": best_x,  # Return the best solution vector
        "time": end_time - start_time,
        "fitness_calls": total_fitness_calls,
        "target_reached": target_reached,
        "generations": n_starts  # Use 'n_starts' as the analogous metric
    }


# -----------------------------------------------------------------
# --- NEW: EXPERIMENT 1: ES PARAMETER TUNING ---
# -----------------------------------------------------------------
def run_experiment_1_es_tuning(n_runs, n_iter, target_fitness, base_bounds):
    """
    Compares different ES configurations.
    """
    # --- Parameters to test ---
    step_sizes = [0.1, 0.5, 1.0]
    mu_lam_pairs = [(50, 1000), (50, 500), (50, 250)]

    all_results_list = []

    print(f"--- Starting Experiment 1: ES Parameter Tuning ({n_runs} runs per config) ---")

    for step_size in step_sizes:
        for (mu, lam) in mu_lam_pairs:

            config_name = f"ES (step={step_size}, mu={mu}, lam={lam})"
            print(f"\n--- Testing Configuration: {config_name} ---")

            for i in range(n_runs):
                optimizer = ES_Algorithm(
                    objective=rastrigin,
                    bounds=base_bounds,
                    mu=mu,
                    lam=lam,
                    n_iter=n_iter,
                    init_step_size=step_size,
                    target_fitness=target_fitness
                )

                best_vector, score = optimizer.run()

                # --- MODIFICATION: Store individual parameters for sorting ---
                all_results_list.append({
                    'config_name': config_name,
                    'step_size': step_size,  # <-- NEW
                    'mu': mu,  # <-- NEW
                    'lam': lam,  # <-- NEW
                    'ratio': lam / mu,  # <-- NEW (This is your selective pressure)
                    'run': i,
                    'time': optimizer.total_time,
                    'generations': optimizer.generations_run,
                    'fitness_calls': optimizer.fitness_calls,
                    'final_fitness': score,
                    'target_reached': optimizer.target_reached,
                    'best_vector': best_vector
                })
                # --- End Modification ---
                print(f"  Run {i + 1}/{n_runs} Complete. Score: {score:.5f}")

    print("\n--- Experiment 1 Complete ---")

    results_df = pd.DataFrame.from_records(all_results_list)

    # --- Print Statistics ---
    print("\n--- Experiment 1: Summary Statistics ---")
    summary = results_df.groupby('config_name').agg(
        avg_time=('time', 'mean'),
        avg_fitness=('final_fitness', 'mean'),
        avg_calls=('fitness_calls', 'mean'),
        avg_gens=('generations', 'mean'),
        success_rate=('target_reached', 'mean')
    )
    print(summary.to_markdown(floatfmt=".3f"))

    # --- Find the best configuration ---
    best_config_name = summary['avg_fitness'].idxmin()
    print(f"\nBest configuration based on avg_fitness: {best_config_name}")

    # --- NEW: Statistical Analysis (Kruskal-Wallis + Dunn's Post-Hoc) ---
    print("\n\n--- Experiment 1: Statistical Analysis ---")
    print("Running Kruskal-Wallis H-test for each metric (alpha=0.05)")
    print("If p < 0.05, a post-hoc Dunn's test will be performed.")

    metrics_to_test = ['final_fitness', 'time', 'fitness_calls', 'generations', 'target_reached']

    for metric in metrics_to_test:
        print(f"\n--- Statistical Test for: {metric} ---")
        # Create a list of arrays, one for each group
        groups = [group[metric].values for name, group in results_df.groupby('config_name')]

        # 1. Omnibus Test: Kruskal-Wallis
        h_stat, p_val = kruskal(*groups)
        print(f"Kruskal-Wallis Test: H-statistic = {h_stat:.4f}, p-value = {p_val:.4f}")

        if p_val < 0.05:
            print("p < 0.05: Significant difference detected. Running Dunn's post-hoc test...")
            # 2. Post-Hoc Test: Dunn's
            # We use the original dataframe which is in 'long' format
            dunn_results = sp.posthoc_dunn(results_df, val_col=metric, group_col='config_name', p_adjust='holm')
            print("Dunn's Post-Hoc Test (p-values, Holm correction):")
            print(dunn_results.to_markdown(floatfmt=".4f"))
        else:
            print("p >= 0.05: No statistically significant difference detected among configurations.")
    # --- End of Statistical Analysis ---

    # --- Plotting ---
    print("\nGenerating and saving plots for Experiment 1...")

    # --- MODIFICATION: Create a custom sort order for the plots ---
    # We sort by ratio (selective pressure) descending, then step_size ascending
    plot_order = results_df.sort_values(
        by=['ratio', 'step_size'],
        ascending=[False, True]
    )['config_name'].unique()
    # --- End Modification ---

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Experiment 1: ES Parameter Tuning Results', fontsize=16, y=1.03)

    # Plot 1: Final Fitness
    # --- MODIFICATION: Added order=plot_order ---
    sns.boxplot(data=results_df, x='config_name', y='final_fitness', ax=axes[0, 0], order=plot_order)
    axes[0, 0].set_title('Final Fitness (Lower is Better)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel(None)
    axes[0, 0].tick_params(axis='x', rotation=30)

    # Plot 2: Execution Time
    # --- MODIFICATION: Added order=plot_order ---
    sns.boxplot(data=results_df, x='config_name', y='time', ax=axes[0, 1], order=plot_order)
    axes[0, 1].set_title('Execution Time')
    axes[0, 1].set_xlabel(None)
    axes[0, 1].tick_params(axis='x', rotation=30)

    # Plot 3: Fitness Calls
    # --- MODIFICATION: Added order=plot_order ---
    sns.boxplot(data=results_df, x='config_name', y='fitness_calls', ax=axes[1, 0], order=plot_order)
    axes[1, 0].set_title('Fitness Function Calls')
    axes[1, 0].set_xlabel(None)
    axes[1, 0].tick_params(axis='x', rotation=30)

    # Plot 4: Success Rate (Bar plot is better for this)
    success_df = summary['success_rate'].reset_index()
    # --- MODIFICATION: Added order=plot_order ---
    sns.barplot(data=success_df, x='config_name', y='success_rate', ax=axes[1, 1], palette='viridis', order=plot_order)
    axes[1, 1].set_title('Success Rate (Target Reached)')
    axes[1, 1].set_xlabel(None)
    axes[1, 1].set_ylabel('Proportion')
    axes[1, 1].tick_params(axis='x', rotation=30)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- MODIFIED: Save the figure ---
    fig.savefig('./experiment1/tuning_summary_boxplots.png', bbox_inches='tight')
    plt.show()

    return best_config_name


# -----------------------------------------------------------------
# --- NEW: EXPERIMENT 2: ALGORITHM COMPARISON ---
# -----------------------------------------------------------------
def run_experiment_2_comparison(best_es_config, n_runs, n_iter, target_fitness, base_bounds):
    """
    Compares the best ES config vs. L-BFGS-B.
    best_es_config is a dict: {'step_size': 0.5, 'mu': 50, 'lam': 1000}
    """

    all_results_list = []
    all_run_histories = []  # For ES-specific plots

    # --- L-BFGS-B Parameter ---
    n_starts_lbfgsb = best_es_config['lam']  # Use 'lam' as the number of starting points

    print(f"--- Starting Experiment 2: ES vs L-BFGS-B ({n_runs} runs) ---")
    print(f"Best ES Config: {best_es_config}")

    # --- Loop N_RUNS times for ES ---
    print(f"\n--- Running ES Algorithm ---")
    for i in range(n_runs):
        print(f"  Starting ES Run {i + 1}/{n_runs}...")
        optimizer = ES_Algorithm(
            objective=rastrigin,
            bounds=base_bounds,
            mu=best_es_config['mu'],
            lam=best_es_config['lam'],
            n_iter=n_iter,
            init_step_size=best_es_config['step_size'],
            target_fitness=target_fitness
        )

        best_vector, score = optimizer.run()

        all_results_list.append({
            'algorithm': 'ES (Best Config)',
            'run': i,
            'time': optimizer.total_time,
            'generations': optimizer.generations_run,
            'fitness_calls': optimizer.fitness_calls,
            'final_fitness': score,
            'target_reached': optimizer.target_reached,
            'best_vector': best_vector
        })
        all_run_histories.append(optimizer.history)

        # --- Loop N_RUNS times for L-BFGS-B ---
    print(f"\n--- Running L-BFGS-B Algorithm ---")
    for i in range(n_runs):
        print(f"  Starting L-BFGS-B Run {i + 1}/{n_runs}...")
        stats = run_lbfgsb_multistart(
            objective=rastrigin,
            gradient=rastrigin_grad,
            bounds=base_bounds,
            n_starts=n_starts_lbfgsb,
            target_fitness=target_fitness
        )

        all_results_list.append({
            'algorithm': 'L-BFGS-B (Multi)',
            'run': i,
            'time': stats['time'],
            'generations': stats['generations'],  # Stores n_starts
            'fitness_calls': stats['fitness_calls'],
            'final_fitness': stats['final_fitness'],
            'target_reached': stats['target_reached'],
            'best_vector': stats['best_vector']
        })

    print("\n--- Experiment 2 Complete ---")

    results_df = pd.DataFrame.from_records(all_results_list)

    # --- NEW: Calculate Solution Distance metric ---
    # This is the L2 norm (Euclidean distance) from the vector to the origin [0, 0, ...]
    results_df['solution_distance'] = results_df['best_vector'].apply(norm)

    # --- Run the analysis from your previous code ---

    # 1. Average Execution Time
    print("\n1. Average Execution Time:")
    avg_time = results_df.groupby('algorithm')['time'].agg(['mean', 'std'])
    print(avg_time.to_markdown(floatfmt=".4f"))

    # 2. Average Solution Vector
    print("\n2. Average Solution Vector (Bias & Variance):")
    vector_analysis = []
    for alg_name, group in results_df.groupby('algorithm'):
        # Handle cases where no vector was found (e.g. if L-BFGS-B failed, though unlikely)
        valid_vectors = group['best_vector'].dropna()
        if len(valid_vectors) > 0:
            vectors = np.stack(valid_vectors.values)
            mean_vector = np.mean(vectors, axis=0)
            dist_of_mean_vec = norm(mean_vector)
            avg_dist_from_mean = np.mean(norm(vectors - mean_vector, axis=1))
        else:
            mean_vector = None
            dist_of_mean_vec = np.nan
            avg_dist_from_mean = np.nan

        vector_analysis.append({
            'Algorithm': alg_name,
            'Mean Solution Vector': mean_vector,
            'Bias (Dist. of Mean Vec from Origin)': dist_of_mean_vec,
            'Avg. Solution Spread (Variance)': avg_dist_from_mean
        })

    for item in vector_analysis:
        print(f"\nAlgorithm: {item['Algorithm']}")
        print(f"  - Mean Solution Vector: {item['Mean Solution Vector']}")
        print(f"  - Bias (Distance of Mean Vector from Origin): {item['Bias (Dist. of Mean Vec from Origin)']:.6f}")
        print(f"  - Spread (Avg. distance of solutions from their mean): {item['Avg. Solution Spread (Variance)']:.6f}")

    print("------------------------------------------")

    # --- Print Other Statistics ---
    print(f"\n--- Other Statistics (For Context) ---")
    summary_stats = results_df.groupby('algorithm').agg(
        avg_final_fitness=('final_fitness', 'mean'),
        std_final_fitness=('final_fitness', 'std'),
        avg_fitness_calls=('fitness_calls', 'mean'),
        std_fitness_calls=('fitness_calls', 'std'),
        success_rate=('target_reached', 'mean')
    )
    summary_stats['success_rate'] = summary_stats['success_rate'].apply(lambda x: f"{x:.2%}")
    print(summary_stats.to_markdown(floatfmt=".4f"))
    print("------------------------------------------")

    # --- NEW: Statistical Analysis (Mann-Whitney U Test) ---
    print("\n\n--- Experiment 2: Statistical Analysis ---")
    print("Running Mann-Whitney U test for each metric (alpha=0.05)")
    print("This tests if the two groups (ES vs. L-BFGS-B) are from different distributions.")

    # Get the data for each group
    group1_data = results_df[results_df['algorithm'] == 'ES (Best Config)']
    group2_data = results_df[results_df['algorithm'] == 'L-BFGS-B (Multi)']

    metrics_to_test = [
        ('Execution Time', 'time'),
        ('Final Fitness', 'final_fitness'),
        ('Final Solution Distance from Origin', 'solution_distance')  # Test the new metric
    ]

    for metric_name, metric_col in metrics_to_test:
        print(f"\n--- Statistical Test for: {metric_name} ---")
        try:
            stat, p_val = mannwhitneyu(group1_data[metric_col], group2_data[metric_col], alternative='two-sided')
            print(f"Mann-Whitney U Test: U-statistic = {stat:.4f}, p-value = {p_val:.4f}")
            if p_val < 0.05:
                print("p < 0.05: The difference between the algorithms is statistically significant.")
            else:
                print("p >= 0.05: No statistically significant difference detected.")
        except ValueError as e:
            print(f"Could not run test for {metric_name}. All values might be identical. Error: {e}")
    # --- End of Statistical Analysis ---

    # --- Plotting ---
    print("\nGenerating and saving plots for Experiment 2...")

    # Plot 1: Summary Boxplots (Comparison)
    fig_summary, axes_summary = plt.subplots(1, 4, figsize=(20, 6))

    sns.boxplot(data=results_df, x='algorithm', y='time', ax=axes_summary[0])
    axes_summary[0].set_title('Distribution of Execution Time')

    sns.boxplot(data=results_df, x='algorithm', y='generations', ax=axes_summary[1])
    axes_summary[1].set_title('Distribution of Generations / Starts')

    sns.boxplot(data=results_df, x='algorithm', y='fitness_calls', ax=axes_summary[2])
    axes_summary[2].set_title('Distribution of Fitness Calls')

    sns.boxplot(data=results_df, x='algorithm', y='final_fitness', ax=axes_summary[3])
    axes_summary[3].set_title('Distribution of Final Best Fitness')
    axes_summary[3].set_yscale('log')

    fig_summary.suptitle(f'Experiment 2: Algorithm Comparison Over {n_runs} Runs', fontsize=16, y=1.03)
    plt.tight_layout()

    # --- MODIFIED: Save the figure ---
    fig_summary.savefig('./experiment2/comparison_summary_boxplots.png', bbox_inches='tight')
    plt.show()

    # Plot 2: ES-Specific Convergence Plots (as before)
    if all_run_histories:
        full_history_df = pd.concat([pd.DataFrame.from_records(h) for h in all_run_histories])
        metrics_df = full_history_df.groupby('epoch').agg(
            avg_best_fitness=('best_fitness', 'mean'),
            std_best_fitness=('best_fitness', 'std'),
            avg_sigma=('avg_sigma', 'mean')
        )

        fig_es, axes_es = plt.subplots(1, 2, figsize=(14, 6))
        fig_es.suptitle('ES Algorithm Internal Convergence', fontsize=16, y=1.03)

        ax = axes_es[0]
        ax.plot(metrics_df.index, metrics_df['avg_best_fitness'], label='Mean Best Fitness')
        ax.fill_between(
            metrics_df.index,
            metrics_df['avg_best_fitness'] - metrics_df['std_best_fitness'],
            metrics_df['avg_best_fitness'] + metrics_df['std_best_fitness'],
            color='blue', alpha=0.2, label='Std. Dev.'
        )
        ax.set_yscale('log')
        ax.set_title(f'Mean Best Fitness per Generation')
        ax.set_ylabel('Best Fitness (log scale)')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)  # Added grid

        ax = axes_es[1]
        ax.plot(metrics_df.index, metrics_df['avg_sigma'], label='Mean of Avg. Sigmas')
        ax.set_title('Average Strategy Parameter (Sigma) Adaptation')
        ax.set_ylabel('Average Sigma Value')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)  # Added grid

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # --- MODIFIED: Save the figure ---
        fig_es.savefig('./experiment2/comparison_es_convergence.png', bbox_inches='tight')
        plt.show()


# -----------------------------------------------------------------
# --- NEW: EXPERIMENT 3: SCALABILITY ANALYSIS ---
# -----------------------------------------------------------------
def run_experiment_3_scalability(best_es_config, n_runs, n_iter, target_fitness):
    """
    Compares the best ES config on problems of increasing dimensionality.
    best_es_config is a dict: {'step_size': 0.5, 'mu': 50, 'lam': 1000}
    """

    # --- Parameters to test ---
    dimensions_to_test = [2, 4, 8, 16]
    base_bound_range = [-2, 2]  # Use the bounds from your old main

    all_results_list = []

    print(f"--- Starting Experiment 3: ES Scalability Analysis ({n_runs} runs per dimension) ---")
    print(f"Using ES Config: {best_es_config}")

    for D in dimensions_to_test:
        print(f"\n--- Testing Dimension: {D} ---")

        # Create new bounds for this dimension
        bounds = asarray([base_bound_range] * D)

        for i in range(n_runs):
            optimizer = ES_Algorithm(
                objective=rastrigin,
                bounds=bounds,
                mu=best_es_config['mu'],
                lam=best_es_config['lam'],
                n_iter=n_iter,
                init_step_size=best_es_config['step_size'],
                target_fitness=target_fitness
            )

            best_vector, score = optimizer.run()

            all_results_list.append({
                'dimensions': D,
                'run': i,
                'time': optimizer.total_time,
                'generations': optimizer.generations_run,
                'fitness_calls': optimizer.fitness_calls,
                'final_fitness': score,
                'target_reached': optimizer.target_reached,
                'best_vector': best_vector
            })
            print(f"  Run {i + 1}/{n_runs} Complete. Score: {score:.5f}")

    print("\n--- Experiment 3 Complete ---")

    results_df = pd.DataFrame.from_records(all_results_list)

    # --- Print Statistics ---
    print("\n--- Experiment 3: Summary Statistics ---")
    summary = results_df.groupby('dimensions').agg(
        avg_time=('time', 'mean'),
        avg_fitness=('final_fitness', 'mean'),
        avg_calls=('fitness_calls', 'mean'),
        avg_gens=('generations', 'mean'),
        success_rate=('target_reached', 'mean')
    )
    print(summary.to_markdown(floatfmt=".3f"))

    # --- NEW: Statistical Analysis (Kruskal-Wallis + Dunn's Post-Hoc) ---
    print("\n\n--- Experiment 3: Statistical Analysis ---")
    print("Running Kruskal-Wallis H-test for each metric (alpha=0.05)")
    print("If p < 0.05, a post-hoc Dunn's test will be performed.")

    metrics_to_test = ['final_fitness', 'time', 'fitness_calls', 'generations', 'target_reached']

    for metric in metrics_to_test:
        print(f"\n--- Statistical Test for: {metric} ---")
        # Create a list of arrays, one for each group
        groups = [group[metric].values for name, group in results_df.groupby('dimensions')]

        # 1. Omnibus Test: Kruskal-Wallis
        h_stat, p_val = kruskal(*groups)
        print(f"Kruskal-Wallis Test: H-statistic = {h_stat:.4f}, p-value = {p_val:.4f}")

        if p_val < 0.05:
            print("p < 0.05: Significant difference detected. Running Dunn's post-hoc test...")
            # 2. Post-Hoc Test: Dunn's
            dunn_results = sp.posthoc_dunn(results_df, val_col=metric, group_col='dimensions', p_adjust='holm')
            print("Dunn's Post-Hoc Test (p-values, Holm correction):")
            print(dunn_results.to_markdown(floatfmt=".4f"))
        else:
            print("p >= 0.05: No statistically significant difference detected among dimensions.")
    # --- End of Statistical Analysis ---

    # --- Plotting ---
    print("\nGenerating and saving plots for Experiment 3...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Experiment 3: ES Scalability Analysis', fontsize=16, y=1.03)

    # Use the raw dataframe for boxplots

    # Plot 1: Execution Time vs. Dimensions
    sns.boxplot(data=results_df, x='dimensions', y='time', ax=axes[0, 0])
    axes[0, 0].set_title('Execution Time vs. Dimensions')
    axes[0, 0].set_ylabel('Time (s)')
    axes[0, 0].set_xlabel('Number of Dimensions')

    # Plot 2: Final Fitness vs. Dimensions
    sns.boxplot(data=results_df, x='dimensions', y='final_fitness', ax=axes[0, 1])
    axes[0, 1].set_title('Final Fitness vs. Dimensions')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylabel('Final Fitness (log scale)')
    axes[0, 1].set_xlabel('Number of Dimensions')

    # Plot 3: Fitness Calls vs. Dimensions
    sns.boxplot(data=results_df, x='dimensions', y='fitness_calls', ax=axes[1, 0])
    axes[1, 0].set_title('Fitness Calls vs. Dimensions')
    axes[1, 0].set_ylabel('Fitness Calls')
    axes[1, 0].set_xlabel('Number of Dimensions')

    # Plot 4: Success Rate vs. Dimensions (Bar plot)
    success_df = summary['success_rate'].reset_index()
    sns.barplot(data=success_df, x='dimensions', y='success_rate', ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Success Rate vs. Dimensions')
    axes[1, 1].set_ylabel('Proportion')
    axes[1, 1].set_xlabel('Number of Dimensions')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- MODIFIED: Save the figure ---
    fig.savefig('./experiment3/scalability_summary_boxplots.png', bbox_inches='tight')
    plt.show()


# -----------------------------------------------------------------
# --- NEW: MAIN EXECUTION (CONTROLLER) ---
# -----------------------------------------------------------------

if __name__ == '__main__':
    # seed the pseudorandom number generator
    seed(23)

    # --- CHOOSE WHICH EXPERIMENT TO RUN ---
    # 1 = ES Parameter Tuning
    # 2 = ES vs. L-BFGS-B Comparison
    # 3 = ES Scalability (Dimensionality)
    EXPERIMENT_TO_RUN = 3
    # --------------------------------------

    # --- NEW: Create output directories ---
    os.makedirs("./experiment1", exist_ok=True)
    os.makedirs("./experiment2", exist_ok=True)
    os.makedirs("./experiment3", exist_ok=True)
    # --------------------------------------

    # --- Global Experiment Parameters ---

    # define range for input (for Exp 1 and 2)
    # Exp 3 will generate its own bounds
    base_bounds = [-2, 2]
    bounds = asarray([base_bounds] * 4)

    # define the total iterations
    n_iter = 1000  # 5000 es mucho para un test, 1000 es suficiente

    # define the target fitness to stop at
    target_stop_fitness = 0.01  # This is 0.01

    # --- Define number of runs ---
    N_RUNS = 30  # Number of times to run the experiment

    # --- "Best" ES config ---
    # Manually set this after running Experiment 1
    # This is needed for Experiments 2 and 3
    best_es_config_params = {
        'step_size': 0.1,  # Un valor inicial más grande es común
        'mu': 50,  # number of parents selected
        'lam': 1000  # the number of children generated by parents
    }

    # --- MODIFIED: Run the selected experiment with logging ---
    if EXPERIMENT_TO_RUN == 1:
        log_path = './experiment1/experiment1_log.txt'
        print(f"--- Running Experiment 1. All output will be logged to {log_path} ---")
        with open(log_path, 'w') as f:
            with contextlib.redirect_stdout(f):
                run_experiment_1_es_tuning(
                    n_runs=N_RUNS,
                    n_iter=n_iter,
                    target_fitness=target_stop_fitness,
                    base_bounds=bounds
                )

    elif EXPERIMENT_TO_RUN == 2:
        log_path = './experiment2/experiment2_log.txt'
        print(f"--- Running Experiment 2. All output will be logged to {log_path} ---")
        print("--- NOTE: Running Exp 2 with *manually set* best_es_config_params ---")  # This print goes to console
        with open(log_path, 'w') as f:
            with contextlib.redirect_stdout(f):
                print("--- NOTE: Running Exp 2 with *manually set* best_es_config_params ---")  # This print goes to log
                run_experiment_2_comparison(
                    best_es_config=best_es_config_params,
                    n_runs=N_RUNS,
                    n_iter=n_iter,
                    target_fitness=target_stop_fitness,
                    base_bounds=bounds
                )

    elif EXPERIMENT_TO_RUN == 3:
        log_path = './experiment3/experiment3_log.txt'
        print(f"--- Running Experiment 3. All output will be logged to {log_path} ---")
        print("--- NOTE: Running Exp 3 with *manually set* best_es_config_params ---")  # This print goes to console
        with open(log_path, 'w') as f:
            with contextlib.redirect_stdout(f):
                print("--- NOTE: Running Exp 3 with *manually set* best_es_config_params ---")  # This print goes to log
                run_experiment_3_scalability(
                    best_es_config=best_es_config_params,
                    n_runs=N_RUNS,
                    n_iter=n_iter,
                    target_fitness=target_stop_fitness
                )

    else:
        print(f"Invalid EXPERIMENT_TO_RUN value: {EXPERIMENT_TO_RUN}. Please set to 1, 2, or 3.")

    print("\n--- All selected experiments finished. ---")