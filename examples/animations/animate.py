import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cmaes._cma import CMA
from src.qaoa import QAOA
from scipy import stats
from protes import animation as protes_animation
import matplotlib.animation as animation
from pylab import rcParams
import math
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
from scipy.optimize import differential_evolution

rcParams["figure.figsize"] = 10, 5

def qaoa(x1, x2):
    x1_arr = np.asarray(x1)
    x2_arr = np.asarray(x2)
    k_sat_hamiltonian = np.array([0., 1., 2., 1., 0., 1., 2., 1., 1., 1., 2., 0., 1., 1., 2., 0.])
    qaoa_instance = QAOA(depth=1, H=k_sat_hamiltonian)
    expectations = np.zeros_like(x1_arr)
    try:
        for i in range(x1_arr.shape[0]):
            expectations[i] = qaoa_instance.expectation([x1_arr[i], x2_arr[i]])
    except:
        expectations = qaoa_instance.expectation([x1_arr, x2_arr])
    return expectations


def qaoa_contour(x1, x2):
    contour_values = np.zeros_like(x1)
    for i in range(len(x1)):
        contour_values[i, :] = np.log(qaoa(x1[i], x2[i]) + 1)
    return contour_values

def cmaes_animation(function, frames=100, seed=1, fpath="test.gif", restart_strategy="ipop", pop_per_frame=1, interval=20):
    fig, (ax1, ax2) = plt.subplots(1, 2)


    if function=='qaoa':
        function_name = "QAOA function"
        objective = qaoa
        contour_function = qaoa_contour
        global_minimums = [(0, 0)]  # Update with actual global minimums
        x1_lower_bound, x1_upper_bound = -np.pi, np.pi  # Update with appropriate bounds
        x2_lower_bound, x2_upper_bound = -np.pi, np.pi  # Update with appropriate bounds

    bounds = np.array([[x1_lower_bound, x1_upper_bound], [x2_lower_bound, x2_upper_bound]])
    sigma0 = (x1_upper_bound - x2_lower_bound) / 5
    optimizer = CMA(mean=np.zeros(2), sigma=sigma0, bounds=bounds, seed=seed)
    solutions = []
    trial_number = 0
    rng = np.random.RandomState(seed)

    inc_popsize = 2
    n_restarts = 0
    small_n_eval, large_n_eval = 0, 0
    popsize0 = optimizer.population_size
    poptype = "small"

    def init():
        ax1.set_xlim(x1_lower_bound, x1_upper_bound)
        ax1.set_ylim(x2_lower_bound, x2_upper_bound)
        ax2.set_xlim(x1_lower_bound, x1_upper_bound)
        ax2.set_ylim(x2_lower_bound, x2_upper_bound)

        # Plot 4 local minimum value
        for m in global_minimums:
            ax1.plot(m[0], m[1], "y*", ms=10)
            ax2.plot(m[0], m[1], "y*", ms=10)

        # Plot contour of himmelblau function
        x1 = np.arange(x1_lower_bound, x1_upper_bound, 0.05)
        x2 = np.arange(x2_lower_bound, x2_upper_bound, 0.05)
        
        x1, x2 = np.meshgrid(x1, x2)

        ax1.pcolormesh(x1, x2, contour_function(x1, x2))

    def get_next_popsize_sigma():
        nonlocal optimizer, n_restarts, poptype, small_n_eval, large_n_eval, sigma0
        if restart_strategy == "ipop":
            n_restarts += 1
            popsize = optimizer.population_size * inc_popsize
            print(f"Restart CMA-ES with popsize={popsize} at trial={trial_number}")
            return popsize, sigma0
        elif restart_strategy == "bipop":
            n_eval = optimizer.population_size * optimizer.generation
            if poptype == "small":
                small_n_eval += n_eval
            else:  # poptype == "large"
                large_n_eval += n_eval

            if small_n_eval < large_n_eval:
                poptype = "small"
                popsize_multiplier = inc_popsize ** n_restarts
                popsize = math.floor(popsize0 * popsize_multiplier ** (rng.uniform() ** 2))
                sigma = sigma0 * 10 ** (-2 * rng.uniform())
            else:
                poptype = "large"
                n_restarts += 1
                popsize = popsize0 * (inc_popsize ** n_restarts)
                sigma = sigma0
            print(
                f"Restart CMA-ES with popsize={popsize} ({poptype}) at trial={trial_number}"
            )
            return popsize, sigma
        raise Exception("Must not reach here")

    def update(frame):
        nonlocal solutions, optimizer, trial_number
        if len(solutions) == optimizer.population_size:
            optimizer.tell(solutions)
            solutions = []

            if optimizer.should_stop():
                popsize, sigma = get_next_popsize_sigma()
                lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
                mean = lower_bounds + (rng.rand(2) * (upper_bounds - lower_bounds))
                optimizer = CMA(
                    mean=mean,
                    sigma=sigma,
                    bounds=bounds,
                    seed=seed,
                    population_size=popsize,
                )

        n_sample = min(optimizer.population_size - len(solutions), pop_per_frame)
        for i in range(n_sample):
            x = optimizer.ask()
            evaluation = objective(x[0], x[1])

            # Plot sample points
            ax1.plot(x[0], x[1], "o", c="r", label="2d", alpha=0.5)

            solution = (
                x,
                evaluation,
            )
            solutions.append(solution)
        trial_number += n_sample

        # Update title
        if restart_strategy == "ipop":
            fig.suptitle(
                f"IPOP-CMA-ES {function_name} trial={trial_number} "
                f"popsize={optimizer.population_size}"
            )
        elif restart_strategy == "bipop":
            fig.suptitle(
                f"BIPOP-CMA-ES {function_name} trial={trial_number} "
                f"popsize={optimizer.population_size} ({poptype})"
            )
        else:
            fig.suptitle(f"CMA-ES {function_name} trial={trial_number}")

        # Plot multivariate gaussian distribution of CMA-ES
        x, y = np.mgrid[
            x1_lower_bound:x1_upper_bound:0.01, x2_lower_bound:x2_upper_bound:0.01
        ]
        rv = stats.multivariate_normal(optimizer._mean, optimizer._C)
        pos = np.dstack((x, y))
        ax2.contourf(x, y, rv.pdf(pos))

        if frame % 50 == 0:
            print(f"Processing frame {frame}")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=False,
        interval=interval,
    )

    writergif = animation.PillowWriter(fps=30)
    ani.save(fpath, writer=writergif)



def func_build_qaoa(n):
    k_sat_hamiltonian = np.array([0., 1., 2., 1., 0., 1., 2., 1., 1., 1., 2., 0., 1., 1., 2., 0.])
    qaoa_instance = QAOA(depth=1, H=k_sat_hamiltonian)
    a = -np.pi
    b = np.pi
    mode_size = [100] * 2

    def func(I):
        return np.array([qaoa_instance.expectation(a + I[i, :] / np.array(mode_size) * (b - a)) for i in
                         range(I.shape[0])])

    i_opt_real = None
    return func, a, b, i_opt_real

def de_animation(task, fpath=None):
    k_sat_hamiltonian = np.array([0., 1., 2., 1., 0., 1., 2., 1., 1., 1., 2., 0., 1., 1., 2., 0.])    
    qaoa_instance = QAOA(depth=1, H=k_sat_hamiltonian)
    solutions = []
    functions = []

    N = 100
    beta_array = np.linspace(0,2*np.pi,N)
    gamma_array = np.linspace(0,2*np.pi,N)
    def save_points(intermediate_result, convergence ):
        solutions.append(intermediate_result)
        functions.append(qaoa_instance.expectation(intermediate_result))

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    result = differential_evolution(qaoa_instance.expectation, bounds=[(0,2*np.pi),(0,2*np.pi)], callback=save_points)

    cost = np.zeros((len(beta_array),len(gamma_array)),dtype = float)
    land = np.dstack(np.meshgrid(beta_array, gamma_array))
    for i in range(len(beta_array)):
        for j in range(len(gamma_array)):
            cost[i,j] = qaoa_instance.expectation(land[i,j])

    # print(solutions)
    def update_plot(frame):
        if frame == 0:
            ax.imshow(cost)
        else:
            # ax.clear()
            solution = solutions[frame - 1]
            function_value = functions[frame - 1]
            ax.plot(solution[0]/ (2*np.pi/N), solution[1]/ (2*np.pi/N), color='red', marker='o', markersize=5)
            ax.set_title(f'Iteration: {frame}')


    # Save the animation as a GIF
    anim = FuncAnimation(fig, update_plot, frames=len(solutions), interval=1000)

    writergif = animation.PillowWriter(fps=10)
    anim.save(fpath, writer=writergif)

def gd_animation(task, fpath=None):
    k_sat_hamiltonian = np.array([0., 1., 2., 1., 0., 1., 2., 1., 1., 1., 2., 0., 1., 1., 2., 0.])    
    qaoa_instance = QAOA(depth=1, H=k_sat_hamiltonian)
    solutions = []
    functions = []

    N = 100
    beta_array = np.linspace(0,2*np.pi,N)
    gamma_array = np.linspace(0,2*np.pi,N)

    def save_points(x):
        solutions.append(x)
        functions.append(qaoa_instance.expectation(angles=x))
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    result = minimize(qaoa_instance.expectation, [np.pi-0.05,np.pi-0.05], callback=save_points)

    cost = np.zeros((len(beta_array),len(gamma_array)),dtype = float)
    land = np.dstack(np.meshgrid(beta_array, gamma_array))
    for i in range(len(beta_array)):
        for j in range(len(gamma_array)):
            cost[i,j] = qaoa_instance.expectation(land[i,j])

    # print(solutions)
    def update_plot(frame):
        if frame == 0:
            ax.imshow(cost)
        else:
            # ax.clear()
            solution = solutions[frame - 1]
            function_value = functions[frame - 1]
            ax.plot(solution[0]/ (2*np.pi/N), solution[1]/ (2*np.pi/N), color='red', marker='o', markersize=5)
            ax.set_title(f'Iteration: {frame}')


    # Save the animation as a GIF
    anim = FuncAnimation(fig, update_plot, frames=len(solutions), interval=1000)

    writergif = animation.PillowWriter(fps=10)
    anim.save(fpath, writer=writergif)




def animate(method, task):
    fpath = os.path.dirname(__file__) + f'/{method}_{task}.gif'
    if method == 'protes':
        if task == 'qaoa':
            n = 101
            f, a, b, i_opt_real = func_build_qaoa(np.array([n, n]))
            print("Running PROTES method for QAOA task")

            protes_animation(f, a, b, n, m=int(2.E+2), k=25, k_top=5, k_gd=10, lr=1.E-2,
                             i_opt_real=i_opt_real, fpath=fpath)

        else:
            raise NotImplementedError(f'Task name "{task}" is not supported for method "{method}"')
    elif method == 'cmaes':
        if task == 'qaoa':
            print("Running CMA-ES method for QAOA task")

            cmaes_animation(task, fpath=fpath)
        else:
            raise NotImplementedError(f'Task name "{task}" is not supported for method "{method}"')
    elif method == 'de':
        if task == 'qaoa':
            print("Running differential evolution method for QAOA task")

            de_animation(task, fpath=fpath)
        else:
            raise NotImplementedError(f'Task name "{task}" is not supported for method "{method}"')
        
    elif method == 'gd':
        if task == 'qaoa':
            print("Running gradient descent method for QAOA task")

            gd_animation(task, fpath=fpath)
        else:
            raise NotImplementedError(f'Task name "{task}" is not supported for method "{method}"')
        

    else:
        raise NotImplementedError(f'Method "{method}" is not supported')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["protes", "cmaes", "minimize", "de", "gd"])
    parser.add_argument("task", choices=["qaoa"])
    args = parser.parse_args()
    animate(args.method, args.task)


if __name__ == "__main__":
    main()

#TODO change reference to src