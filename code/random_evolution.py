import gc
import os

import psutil
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

import logging


def random_evolution(func, bounds, args=(), maxiter=1000, popsize=15):
    """Tries random perturbations at each iteration. Used as the baseline to compare against the evolutionary algorithms
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    maxiter : int, optional
        The maximum number of iterations.
    popsize : int, optional
        The size of the population.
    Returns
    -------
    res : RandomSolverResult
        The random evolution result represented a tuple which represents the (attack success, perturbation)
        In the case we have attack_success = True then the perturbation
        is the one for which we have a successful attack, or random vector otherwise.
    """

    solver = RandomSolver(func, bounds, args=args, maxiter=maxiter,
                          popsize=popsize)
    return solver.solve()


class RandomSolver(object):
    """This class implements the random evolution solver
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    maxiter : int, optional
        The maximum number of iterations.
    popsize : int, optional
        The size of the population.
    """

    def __init__(self, func, bounds, args=(), maxiter=1000, popsize=15):

        logging.info("Start the initialization")
        # relative and absolute tolerances for convergence
        # self.tol, self.atol = tol, atol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = bounds.T
        self.parameter_size = self.limits.size(1)
        # print(parameters_size)
        if self.limits.size(0) != 2:
            raise ValueError('bounds should be a tensor containing real valued (min, max) pairs for each value in x')
        self.maxiter = maxiter

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1]).to(device)
        self.__scale_arg2 = torch.abs(self.limits[0] - self.limits[1]).to(device)

        self.num_population_members = popsize

        self.population_shape = (self.num_population_members,
                                 self.parameter_size)

        logging.info("Finish the initialization of the RE")

    def solve(self):
        """
        Runs the RandomEvolutionSolver.
        Returns
        -------
        res : a tuple which represents the (attack success, perturbation)
            In the case we have attack_success = True then the perturbation
            is the one for which we have a successful attack, or random vector otherwise.
        """
        logging.info("Start to solve")

        for nit in range(0, self.maxiter):
            result, vector = self._calculate_population_energies()
            logging.info('RE step {}'.format(nit))
            gc.collect()
            torch.cuda.empty_cache()
            if result:
                logging.info("Finding randomly a an agent")
                logging.info(result)
                return True, vector
            if nit == self.maxiter-1:
                logging.info("Finalising with the max iterations")
                logging.info(result)
                return False, vector

    def _calculate_population_energies(self):
        """
        Calculate the energies of all the population members at the same time.
        Puts the best member in first place. Useful if the population has just
        been initialised.
        """

        ##############
        ## CHANGES: self.func operates on the entire parameters array
        ##############
        # logging.info('Calculate the population energy - starting function')
        self.population = torch.rand(self.population_shape, dtype=torch.float32, device=device)
        parameters = self.population.clone()
        # logging.info("Checking for the constraints of the nre trials")
        for i in range(len(parameters)):
            parameters[i] = self._ensure_constraint(parameters[i])
        # logging.info('Scale the parameters')
        parameters = self._scale_parameters(parameters)
        result, index = self.func(parameters, *self.args)
        if result:
            return result, self._scale_parameters(self.population[index])
        else:
            return result, self._scale_parameters(self.population[index])

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """
        make sure the parameters lie between the limits
        """
        condition = (trial < 0) | (trial > 1).to(device)
        # Use torch.where() to get the indices where the condition is True
        indices = torch.where(condition)
        for index in indices[0]:
            trial[index] = torch.rand(1).item()
        return trial

    def check_memory(self):
        # memory_info = psutil.virtual_memory()
        # total_memory_gb = memory_info.total / (1024 ** 3)  # 1 GB = 1024^3 bytes
        # available_memory_gb = memory_info.available / (1024 ** 3)
        # used_memory_gb = memory_info.used / (1024 ** 3)
        # logging.info(f"Total memory: {total_memory_gb:.2f} GB")
        # logging.info(f"Available memory: {available_memory_gb:.2f} GB")
        # logging.info(f"Used memory: {used_memory_gb:.2f} GB")
        pid = os.getpid()
        # Get the process information
        process = psutil.Process(pid)
        # Get memory usage information
        memory_info = process.memory_info()
        # Convert memory usage to gigabytes
        # memory_usage_gb = memory_info.rss / (1024 ** 3)  # Resident Set Size (RSS)
        # logging.critical(f"CPU Memory Usage: {memory_usage_gb:.2f} GB")
        logging.critical(f"CPU Memory Usage: {memory_info.rss} bytes`")
        # Check GPU memory usage
        # allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to megabytes
        # cached_memory = torch.cuda.memory_cached(device) / (1024 ** 2)  # Convert to megabytes
        #
        # logging.critical(f"Allocated GPU Memory: {allocated_memory:.2f} MB")
        # logging.critical(f"Cached GPU Memory: {cached_memory:.2f} MB")

    def check_gpu_variable(self, x):
        if x.is_cuda:
            logging.info("Variable is on GPU")
        else:
            logging.info("Variable is on CPU")
