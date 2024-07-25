"""
A slight modification to Scipy's implementation of differential evolution. To speed up predictions, the entire parameters array is passed to `self.func`, where a neural network model can batch its computations.

Taken from
https://github.com/scipy/scipy/blob/70e61dee181de23fdd8d893eaa9491100e2218d7/scipy/optimize/_differentialevolution.py

----------

differential_evolution: The differential evolution global optimization algorithm
Original Added by Andrew Nelson 2014
Modified for the research purpose using only PyTorch for optimization on the GPU and speed up the computation
"""
import gc
import os
import random

import psutil
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from torch.optim import LBFGS

import logging


def differential_evolution(func, bounds: list, init_population: torch.Tensor, callback_fn, args=(),
                           strategy: str = 'best1bin', maxiter: int = 1000, popsize: int = 15, mutation: float = 0.8,
                           crossover: float = 0.7, polish: bool = False, max_iter_lbfgs: int = 10, stats: bool = False,
                           random_vec: list = []) -> object:
    """Finds the global minimum of a multivariate function.
    Differential Evolution is stochastic in nature (does not use gradient
    methods) to find the minimium, and can search large areas of candidate
    space, but often requires larger numbers of function evaluations than
    conventional gradient based techniques.
    The algorithm is due to Storn and Price [1]_.
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
    init_population : torch.Tensor
        Tensor which is on GPU (in case it is used) and represents the population made from the initialization process.
    callback_fn : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals (unless the initial population is
        supplied via the `init` keyword).
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        ``U[min, max)``. Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    crossover : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly.
    max_iter_lbfgs: int, optional
        The maximum number of iterations if the `L-BFGS-B` is used.
    stats: bool, optional
        If True, then the progress of the fitness function will be stored, otherwise no. The default value is False.
    random_vec: list, optional
        1.This list represents the indices for which function to be used in case of the fitness functions,
        where at each iteration they use another function. (i.e. F8, F9).
        2. This list represents the indices which have to be used in specific fitness functions (F10, F11, F12).
    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        It contains the most successful image pertubration encoding (based agent) and the list of stats
        (empty list if the stats are not required).
    Notes
    -----
    Differential evolution is a stochastic population based method that is
    useful for global optimization problems. At each pass through the population
    the algorithm mutates each candidate solution by mixing with other candidate
    solutions to create a trial candidate. There are several strategies [2]_ for
    creating trial candidates, which suit some problems more than others. The
    'best1bin' strategy is a good starting point for many systems. In this
    strategy two members of the population are randomly chosen. Their difference
    is used to mutate the best member (the `best` in `best1bin`), :math:`b_0`,
    so far:
    .. math::
        b' = b_0 + mutation * (population[rand0] - population[rand1])
    A trial vector is then constructed. Starting with a randomly chosen 'i'th
    parameter the trial is sequentially filled (in modulo) with parameters from
    `b'` or the original candidate. The choice of whether to use `b'` or the
    original candidate is made with a binomial distribution (the 'bin' in
    'best1bin') - a random number in [0, 1) is generated.  If this number is
    less than the `recombination` constant then the parameter is loaded from
    `b'`, otherwise it is loaded from the original candidate.  The final
    parameter is always loaded from `b'`.  Once the trial candidate is built
    its fitness is assessed. If the trial is better than the original candidate
    then it takes its place. If it is also better than the best overall
    candidate it also replaces that.
    To improve your chances of finding a global minimum use higher `popsize`
    values, with higher `mutation` and (dithering), but lower `recombination`
    values. This has the effect of widening the search radius, but slowing
    convergence.
    .. versionadded:: 0.15.0
    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`.

    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)
    Next find the minimum of the Ackley function
    (http://en.wikipedia.org/wiki/Test_functions_for_optimization).

    (array([ 0.,  0.]), 4.4408920985006262e-16)
    References
    ----------
    .. [1] Storn, R and Price, K, Differential Evolution - a Simple and
           Efficient Heuristic for Global Optimization over Continuous Spaces,
           Journal of Global Optimization, 1997, 11, 341 - 359.
    .. [2] http://www1.icsi.berkeley.edu/~storn/code.html
    .. [3] http://en.wikipedia.org/wiki/Differential_evolution
    """

    solver = DifferentialEvolutionSolver(func, bounds, args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize,
                                         mutation=mutation,
                                         crossover=crossover,
                                         polish=polish, max_iter_lbfgs=max_iter_lbfgs,
                                         callback_fn=callback_fn, init_population=init_population, stats=stats,
                                         random_vec=random_vec)
    return solver.solve()


class DifferentialEvolutionSolver(object):
    """This class implements the differential evolution solver
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
    init_population : torch.Tensor
        Tensor which is on GPU (in case it is used) and represents the population made from the initialization process.
    callback_fn : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals (unless the initial population is
        supplied via the `init` keyword).
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        ``U[min, max)``. Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    crossover : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly.
    max_iter_lbfgs: int, optional
        The maximum number of iterations if the `L-BFGS-B` is used.
    stats: bool, optional
        If True, then the progress of the fitness function will be stored, otherwise no. The default value is False.
    random_vec: list, optional
        1.This list represents the indices for which function to be used in case of the fitness functions,
        where at each iteration they use another function. (i.e. F8, F9).
        2. This list represents the indices which have to be used in specific fitness functions (F10, F11, F12).
    all_energies: list
        The list which contains all the energies for each iteration step. Used if we want to store the stats.

    """

    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'currenttobest1bin': '_currenttobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'currenttobest1exp': '_currenttobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    __init_error_msg = ("The population initialization method must be one of "
                        "'latinhypercube' or 'random', or an array of shape "
                        "(M, N) where N is the number of parameters and M>5")

    def __init__(self, func, bounds, callback_fn, init_population, args=(),
                 strategy='best1bin', maxiter=1000, popsize=15, mutation=0.5, crossover=0.7, polish=False,
                 max_iter_lbfgs=None, stats=False, random_vec=None):

        if random_vec is None:
            random_vec = []
        self.stats = stats
        self.all_energies = []
        self.iter = 0

        logging.info("Start the initialization of the DE")
        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy
        self.callback_fn = callback_fn
        self.polish = polish
        if self.polish:
            self.max_iter_lbfgs = max_iter_lbfgs
            if self.max_iter_lbfgs == None:
                raise ValueError("Please select a valid max_iter_lbfgs number")

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.mutation = mutation
        if mutation > 2 or mutation < 0:
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2)')

        self.crossover = crossover

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
        self.random_vec = random_vec

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        # the minimum is 5 because 'best2bin' requires a population that's at
        # least 5 long
        # self.num_population_members = max(popsize * self.parameter_size)
        self.num_population_members = popsize

        self.population_shape = (self.num_population_members,
                                 self.parameter_size)

        self.population = torch.clamp(self._unscale_parameters(init_population), 0, 1).to(device)

        self.best_energy = torch.tensor(float('inf'), device=device)
        self.best_agent = torch.full((self.num_population_members,), float(-1), device=device)
        logging.info("Finish the initialization of the DE")

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            It contains the most successful image pertubration encoding (based agent) and the list of stats
            (empty list if the stats are not required).
        """
        logging.info("Start to solve")

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).

        # do the optimisation.
        self._calculate_population_energies()
        if self.stats:
            s = self.population_energies.clone()
            s = s.detach().cpu()
            self.all_energies.append(s)

        for nit in range(1, self.maxiter + 1):
            # evolve the population by a generation
            logging.info('DE step {}'.format(nit))
            self.iter = nit
            try:
                next(self)
                gc.collect()
                torch.cuda.empty_cache()
            except StopIteration:
                logging.info("status msg - early stop")
                gc.collect()
                torch.cuda.empty_cache()
                break

        result, vector = self.stopping_condition()

        logging.info("Finalising with the mutation iterations")
        logging.info(result)

        if self.polish:
            # Ensure that initial_params is on GPU
            logging.info("Starting the LBFGSB function optimization")
            init_vector = vector.clone()
            init_vector = init_vector.to(device)
            init_vector = self._unscale_parameters(init_vector)
            init_vector.requires_grad = True
            self.best_agent = self.lbfgs_optimization_gpu(init_vector, max_iterations=self.max_iter_lbfgs).to(device)
            vector = self._scale_parameters(self.best_agent)
            logging.info("Finalizing the LBFGSB function optimization")

        return vector, self.all_energies

    def _calculate_population_energies(self):
        """
        Calculates the energies of all the population members at the same time.
        Puts the best member in first place. Useful if the population has just
        been initialised.
        """
        parameters = self.population.clone()
        for i in range(len(parameters)):
            parameters[i] = self._ensure_constraint(parameters[i])
        parameters = self._scale_parameters(parameters)

        # For the functions which involves randomness
        rv = 0
        if len(self.random_vec) == 100:
            # We do this because it is only the first iteration so I can do the random value 0.
            rv = 0
        elif len(self.random_vec) > 1:
            rv = random.choice(self.random_vec)
        self.population_energies = self.func(parameters, rv, *self.args).to(device)
        # logging.debug("I want to check the energies pls")
        # logging.debug(self.population_energies)
        logging.debug("I want to check the energies pls")
        logging.debug(self.population_energies)

        max_value, max_index = torch.max(self.population_energies, dim=0)
        self.best_energy = max_value.item()
        self.best_agent = self._scale_parameters(self.population[max_index.item()].to(device))
        logging.debug('Calculate the population energy - final function')

    def stopping_condition(self):
        """
        The stopping condition for the DE. Check if it is the case (True).
        Returns also the best agent encoding
        """
        if self.callback_fn(self.best_agent) == True:
            logging.debug('I want to print the best energy')
            logging.debug(self.best_energy)
            return True, self.best_agent
        else:
            return False, self.best_agent

    def __iter__(self):
        return self

    def __next__(self):
        """
        Evolves the population by a single generation.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        # logging.info("----------------------")
        # logging.info("----------------------")
        # logging.info("----------------------")
        # logging.info("starting the iteration")
        # logging.info("----------------------")
        # self.check_memory()

        result, _ = self.stopping_condition()
        if result:
            raise StopIteration

        tensor_trials = [torch.rand(self.parameter_size, device=device) for _ in range(self.num_population_members)]
        trials = torch.stack(tensor_trials)
        for c in range(self.num_population_members):
            trials[c] = self._mutate(c)
        gc.collect()
        for i in range(len(trials)):
            trials[i] = self._ensure_constraint(trials[i])
        parameters = trials.clone()
        parameters = self._scale_parameters(parameters)

        # choosing a random value for the fitness functions which involves randomness
        rv = 0
        if len(self.random_vec) == 100:
            rv = self.random_vec[self.iter - 1]
        elif len(self.random_vec) > 1:
            rv = random.choice(self.random_vec)
        energies = self.func(parameters, rv, *self.args).to(device).clone()
        energies_total = torch.cat((energies, self.population_energies))
        population_total = torch.cat((trials, self.population))

        sorted_indices = torch.argsort(energies_total, descending=True)
        sorted_indices = sorted_indices[:self.num_population_members]
        self.population = population_total[sorted_indices]
        self.population_energies = energies_total[sorted_indices]
        self.best_energy = self.population_energies[0]
        self.best_agent = self._scale_parameters(self.population[0])

        logging.debug("I want to check the energies pls")
        logging.debug(self.population_energies)

        # logging.info('Randomize the energies for a better solution')
        shuffled_indices = torch.randperm(self.population_energies.size(0)).to(device)
        self.population_energies = self.population_energies[shuffled_indices]

        if self.stats:
            s = self.population_energies.clone()
            s = s.detach().cpu()
            self.all_energies.append(s)

    def next(self):
        """
        Evolves the population by a single generation.
        """
        # next() is required for compatibility with Python2.7.
        return self.__next__()

    def _scale_parameters(self, trial):
        """
        Scales from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        Scales from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """
        Makes sure the parameters lie between the limits.
        """
        condition = (trial < 0) | (trial > 1).to(device)
        # Use torch.where() to get the indices where the condition is True
        indices = torch.where(condition)
        for index in indices[0]:
            trial[index] = torch.rand(1).item()
        return trial

    def _mutate(self, candidate):
        """
        Creates a trial vector based on a mutation strategy.
        """
        # self.check_memory()
        # self.check_gpu_variable(self.population[candidate])
        trial = self.population[candidate].detach().clone()
        # self.check_memory()
        # self.check_gpu_variable(trial)

        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate,
                                        self._select_samples(candidate, 5)).to(device)
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5)).to(device)

        big_r = torch.randint(self.parameter_size, (1,)).to(device)
        # print(big_r)

        if self.strategy in self._binomial:
            crossover_mask = (torch.rand(self.parameter_size, device=device) < self.crossover)
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossover_mask[big_r] = True
            trial = torch.where(crossover_mask, bprime, trial)

            return trial

        elif self.strategy in self._exponential:
            i = 0
            while (i < self.parameter_size):
                r = torch.rand(1)
                if self.crossover < r:
                    break
                trial[big_r] = bprime[big_r]
                big_r = (big_r + 1) % self.parameter_size
                i += 1
            return trial

    def _best1(self, samples):
        """
        best1bin, best1exp
        """
        r0, r1 = samples[:2]
        return (self.population[0] + self.mutation *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """
        rand1bin, rand1exp
        """
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.mutation *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, samples):
        """
        randtobest1bin, randtobest1exp
        """
        r0, r1, r2 = samples[:3]
        bprime = self.population[r0].clone().to(device)
        bprime += self.mutation * (self.population[0] - bprime)
        bprime += self.mutation * (self.population[r1] -
                                   self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """
        currenttobest1bin, currenttobest1exp
        """
        r0, r1 = samples[:2]
        bprime = (self.population[candidate] + self.mutation *
                  (self.population[0] - self.population[candidate] +
                   self.population[r0] - self.population[r1]))
        return bprime

    def _best2(self, samples):
        """
        best2bin, best2exp
        """
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.mutation *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """
        rand2bin, rand2exp
        """
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.mutation *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate_index, number_samples):
        """
        Obtain random integers from range(self.num_population_members),
        without replacement.  You can't have the original candidate either.
        """
        idxs = torch.randint(self.num_population_members, (number_samples,))
        while True:
            if torch.any(idxs == candidate_index):
                idxs = torch.randint(self.num_population_members, (number_samples,))
            else:
                return idxs.to(device)

    def lbfgs_optimization_gpu(self, initial_params, max_iterations=100):
        """
        Perform LBFGS optimization using PyTorch on GPU with a lambda function.

        Args:
            initial_params (Tensor, on GPU): The initial parameter tensor on GPU.
            max_iterations (int, optional): Maximum number of iterations.

        Returns:
            Tensor: The optimized parameters on GPU.
        """

        # Define the optimization problem using LBFGS optimizer
        optimizer = LBFGS([initial_params])
        iteration = 0

        # Optimization loop
        def closure():
            optimizer.zero_grad()
            loss = self.func(self._scale_parameters(initial_params))
            loss.requires_grad = True
            loss.backward()
            return loss

        while iteration < max_iterations:
            optimizer.step(closure)
            initial_params.data.clamp_(0, 1)
            iteration += 1

        # Return the optimized parameters (still on GPU)
        return initial_params

    # These functions are used for debugging purposes and to not have a memory leakage.
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

    # This function is used for debugging to check if a variable is stored on a GPU or not
    def check_gpu_variable(self, x):
        if x.is_cuda:
            logging.info("Variable is on GPU")
        else:
            logging.info("Variable is on CPU")
