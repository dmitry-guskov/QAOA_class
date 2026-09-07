"""
QAOA Implementation for Ising Hamiltonian, k-SAT Hamiltonian and maxCUT Hamiltonian

This script provides an implementation of the Quantum Approximate Optimization Algorithm (QAOA) to solve optimization problems.
It includes functions to construct maxCUT Hamiltonian, Ising Hamiltonian, k-SAT Hamiltonian, QAOA ansatz, and optimization methods.

Credits:
- Main contributors: Ernesto Luis Campos Espinoza, Akshay Vishwanathan
- Additional contributors: Andrey Kardashin 
- Assembled by Dmitry Guskov
© 2023 Deep Quantum Lab. All rights reserved.
"""


import numpy as np
import random
import time
from functools import reduce
from typing import List
from numpy import ndarray

from scipy.optimize import differential_evolution
from scipy.optimize import minimize


def w_operator (x, z) :
    ''' generates the Walsh operator

    Args:
        x (_type_): _description_
        z (_type_): _description_

    Returns:
        _type_: _description_
    '''
    PAULIS = {'I': np.eye(2),
          'X': np.array([[0, 1], [1, 0]]),
          'Y': np.array([[0, -1j], [1j, 0]]),
          'Z': np.diag([1, -1]),
          'H': (1/np.sqrt(2))*np.array([[1, 1], [1, -1]]),
          'XZ': np.array([[0,-1],[1,0]])}
    
    ans_string = []
    for i in range(len(x)):
        if x[i] and z[i]:
            ans_string.append('XZ')
        elif x[i] and not z[i]:
            ans_string.append('X')
        elif not x[i] and  z[i]:
            ans_string.append('Z')
        elif not x[i] and not z[i]:
            ans_string.append('I')
    ans = reduce(np.kron, [PAULIS[s] for s in ans_string])
    coef = (1j)**np.sum(np.multiply(x,z))
    return ans * coef


def Graph_to_Hamiltonian(G):
    """Converts an adjacency matrix into a Hamiltonian instance assuming connectivity is a ZZ gate.

    Args:
        G (list or ndarray): Adjacency matrix representing the graph.
    """    
    def tensor(k):
        t = k[0]
        i = 1

        while i < len(k):
            t = np.kron(t, k[i])
            i += 1
        return t
    
    if isinstance(G, ndarray):
        n = G.shape[0]
    else:
        n = len(G)
    
    H = np.zeros((2**n), dtype='float64')
    Z = np.array([1, -1], dtype='float64')

    for i in range(n):
        j = i + 1

        while j < n:
            k = [[1, 1]] * n
            k = np.array(k, dtype='float64')

            if G[i][j] != 0:
                k[i] = Z
                k[j] = Z
                H += tensor(k) * G[i][j]

            j += 1

    return H


def H_zz_Ising(n_qubits, bc="closed"):
    """Constructs the Hamiltonian for the ZZ model (simpliest Ising example).

    Args:
        n_qubits (int): Number of qubits representing the system.
        bc (str, optional): Boundary condition. Defaults to "closed".

    Returns:
        ndarray: Diagonal elements of the Hamiltonian.
    """    
    if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1 or bc not in ("open", "closed"):
        raise ValueError("Expected a positive qubit count and open/closed boundary.")
    # Periodic indexed bonds: n=1 gives I; n=2 counts the bond twice.
    if n_qubits == 1:
        return np.ones(2) if bc == "closed" else np.zeros(2)
    # Constructs the Ising Hamiltonian
    Z = np.array([1., -1.])
    I = np.array([1., 1.])
    
    Hzz = np.zeros(2**n_qubits)
    for q in range(n_qubits - 1):
        Hzz = Hzz + reduce(np.kron, [I]*q + [Z, Z] + [I]*(n_qubits-q-2))
    if bc == "closed":
        Hzz = Hzz + reduce(np.kron, [Z] + [I]*(n_qubits-2) + [Z])

    return Hzz


def H_sat(n, k, alpha):
    """Constructs the Hamiltonian for k-SAT problems from random clauses.

    Args:
        n (int): Number of variables.
        k (int): Number of literals in each clause.
        alpha (float): Scaling factor for the number of clauses.

    Returns:
        ndarray: Diagonal elements of the Hamiltonian for k-SAT.
    """    
    I_prime = np.array([1, 1])
    rho_0_prime = np.array([1, 0])
    rho_1_prime = np.array([0, 1])

    h = np.zeros(2**n)

    for i in range(int(alpha*n)):
        t = np.random.choice(n, n - k, replace=False)  # Positions of I matrix
        C_prime = 1
        for j in range(n):
            if j in t:
                C_prime = np.kron(C_prime, I_prime)
            else:
                if np.random.randint(2) < 1:
                    C_prime = np.kron(C_prime, rho_0_prime)
                else:
                    C_prime = np.kron(C_prime, rho_1_prime)
        h += C_prime

    return h


def plus_state(n_qubits):
    """Generates a |+⟩ state to the tensor power of n_qubits.

    Args:
        n_qubits (int): Number of qubits.

    Returns:
        ndarray: Superposition state |+⟩^n_qubits.
    """    
    d = 2**n_qubits
    return np.array([1/np.sqrt(d)]*d, dtype='complex128')  # |+⟩

def fidelity(state1, state2):
    '''Calculates fidelity between two pure states

    Args:
        state1 (ndarray): quatnum state
        state2 (ndarray): quantum state
    '''
    F = np.real(np.vdot(state1, state2) *np.vdot(state2, state1))
    return F

def create_depolarization_kraus(p_depolarization):
    """Return Pauli operators and probabilities for rho -> (1-p)rho+p I/2."""
    p = float(p_depolarization)
    if not 0 <= p <= 1:
        raise ValueError("Depolarization probability must be between zero and one.")
    return [np.eye(2), np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])], [1-3*p/4, p/4, p/4, p/4]


def create_amplitude_damping_kraus(p_amplitude_damping):
    """Return standard Kraus operators and None (Born-rule branch probabilities).

    Passing this pair to apply_ansatz_noise handles state-dependent sampling.
    """
    p = float(p_amplitude_damping)
    if not 0 <= p <= 1:
        raise ValueError("Damping probability must be between zero and one.")
    return [np.diag([1, np.sqrt(1-p)]), np.array([[0, np.sqrt(p)], [0, 0]])], None


def create_phase_flip_kraus(p_phase_flip):
    """Return I/Z operators and the actual phase-flip probabilities."""
    p = float(p_phase_flip)
    if not 0 <= p <= 1:
        raise ValueError("Phase-flip probability must be between zero and one.")
    return [np.eye(2), np.diag([1, -1])], [1-p, p]


def _channel_kraus(noise_prob, kraus_ops):
    """Convert weighted operators or standard Kraus operators to a CPTP channel."""
    ops = np.asarray(kraus_ops, dtype=complex)
    if ops.ndim != 3 or ops.shape[1:] != (2, 2) or len(ops) == 0 or not np.isfinite(ops).all():
        raise ValueError("Expected a nonempty collection of finite 2x2 operators.")
    if noise_prob is not None:
        probabilities = np.asarray(noise_prob, dtype=float)
        if (probabilities.shape != (len(ops),) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError("Noise probabilities must be nonnegative and sum to one.")
        ops = np.sqrt(probabilities)[:, None, None] * ops
    completeness = sum(op.conj().T @ op for op in ops)
    if not np.allclose(completeness, np.eye(2), rtol=1e-10, atol=1e-12):
        raise ValueError("The supplied operators do not define a trace-preserving channel.")
    return ops


def _local_channel_trajectory(state, ops, qubit, n_qubits, rng):
    """Apply a one-qubit channel to a normalized state; qubit zero is leftmost."""
    tensor = np.moveaxis(state.reshape((2,) * n_qubits), qubit, 0)
    branches = [np.moveaxis((op @ tensor.reshape(2, -1)).reshape(tensor.shape),
                           0, qubit).reshape(-1) for op in ops]
    probabilities = np.array([np.vdot(branch, branch).real for branch in branches])
    index = rng.choice(len(branches), p=probabilities / probabilities.sum())
    return branches[index] / np.sqrt(probabilities[index])



##################################################################################################
##################################################################################################        
#####------------------------ MAIN     QAOA     CLASS ---------------------------------------#####
##################################################################################################    
##################################################################################################


class QAOA:
    """ Encapsulates the Quantum Approximate Optimization Algorithm (QAOA) logic.
    """    
    def __init__(self, depth: int, H: ndarray):
        """
        Initialize the QAOA class.
        Args:
            depth (int): QAOA depth.
            H (ndarray): Diagonal Hamiltonian.
        """
        diagonal = np.asarray(H)
        if (diagonal.ndim != 1 or len(diagonal) < 2 or len(diagonal) & (len(diagonal)-1)
                or not np.isrealobj(diagonal) or not np.isfinite(diagonal).all()):
            raise ValueError("H must be a finite real power-of-two diagonal with at least one qubit.")
        if not isinstance(depth, (int, np.integer)) or depth < 1:
            raise ValueError("depth must be a positive integer.")
        self.H = diagonal.astype(float, copy=True)
        self.n_qubits = int(np.log2(len(self.H)))
        self.x_list = self.mixer_list()
        self.min = min(self.H)
        self.deg = len(self.H[self.H == self.min])
        self.p = depth

        self.opt_angles = None
        self.exe_time = None
        self.opt_iter = None
        self.q_energy = None
        self.q_error = None
        self.f_state = None
        self.olap = None
        self.log = None
        self.eval_num = 0
        self.track_eval = True
        self.track_cost = False
        self.tracked_cost = []
        self.protes_log = None


    def reset(self):

        self.opt_angles = None
        self.exe_time = None
        self.opt_iter = None
        self.q_energy = None
        self.q_error = None
        self.f_state = None
        self.olap = None
        self.log = None
        self.eval_num = 0
        self.tracked_cost = []
        self.protes_log = None
        self.lw_log = None

    def mixer_list(self):
        """Indices for single-qubit X flips, with qubit zero most significant."""
        indices = np.arange(2**self.n_qubits)
        return [indices ^ (1 << (self.n_qubits-1-q)) for q in range(self.n_qubits)]

    def apply_gamma(self, gamma: float, statevector: ndarray) -> ndarray:
        """Applies the gamma operator to the state vector.

        Args:
            gamma (float): _description_
            statevector (ndarray): _description_

        Returns:
            ndarray: _description_
        """        
        return  np.exp(-1j * gamma * self.H) * statevector

    def apply_Hx(self, statevector: ndarray) -> ndarray:
        """Applies the Hx operator to the state vector.
            used in qaoa_qfi_matrix
        Args:
            statevector (ndarray): _description_

        Returns:
            ndarray: _description_
        """        
        x_list = self.x_list
        n_qubits = self.n_qubits

        statevector_new = np.zeros(len(statevector), dtype=complex)
        for i in range(n_qubits):
            statevector_swap = statevector[x_list[i]]
            statevector_new = statevector_swap + statevector_new
        return statevector_new

    def apply_beta(self, beta: float, statevector: ndarray) -> ndarray:
        """Applies the beta operator to the state vector.
            stands for mixer
        Args:
            beta (float): _description_
            statevector (ndarray): _description_

        Returns:
            ndarray: _description_
        """        
        x_list = self.x_list
        n_qubits = self.n_qubits
        c = np.cos(beta)
        s = np.sin(beta)
        statevector_new = np.copy(statevector)
        for i in range(n_qubits):
            statevector_swap = statevector_new[x_list[i]]
            statevector_new = -1j * s * statevector_swap + c * statevector_new
        return statevector_new
    
    def qaoa_ansatz(self, angles: List[float]) -> ndarray:
        """ Generates the QAOA ansatz state for a given set of angles.
            # NOTE this is the same as apply_ansatz(angles, plus_state)
        Args:
            angles (List[float]): _description_

        Returns:
            ndarray: _description_
        """        
        state = plus_state(self.n_qubits)

        if len(angles) % 2 != 0:
            raise ValueError("Number of angles must be even.")
        p = len(angles) // 2
        
        for i in range(p):
            state = self.apply_gamma(angles[i], state)
            state = self.apply_beta(angles[p + i], state)
        return state

    def apply_ansatz(self, angles: List[float], state: ndarray) -> ndarray:
        """Applies the QAOA ansatz to the state vector.

        Args:
            angles (List[float]): _description_
            state (ndarray): _description_

        Returns:
            ndarray: _description_
        """        
        if len(angles) % 2 != 0:
            raise ValueError("Number of angles must be even.")
        
        p = len(angles) // 2
        for i in range(p):
            state = self.apply_gamma(angles[i], state)
            state = self.apply_beta(angles[p + i], state)
        return state

    def expectation(self, angles: List[float]) -> float:
        """Computes the expectation value of the Hamiltonian for a set of angles.

        Args:
            angles (List[float]): _description_

        Returns:
            float: _description_
        """        
        state = self.qaoa_ansatz(angles)
        ex = np.real(np.vdot(state, state * self.H))
        # count this step as evaluation 
        if self.track_eval:
            self.eval_num += 1
        if self.track_cost:
            self.tracked_cost.append(ex)
        
        return ex

    def qaoa_operator(self,angles: List[float]):
        '''Generates QAOA operator, e.i. U(theta), so (plus_state(n_qubits)@op)@np.diag(hamiltonian)@(plus_state(n_qubits)@op).T.conj() equivalent to call expectation 

        Args:
            angles (List[float]): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        '''
        if len(angles) % 2 != 0:
            raise ValueError("Number of angles must be even.")
        state = np.zeros(2**self.n_qubits)
        state[0] = 1
        ans =  np.outer(state,self.apply_ansatz(angles, state))
        state[0] = 0

        for i in range(1,len(state)):
            state[i] = 1
            ans += np.outer(state,self.apply_ansatz(angles, state))
            state[i] = 0

        return ans
    
    def construct_QAOA_operator_term(self, angles: List[float], part_hamiltonian=None):
        """Return U^dagger diag(part_hamiltonian) U in the Heisenberg picture.

        qaoa_operator retains its historical row-vector convention and returns
        U.T. A zero observable is valid and must not be replaced by self.H.
        """
        if len(angles) % 2:
            raise ValueError("Number of angles must be even.")
        diagonal = self.H if part_hamiltonian is None else np.asarray(part_hamiltonian)
        if diagonal.shape != self.H.shape:
            raise ValueError("Expected one observable diagonal entry per basis state.")
        row_operator = self.qaoa_operator(angles)
        return row_operator.conj() @ np.diag(diagonal) @ row_operator.T

    def apply_ansatz_noise(self, angles, noise_prob, kraus_ops, rng=None):
        """Sample one normalized trajectory, with local noise after each layer.

        Angles are [gamma_1,...,gamma_p,beta_1,...,beta_p]. noise_prob=None
        means standard Kraus operators; otherwise weights multiply operators
        by sqrt(probability). Branches always use their state-dependent norms.
        rng may be a NumPy Generator; by default np.random.seed is respected.
        """
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (2*self.p,) or not np.isfinite(angles).all():
            raise ValueError("Expected exactly 2*p finite angles.")
        ops = _channel_kraus(noise_prob, kraus_ops)
        rng = np.random if rng is None else rng
        state = plus_state(self.n_qubits)
        for i in range(self.p):
            state = self.apply_gamma(angles[i], state)
            state = self.apply_beta(angles[self.p+i], state)
            for qubit in range(self.n_qubits):
                state = _local_channel_trajectory(state, ops, qubit, self.n_qubits, rng)
        return state

    def expectation_noise(self, angles, noise_prob, kraus_ops, num_samples, return_state=False, rng=None):
        """Average trajectory energies, never coherent trajectory amplitudes.

        If return_state=True, the second result is the empirical density matrix,
        not a statevector. Noisy output generally represents a mixed state.
        """
        if not isinstance(num_samples, (int, np.integer)) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer.")
        energy = 0.0
        density = np.zeros((len(self.H), len(self.H)), complex) if return_state else None
        for _ in range(num_samples):
            state = self.apply_ansatz_noise(angles, noise_prob, kraus_ops, rng=rng)
            energy += np.vdot(state, self.H * state).real
            if return_state:
                density += np.outer(state, state.conj())
        energy = float(energy / num_samples)
        if return_state:
            return energy, density / num_samples
        return energy

    def finite_diff_grad(self, angles: List[float], delta=1e-3) -> ndarray:
        """Computes the approximation value of the expectation functions gradient for a set of angles.

        Args:
            angles (List[float]): _description_

        Returns:
            ndarray: _description_
        """        
        n_params = len(angles)
        output = np.zeros(n_params)

        for i in range(n_params):
            angles[i] += delta
            f1 = self.expectation(angles)
            angles[i] -= 2*delta
            f2 = self.expectation(angles)
            angles[i] += delta

            output[i] = np.real((f1 - f2)/(2 * delta))

        return output
    

    def overlap(self, state: ndarray) -> float:
        """Calculates the overlap of a state with the ground state.

        Args:
            state (ndarray): _description_

        Returns:
            float: _description_
        """        
        g_ener = min(self.H)
        olap = 0
        for i in range(len(self.H)):
            if self.H[i] == g_ener:
                olap += np.absolute(state[i])**2
        return olap
    


##################################################################################################
##################################################################################################        
#####------------------------ OPTIMIZATION STRATEGIES ---------------------------------------#####
##################################################################################################    
##################################################################################################



    def run(self, track_energy=False, initial_params=None):
        """Runs the QAOA using the heuristic L-BFGS-B optimization method
            
        Returns:
            _type_: _description_
        """           
        if initial_params is None:
            if self.opt_angles is None:
                initial_angles = np.random.uniform(0, np.pi, 2*self.p)
            else: 
                initial_angles = self.opt_angles
        else: 
            initial_angles = initial_params

        bds = [(0.0, 2 * np.pi)] * self.p + [(0.0, 2 * np.pi)] * self.p

        if track_energy:
            self.track_cost = True

        t_start = time.time()
        res = minimize(
            self.expectation,
            initial_angles,
            method='L-BFGS-B',
            jac=None,
            bounds=bds,
            options={'maxiter': 10000},
        )
        t_end = time.time()

        self.opt_angles = res.x
        self.exe_time = float(t_end - t_start)
        self.opt_iter = res.nfev
        self.q_energy = res.fun
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(res.x)
        self.olap = self.overlap(self.f_state)

        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')
        if track_energy:
            self.track_cost = False


    def run_heuristic_LW(self, track_energy=False, heruistic_LW_seed1=20,
                         heruistic_LW_seed2=20, stop_on_min=False, track_min=True,
                         bds=None, *, seed=None):
        """Select optimized restart results, then grow the circuit layer by layer.

        Return the selected OptimizeResult with total energy-evaluation accounting.
        A first-hit depth is a numerical upper bound, not a minimum-depth proof.
        """
        from scipy.optimize import OptimizeResult
        if (self.p < 1 or any(not isinstance(v, (int, np.integer)) or v < 1
                             for v in (heruistic_LW_seed1, heruistic_LW_seed2))):
            raise ValueError("Positive depth and restart counts are required.")
        if bds is None or len(bds) == 0:
            bds = [(0., 2*np.pi)]*(2*self.p)
        bounds = np.asarray(bds, dtype=float)
        if (bounds.shape != (2*self.p, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] > bounds[:, 1])):
            raise ValueError("Expected finite ordered bounds for all 2*p parameters.")
        rng = np.random if seed is None else np.random.default_rng(seed)
        def layer_bounds(depth):
            return np.concatenate((bounds[:depth], bounds[self.p:self.p+depth]))
        def guess(depth):
            b = layer_bounds(depth)
            return rng.uniform(b[:, 0], b[:, 1])
        total_nfev, total_nit = 0, 0
        def optimize(fun, x, depth):
            nonlocal total_nfev, total_nit
            result = minimize(fun, x, method='L-BFGS-B', bounds=layer_bounds(depth),
                              options={'maxfun': 10000})
            total_nfev += result.nfev
            total_nit += result.get('nit', 0)
            return result
        previous_tracking = self.track_cost
        self.track_cost = previous_tracking or track_energy
        start = time.time()
        self.lw_log = None
        try:
            selected = min((optimize(self.expectation, guess(1), 1)
                            for _ in range(heruistic_LW_seed1)), key=lambda r: r.fun)
            angles = selected.x.copy()
            depth = 1
            while True:
                at_minimum = np.isclose(selected.fun, self.min, atol=.001, rtol=0)
                if track_min and self.lw_log is None and at_minimum:
                    self.lw_log = (time.time()-start, depth)
                if depth == self.p or (stop_on_min and at_minimum):
                    break
                state = self.qaoa_ansatz(angles)
                def partial_energy(x):
                    evolved = self.apply_ansatz(x, state)
                    value = float(np.vdot(evolved, self.H*evolved).real)
                    if self.track_eval:
                        self.eval_num += 1
                    if self.track_cost:
                        self.tracked_cost.append(value)
                    return value
                # Appended angles obey the bounds for the new chronological layer.
                new_bounds = bounds[[depth, self.p+depth]]
                candidates = []
                for _ in range(heruistic_LW_seed2):
                    x = rng.uniform(new_bounds[:, 0], new_bounds[:, 1])
                    result = minimize(partial_energy, x, method='L-BFGS-B',
                                      bounds=new_bounds, options={'maxfun': 10000})
                    total_nfev += result.nfev
                    total_nit += result.get('nit', 0)
                    candidates.append(result)
                appended = min(candidates, key=lambda r: r.fun).x
                angles = np.concatenate((angles[:depth], appended[:1],
                                         angles[depth:], appended[1:]))
                depth += 1
                selected = optimize(self.expectation, angles, depth)
                angles = selected.x.copy()
            # Tie every reported field to the same selected circuit, including p=1.
            energy = self.expectation(angles)
            total_nfev += 1
        finally:
            self.track_cost = previous_tracking
        result = OptimizeResult(selected)
        result.x, result.fun = angles, energy
        result.nfev, result.nit = total_nfev, total_nit
        result.depth = depth
        self.optimization_result = result
        # Keep the requested-depth warm start valid after early stopping.
        self.optimized_depth = depth
        padding = np.zeros(self.p-depth)
        self.opt_angles = np.concatenate((angles[:depth], padding, angles[depth:], padding))
        self.q_energy = energy
        self.exe_time, self.opt_iter = time.time()-start, total_nfev
        self.q_error = energy-self.min
        self.f_state = self.qaoa_ansatz(angles)
        self.olap = self.overlap(self.f_state)
        self.log = (f'Depth: {depth}\nError: {self.q_error}\nQAOA_Eg: {energy}\n'
                    f'Exact_Eg: {self.min}\nOverlap: {self.olap}\n'
                    f'Energy evaluations: {total_nfev}')
        return result
        


        
    def qaoa_qfi_matrix(self, params, state_ini=None, return_grad=False):
        """Computes the Quantum Fisher Information (QFI) matrix for parameter sensitivity analysis.
        on state ini will be applied .ravel()
        The order of the params is [gamma_1,...,gamma_p,beta_1,...beta_p]
        # NOTE keep track of cost or number of evals doesn't work
        Args:
            params (_type_): _description_
            state_ini (_type_): _description_
            return_grad (bool): If True, returns the gradient of the circuit at the point (params)
            return_cost (bool): If True, returns the cost

        Returns:
            _type_: Quantum Fisher Information matrix and a gradient (if return_grad=True), 
            or just the Quantum Fisher Information matrix (if both return_grad and return_cost are False)
        """ 
        if  state_ini is None:
            state_ini = plus_state(self.n_qubits)       
        else:
            state_ini = state_ini.ravel()
        n_params = len(params)
        p = self.p
        QFI_matrix = np.zeros((n_params, n_params), dtype=float)

        statevector = np.copy(state_ini)
        statevectors_der = np.zeros((n_params,len(state_ini)),dtype=complex)

        for i in range(p):

            statevector_der_gamma = -1j *( self.H * statevector )
            statevector_der_gamma = self.apply_gamma( params[i], statevector_der_gamma)
            statevector_der_gamma = self.apply_beta(params[i + p], statevector_der_gamma)

            statevector = self.apply_gamma(params[i], statevector)
            statevector = self.apply_beta(params[i + p], statevector)

            statevector_der_beta = -1j * self.apply_Hx(statevector)

            statevectors_der[i] = statevector_der_gamma
            statevectors_der[i+p] = statevector_der_beta

            for j in range(i):
                statevectors_der[j] = self.apply_gamma( params[i], statevectors_der[j])
                statevectors_der[j] = self.apply_beta(params[i + p], statevectors_der[j])
                statevectors_der[j+p] = self.apply_gamma( params[i], statevectors_der[j+p])
                statevectors_der[j+p] = self.apply_beta(params[i + p], statevectors_der[j+p])

            for a in range(i+1):
                for b in [i-1,i,i-1+p,i+p]:
                    term_1 = np.vdot(statevectors_der[a], statevectors_der[b])
                    term_2 = np.vdot(statevectors_der[a], statevector) * np.vdot(statevector, statevectors_der[b])
                    QFI_ab = 4 * (term_1 - term_2).real
                    QFI_matrix[a][b] = QFI_matrix[b][a] = QFI_ab

                    term_1 = np.vdot(statevectors_der[a+p], statevectors_der[b])
                    term_2 = np.vdot(statevectors_der[a+p], statevector) * np.vdot(statevector, statevectors_der[b])
                    QFI_ab = 4 * (term_1 - term_2).real
                    QFI_matrix[a+p][b] = QFI_matrix[b][a+p] = QFI_ab
        
        if return_grad:
            # Compute the gradient of the circuit if return_grad=True
            gradient_list = np.zeros(n_params,float)
            for i in range(len(statevectors_der)):
                gradient_list[i] = 2 * np.real(np.vdot(
                    statevectors_der[i],
                    self.H * statevector
                ))
            # If  return_grad is True, return  the gradient
            return QFI_matrix, gradient_list
        
        # If return_grad  is not True, return just the QFI_matrix
        return QFI_matrix

    def run_QFI(self, track_energy=False, estimate_QFI_iter=False, initial_params=None,
                bds=None, *, learning_rate=1.0, regularization=1e-6,
                maxiter=1000, gtol=1e-7, maxls=30):
        """Damped natural-gradient descent with a feasible Armijo line search.

        F^{-1} grad(E) is a search direction, not a Euclidean derivative for
        L-BFGS-B. QFI evaluations are recorded separately; estimate_QFI_iter
        is retained for call compatibility and no longer fabricates cost calls.
        """
        from scipy.optimize import OptimizeResult
        if (not np.isfinite([learning_rate, regularization, gtol]).all()
                or learning_rate <= 0 or regularization <= 0 or gtol <= 0
                or not isinstance(maxiter, (int, np.integer)) or maxiter < 0
                or not isinstance(maxls, (int, np.integer)) or maxls < 1):
            raise ValueError("Expected positive step/damping/tolerance and valid iteration limits.")
        if initial_params is None:
            initial_params = self.opt_angles
        if initial_params is None:
            initial_params = np.random.uniform(0, np.pi, 2*self.p)
        x = np.asarray(initial_params, dtype=float).copy()
        if x.shape != (2*self.p,) or not np.isfinite(x).all():
            raise ValueError("Expected exactly 2*p finite initial parameters.")
        if bds is None or len(bds) == 0:
            bds = [(0.0, 2*np.pi)] * (2*self.p)
        bounds = np.asarray(bds, dtype=float)
        if (bounds.shape != (2*self.p, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] > bounds[:, 1])):
            raise ValueError("Expected one finite ordered bounds pair per parameter.")
        x = np.clip(x, bounds[:, 0], bounds[:, 1])
        previous_tracking = self.track_cost
        self.track_cost = track_energy or previous_tracking
        start = time.time()
        nfev, njev, nit = 0, 0, 0
        success, message = False, "Maximum iterations reached."
        try:
            cost = self.expectation(x)
            nfev += 1
            for _ in range(maxiter):
                fisher, gradient = self.qaoa_qfi_matrix(x, return_grad=True)
                njev += 1
                projected_gradient = x - np.clip(x-gradient, bounds[:, 0], bounds[:, 1])
                if np.linalg.norm(projected_gradient, np.inf) <= gtol:
                    success, message = True, "Projected gradient converged."
                    break
                metric = (fisher + fisher.T) / 2 + regularization*np.eye(len(x))
                direction = -np.linalg.solve(metric, gradient)
                # At active box constraints the full metric step can point out
                # of the domain. Use a projected gradient direction if necessary.
                displacement = np.clip(x+learning_rate*direction, bounds[:, 0], bounds[:, 1])-x
                if gradient @ displacement >= 0:
                    direction = -projected_gradient
                step = learning_rate
                accepted = False
                for _ in range(maxls):
                    candidate = np.clip(x+step*direction, bounds[:, 0], bounds[:, 1])
                    displacement = candidate-x
                    slope = gradient @ displacement
                    if slope < 0:
                        candidate_cost = self.expectation(candidate)
                        nfev += 1
                        if candidate_cost <= cost + 1e-4*slope:
                            x, cost = candidate, candidate_cost
                            accepted = True
                            nit += 1
                            break
                    step *= 0.5
                if not accepted:
                    message = "Line search could not find a decreasing feasible step."
                    break
        finally:
            self.track_cost = previous_tracking
        res = OptimizeResult(x=x, fun=cost, nit=nit, nfev=nfev, njev=njev,
                             success=success, message=message)
        self.optimization_result = res
        self.qfi_evaluations = njev
        self.opt_angles = x
        self.exe_time = time.time()-start
        self.opt_iter = nfev
        self.q_energy = cost
        self.q_error = cost-self.min
        self.f_state = self.qaoa_ansatz(x)
        self.olap = self.overlap(self.f_state)
        self.log = (f'Depth: {self.p}\nError: {self.q_error}\nQAOA_Eg: {cost}\n'
                    f'Exact_Eg: {self.min}\nOverlap: {self.olap}\n'
                    f'Energy evaluations: {nfev}\nQFI evaluations: {njev}\n{message}')
        return res
            


    def run_PROTES(self, size=100, m=int(2.E+3), k=100, k_top=10, track_energy=False):
        """Runs the QAOA using the PROTES optimization method
            
        Returns:
            _type_: _description_
        """           
        from protes import protes_general

        # a = 0        # Grid lower bound
        # b = 2 * np.pi        # Grid upper bound
        bds = [(0.0, 2 * np.pi)] * self.p + [(0.0, 2 * np.pi)] * self.p
        a = np.array([val[0] for val in bds])
        b = np.array([val[1] for val in bds])
        mode_size = [size]*2*self.p # number of points in grid
        # m = int(2.E+3)   # Number of requests to the objective function

        def func(I):
            """Target function: y=f(I); [samples,d] -> [samples]."""
            return  np.array([self.expectation(a + I[i,:]/np.array(mode_size)*(b-a)) for i in range(I.shape[0])])     

        if track_energy:
            self.track_cost = True
        protes_log = {}

        t_start = time.time()
        
        i_opt, y_opt  = protes_general(func, mode_size, m, k, k_top, info=protes_log,  with_info_i_opt_list=True)

        t_end = time.time()

        x = a + i_opt/np.array(mode_size) * (b-a)


        self.protes_log = protes_log
        self.opt_angles = x
        self.exe_time = float(t_end - t_start)
        self.opt_iter = protes_log['m']
        self.q_energy = y_opt
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)

        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')
        if track_energy:
            self.track_cost = False


    def run_cmaes(self, generations=100, track_energy=False, initial_params=None, sigma=1.,n_max_resampling=100, lr_adapt=True, population_size=None, *, seed=None):
        """Runs the QAOA using the cmaes optimization method
            
        Returns:
            _type_: _description_
        """      
        if not isinstance(generations, (int, np.integer)) or generations < 1:
            raise ValueError("generations must be a positive integer.")
        from cmaes import CMA
        # mean: ndarray,
        # sigma: float,
        # bounds: ndarray | None = None,
        # n_max_resampling: int = 100,
        # seed: int | None = None,
        # population_size: int | None = None,
        # cov: ndarray | None = None,
        # lr_adapt: bool = False
        if initial_params is None:
            if self.opt_angles is None:
                initial_angles = np.random.uniform(0, np.pi, 2*self.p)
            else: 
                initial_angles = self.opt_angles
        else: 
            initial_angles = initial_params
        bds = [(0.0, 2 * np.pi)] * self.p + [(0.0, 2 * np.pi)] * self.p
        bds_f = lambda x: np.array([[bds[i][0],bds[i][1]] for i in range(x)] + [[bds[i+x][0],bds[i+x][1]] for i in range(x)])

        initial_angles = np.asarray(initial_angles, dtype=float).copy()
        if initial_angles.shape != (2*self.p,) or not np.isfinite(initial_angles).all():
            raise ValueError("Expected exactly 2*p finite initial parameters.")
        optimizer = CMA(mean=initial_angles, seed=seed, sigma=sigma, n_max_resampling=n_max_resampling, lr_adapt=lr_adapt, population_size=population_size, bounds=bds_f(self.p))


        previous_tracking = self.track_cost
        self.track_cost = previous_tracking or track_energy

        t_start = time.time()
        
        opt_iter = 0
        best_value, best_angles = np.inf, None

        try:
            for _ in range(generations):
                solutions = []
                for _ in range(optimizer.population_size):
                    x = optimizer.ask()
                    value = self.expectation(x)
                    solutions.append((x, value))
                    if value < best_value:
                        best_value, best_angles = value, np.asarray(x).copy()
                    opt_iter+=1
                optimizer.tell(solutions)

                if optimizer.should_stop():
                    break


        finally:
            self.track_cost = previous_tracking

        t_end = time.time()



        self.opt_angles = best_angles
        self.exe_time = float(t_end - t_start)
        self.opt_iter = opt_iter
        self.q_energy = best_value
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)

        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')
        self.track_cost = previous_tracking

    def run_de(self, track_energy=False):
        """Runs the QAOA using the differential evolution optimization method
            The maximum number of function evaluations (with no polishing) is: (maxiter + 1) * popsize * (N - N_equal)
        Returns:
            None
        """      
        if track_energy:
            self.track_cost = True

        # Define the objective function for optimization
        def objective_function(x):
            return self.expectation(x)

        # Define bounds for the variables
        bds = [(0.0, 2 * np.pi)] * self.p + [(0.0, 2 * np.pi)] * self.p

        t_start = time.time()

        # Run differential evolution optimization
        result = differential_evolution(objective_function,bounds=bds, updating='immediate', polish=False, atol=1e-10)

        t_end = time.time()

        # Extract optimization results
        self.opt_angles = result.x
        self.exe_time = float(t_end - t_start)
        self.opt_iter = result.nfev  # Number of function evaluations
        self.q_energy = result.fun
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)

        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iterations: {self.opt_iter}')
        
        if track_energy:
            self.track_cost = False



    def run_mcts(self, track_energy=False, b=5, simulations=1000, exploration_weight=1.0):
        """Runs the QAOA using Monte Carlo Tree Search (MCTS) for parameter optimization."""
        t_start = time.time()

        if (not isinstance(simulations, (int, np.integer)) or simulations < 1
                or not isinstance(b, (int, np.integer)) or b < 2):
            raise ValueError("Expected positive simulations and at least two grid values.")
        previous_tracking = self.track_cost
        self.track_cost = previous_tracking or track_energy
        best_cost, best_angles = np.inf, None
        # Initialize root node and discretized parameter space
        root = MCTSNode()
        parameter_space = np.linspace(0, np.pi, b)  # Discretized parameter values

        for _ in range(simulations):
            # Selection: Traverse the tree to a leaf node using UCB
            node = root
            while node.is_fully_expanded(len(parameter_space)) and len(node.parameters) < 2 * self.p:
                node = node.best_child(exploration_weight)

            # Expansion: Add a new child if not fully expanded
            if len(node.parameters) < 2 * self.p:
                next_param = parameter_space[len(node.children)]  # Select the next discrete value
                node = node.expand(next_param)

            # Simulation: Complete the parameters randomly and evaluate the cost function
            remaining_params = 2 * self.p - len(node.parameters)
            complete_params = node.parameters + list(np.random.choice(parameter_space, remaining_params))
            cost = self.expectation(complete_params)
            if cost < best_cost:
                best_cost, best_angles = cost, np.array(complete_params).copy()

            # Backpropagation: Update the tree with the simulation result
            while node:
                node.backpropagate(-cost)  # Negate cost for minimization
                node = node.parent

        t_end = time.time()

        # Save results
        self.opt_angles = best_angles
        self.exe_time = float(t_end - t_start)
        self.q_energy = best_cost
        self.opt_iter = simulations
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(best_angles)
        self.olap = self.overlap(self.f_state)

        self.log = (f'Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f'Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n')

        self.track_cost = previous_tracking




class MCTSNode:
    def __init__(self, parameters=None, parent=None):
        self.parameters = parameters or []  # Current parameters at this node
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.visits = 0  # Number of visits to this node
        self.value = 0  # Cumulative value (cost) of this node

    def is_fully_expanded(self, branching_factor):
        return len(self.children) >= branching_factor

    def best_child(self, exploration_weight=1.0):
        """Select child node based on UCB1."""
        return max(
            self.children,
            key=lambda child: child.value / child.visits + exploration_weight * np.sqrt(np.log(self.visits) / (child.visits + 1))
        )

    def expand(self, next_param):
        """Add a new child node with the next parameter."""
        new_node = MCTSNode(self.parameters + [next_param], parent=self)
        self.children.append(new_node)
        return new_node

    def backpropagate(self, result):
        """Update node statistics during backpropagation."""
        self.visits += 1
        self.value += result