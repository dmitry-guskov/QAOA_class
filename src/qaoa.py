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

from protes import protes_general
from cmaes import CMA
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
    """Creates Kraus operators and probabilities for the depolarization channel.

    Args:
        p_depolarization (float): Probability of depolarization.

    Returns:
        List[np.ndarray], List[float]: List of Kraus operators and their probabilities.
    """
    if not 0 <= p_depolarization <= 1:
        raise ValueError("Depolarization probability must be between 0 and 1.")
    kraus_ops = [
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]])
    ]

    probabilities = [np.sqrt(1 - 3*p_depolarization/4)] + [np.sqrt(p_depolarization) / 2] * 3

    return kraus_ops, probabilities

def create_amplitude_damping_kraus(p_amplitude_damping):
    """Create Kraus operators and probabilities for the amplitude damping channel.
    
    Args:
        p_amplitude_damping (float): Probability of amplitude damping.

    Returns:
        List[np.ndarray], List[float]: List of Kraus operators and their probabilities.
    """
    if not 0 <= p_amplitude_damping <= 1:
        raise ValueError("Depolarization probability must be between 0 and 1.")
    kraus_ops = [
         np.array([[1/np.sqrt(1 - p_amplitude_damping),0],[0,1]]),
         np.array([[0,1],[0,0]])
    ]

    probabilities = [np.sqrt(1 - p_amplitude_damping), np.sqrt(p_amplitude_damping)]

    return kraus_ops, probabilities

def create_phase_flip_kraus(p_phase_flip):
    """Create Kraus operators and probabilities for the phase flip channel.
    
    Args:
        p_phase_flip (float): Probability of phase flip.

    Returns:
        List[np.ndarray], List[float]: List of Kraus operators and their probabilities.
    """
    if not 0 <= p_phase_flip <= 1:
        raise ValueError("Depolarization probability must be between 0 and 1.")
    kraus_ops = [
        np.eye(2),
        np.array([[1, 0], [0, -1]])
    ]

    probabilities = [np.sqrt(1 - p_phase_flip), np.sqrt(p_phase_flip)]

    return kraus_ops, probabilities



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
        self.H = H
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
        """Generates a list of indices for state vector swapping in the mixer.

        Returns:
           List: x_list  - List of lists of indices.
        """             
        def split(x, k):
            return x.reshape((2**k, -1))
        
        def sym_swap(x):
            return np.asarray([x[-1], x[-2], x[1], x[0]])
        
        n_qubits = self.n_qubits
        x_list = []
        t1 = np.asarray([np.arange(2**(n_qubits-1), 2**n_qubits), np.arange(0, 2**(n_qubits-1))])
        t1 = t1.flatten()
        x_list.append(t1.flatten())
        t2 = t1.reshape(4, -1)
        t3 = sym_swap(t2)
        t1 = t3.flatten()
        x_list.append(t1)
        k = 1
        
        while k < (n_qubits - 1):
            t2 = split(t1, k)
            t2 = np.asarray(t2)
            t1 = []
            for y in t2:
                t3 = y.reshape((4, -1))
                t4 = sym_swap(t3)
                t1.append(t4.flatten())
            t1 = np.asarray(t1)
            t1 = t1.flatten()
            x_list.append(t1)
            k += 1
        
        return x_list

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
    
    def construct_QAOA_operator_term(self,  angles: List[float], part_hamiltonian=None):
        """
        doesn't work yet! 
        Construct the QAOA operator term, i.e. takes part_hamiltonian as a term of the QAOA Hamiltonian and sandwiches it with mixers and exponents of QAOA Hamiltonian. 
        E.g. qaoao H = Z1Z2 + Z2Z3 + Z1Z3 - triangle , part_hamiltonian = Z1Z2 - term, QAOA = exp_H @ exp_X @ part_hamiltonian @ exp_X.T.conjugate() @ exp_H.T.conjugate()

        Args:
            angles (List[float]): _description_
            part_hamiltonian: part of QAOA Hamiltonian or if the QAOA Hailtonian itself by default.
        Returns:
            np.ndarray: QAOA operator term.
        """
        # Define angles for mixers
        if len(angles) % 2 != 0:
            raise ValueError("Number of angles must be even.")

        if part_hamiltonian is None or not part_hamiltonian.any():
            part_hamiltonian = self.H
        qaoa_operator = self.qaoa_operator(angles)

        ans = qaoa_operator @ np.diag(part_hamiltonian) @ qaoa_operator.T.conj()

        return ans 

    def apply_ansatz_noise(self, angles: List[float], noise_prob: List[float], kraus_ops: List[np.ndarray]) -> np.ndarray:
        """Applies the QAOA ansatz to the state vector with noise.

        Args:
            angles (List[float]): The list of QAOA angles.
            noise_prob (List[float]): Probability of applying noise after each layer.
            kraus_ops (List[np.ndarray]): List of Kraus operators representing different noise channels.

        Returns:
            np.ndarray: The state vector with noise applied.
        """
        state = plus_state(self.n_qubits)
        p = self.p
        noise_prob /= np.array(noise_prob).sum()

        I = np.eye(2)
        for i in range(p):
            state = self.apply_gamma(angles[i], state)
            state = self.apply_beta(angles[p + i], state)
            #TODO this step can be optimized by makeing the whole noise operator in place insted of applying 1-local operators
            noise_inds = []
            for q in range(self.n_qubits):
                noise_ind = np.random.choice(len(noise_prob),size=1,p=noise_prob)[0]
                noise_inds.append(noise_ind)                
                # Apply noise by randomly selecting a Kraus operator
                kraus_op = kraus_ops[noise_ind]
                
                # Generate Kraus operators for each qubit
                kraus_operator_q = reduce(np.kron, [I]*q + [kraus_op] + [I]*(self.n_qubits-q-1))
                state = np.dot(kraus_operator_q, state)

        return state


    def expectation_noise(self, angles: List[float], noise_prob: List[float], kraus_ops: List[np.ndarray], num_samples: int, return_state=False) -> float:
        """Calculates the average expectation value with noise.

        Args:
            angles (List[float]): The list of QAOA angles.
            noise_prob (List[float]): Probability of applying noise after each layer.
            kraus_ops (List[np.ndarray]): List of Kraus operators representing different noise channels.
            num_samples (int): Number of samples to average over.

        Returns:
            float: The average expectation value with noise.
        """
        noise_prob /= np.array(noise_prob).sum()
        noisy_state = self.apply_ansatz_noise(angles, noise_prob, kraus_ops)
        for _ in range(num_samples-1):
            noisy_state += self.apply_ansatz_noise(angles, noise_prob, kraus_ops)
        total_state = noisy_state/np.linalg.norm(noisy_state)
        if return_state:
            return np.real(np.vdot(total_state, total_state * self.H)), total_state
        # NOTE doesn't implement racking of evals 
        return np.real(np.vdot(total_state, total_state * self.H))

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
        if initial_params == None:
            if self.opt_angles is None or not self.opt_angles.any(): 
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


    def run_heuristic_LW(self, track_energy=False, heruistic_LW_seed1=20, heruistic_LW_seed2=20, stop_on_min=False, track_min=True ):
        """Runs the QAOA using the heuristic L-BFGS-B optimization method with layer-wise learning approach.
            
        Returns:
            _type_: _description_
        """        

        initial_guess = lambda x: (
            [random.uniform(0, 2 * np.pi) for _ in range(x)] + [random.uniform(0, np.pi) for _ in range(x)]
        )

        bds = [(0.0, 2 * np.pi)] * self.p + [(0.0, 2 * np.pi)] * self.p
        bds_f = lambda x: [bds[i] for i in range(x)] + [bds[i+self.p] for i in range(x)]
        # [(0., 2 * np.pi)] * x + [(0., 2 * np.pi)] * x

        def combine(a, b): # function to insert angles of new layer to previously found optimum 
            a = list(a)
            b = list(b)
            a1 = a[0:int(len(a) / 2)]
            a2 = a[int(len(a) / 2)::]
            b1 = b[0:int(len(b) / 2)]
            b2 = b[int(len(b) / 2)::]
            a = a1 + b1
            b = a2 + b2
            return a + b
        

        if track_energy:
            self.track_cost = True

        temp = []

        t_start = time.time()


        # find a good starting point for layer one
        for _ in range(heruistic_LW_seed1):
            initial_guess_p1 = initial_guess(1)
            res = minimize(
                lambda x:  self.expectation(x),
                initial_guess_p1,
                method='L-BFGS-B',
                jac=None,
                bounds=bds_f(1),
                options={'maxfun': 10000},
            )

            temp.append([res.fun, initial_guess_p1])

        temp = np.asarray(temp, dtype=object)
        idx = np.argmin(temp[:, 0])
        opt_angles = temp[idx][1]
        p_min = -1
        # optimize untill we find all params
        while len(opt_angles) < 2 * self.p: 
            # print('LW point now:', len(opt_angles) / 2)

            t_state = self.qaoa_ansatz(opt_angles)

            # function to find optimum with respect to the fixed found angles
            def partial_expectation(x):
                if self.track_eval:
                    self.eval_num += 1
                en = np.real(np.vdot(
                self.apply_ansatz(x, t_state),
                self.apply_ansatz(x, t_state) * self.H))
                if self.track_cost:
                    self.tracked_cost.append(en)
                return en
            
            temp = []

            for _ in range(heruistic_LW_seed2):
                res = minimize(
                    partial_expectation,
                    initial_guess(1),
                    method='L-BFGS-B',
                    jac=None,
                    bounds=bds_f(1),
                    options={'maxfun': 10000},
                )

                temp.append([res.fun, res.x])

            temp = np.asarray(temp, dtype=object)
            idx = np.argmin(temp[:, 0])
            lw_angles = temp[idx][1]

            # now combine angles of new added layer and optimize all parameters together
            opt_angles = combine(opt_angles, lw_angles)
            res = minimize(
                self.expectation,
                opt_angles,
                method='L-BFGS-B',
                jac=None,
                bounds=bds_f(int(len(opt_angles) / 2)),
                options={'maxfun': 10000},
            )
            opt_angles = res.x

            if stop_on_min and np.isclose(res.fun, self.min, atol=0.001):
                break
            if track_min and np.isclose(res.fun, self.min, atol=0.001):
                t_min = time.time()
                p_min = len(opt_angles)//2
                track_min = False
        t_end = time.time()


        self.opt_angles = opt_angles
        
        if p_min != -1:
            self.lw_log = (float(t_min - t_start), p_min)

        self.exe_time = float(t_end - t_start)
        self.opt_iter = res.nfev
        self.q_energy = res.fun 
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)
        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')
        if track_energy:
            self.track_cost = False
        


        
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

    def run_QFI(self, track_energy=False, estimate_QFI_iter=False, initial_params=None):
        """Runs the QAOA optimization using the Quantum Fisher Information/natural gradient descent optimization method
            # NOTE keep track of cost or number of evals doesn't work properly due to QFI matrix calculation 
            # TODO estimate how many evals on one call of QFI, it should depend on depth 
        Returns:
            _type_: _description_
        """           
        if initial_params == None:
            if self.opt_angles is None or not self.opt_angles.any(): 
                initial_angles = np.random.uniform(0, np.pi, 2*self.p)
            else: 
                initial_angles = self.opt_angles
        else: 
            initial_angles = initial_params
        bds = [(0.0, 2 * np.pi)] * self.p + [(0.0, 2 * np.pi)] * self.p


        def expectation_and_grad(x):
            qfi_matrix, grad = self.qaoa_qfi_matrix(x, plus_state(self.n_qubits), return_grad=True) 
            cost = self.expectation(x)
            # estimation for QFI numbers of eval:
            if estimate_QFI_iter and self.track_cost: 
                self.tracked_cost += [cost]*self.p*2
            # Compute the natural gradient
            try:
                natural_grad = np.linalg.inv(qfi_matrix) @ grad
            except np.linalg.LinAlgError:
                # If the matrix is not invertible, use the gradient instead
                natural_grad = grad

            return cost, natural_grad
        
        if track_energy:
            self.track_cost = True
        
        t_start = time.time()
        
        res = minimize(
            expectation_and_grad,
            initial_angles,
            method='L-BFGS-B',
            jac=True,  # jac=True indicates that the function returns both the value and the gradient
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
                    f' Iterations: {self.opt_iter}')
        
        if track_energy:
            self.track_cost = False
            


    def run_PROTES(self, size=100, m=int(2.E+3), k=100, k_top=10, track_energy=False):
        """Runs the QAOA using the PROTES optimization method
            
        Returns:
            _type_: _description_
        """           

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


    def run_cmaes(self, generations=100, track_energy=False, initial_params=None, sigma=1.,n_max_resampling=100, lr_adapt=True, population_size=None):
        """Runs the QAOA using the cmaes optimization method
            
        Returns:
            _type_: _description_
        """      
        # mean: ndarray,
        # sigma: float,
        # bounds: ndarray | None = None,
        # n_max_resampling: int = 100,
        # seed: int | None = None,
        # population_size: int | None = None,
        # cov: ndarray | None = None,
        # lr_adapt: bool = False
        if initial_params == None:
            if self.opt_angles is None or not self.opt_angles.any(): 
                initial_angles = np.random.uniform(0, np.pi, 2*self.p)
            else: 
                initial_angles = self.opt_angles
        else: 
            initial_angles = initial_params
        bds = [(0.0, 2 * np.pi)] * self.p + [(0.0, 2 * np.pi)] * self.p
        bds_f = lambda x: np.array([[bds[i][0],bds[i][1]] for i in range(x)] + [[bds[i+x][0],bds[i+x][1]] for i in range(x)])

        optimizer = CMA(mean=initial_angles, sigma=sigma, n_max_resampling=n_max_resampling, lr_adapt=lr_adapt, population_size=population_size, bounds=bds_f(self.p))


        if track_energy:
            self.track_cost = True

        t_start = time.time()
        
        opt_iter = 0

        for _ in range(generations):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = self.expectation(x)
                solutions.append((x, value))
                opt_iter+=1
            optimizer.tell(solutions)

            if optimizer.should_stop():
                break


        t_end = time.time()



        self.opt_angles = x
        self.exe_time = float(t_end - t_start)
        self.opt_iter = opt_iter
        self.q_energy = value
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)

        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')
        if track_energy:
            self.track_cost = False

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

            # Backpropagation: Update the tree with the simulation result
            while node:
                node.backpropagate(-cost)  # Negate cost for minimization
                node = node.parent

        # Select the best parameters from the root's children
        best_node = max(root.children, key=lambda child: child.value / child.visits)
        best_angles = best_node.parameters + list(np.random.choice(parameter_space, 2 * self.p - len(best_node.parameters)))

        t_end = time.time()

        # Save results
        self.opt_angles = best_angles
        self.exe_time = float(t_end - t_start)
        self.q_energy = self.expectation(best_angles)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(best_angles)
        self.olap = self.overlap(self.f_state)

        self.log = (f'Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f'Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n')

        if track_energy:
            self.track_cost = False




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