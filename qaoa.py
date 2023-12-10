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
from scipy.optimize import minimize
import random
import time
from functools import reduce
from typing import List
from numpy import ndarray



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


def create_depolarization_kraus(p_depolarization):
    """Creates Kraus operators and probabilities for the depolarization channel.

    Args:
        p_depolarization (float): Probability of depolarization.

    Returns:
        List[np.ndarray], List[float]: List of Kraus operators and their probabilities.
    """
    kraus_ops = [
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]])
    ]

    probabilities = [np.sqrt(1 - 3*p_depolarization/4), np.sqrt(p_depolarization) / 2,
                     np.sqrt(p_depolarization) / 2, np.sqrt(p_depolarization) / 2]

    return kraus_ops, probabilities

def create_amplitude_damping_kraus(p_amplitude_damping):
    """Create Kraus operators and probabilities for the amplitude damping channel.
    
    Args:
        p_amplitude_damping (float): Probability of amplitude damping.

    Returns:
        List[np.ndarray], List[float]: List of Kraus operators and their probabilities.
    """
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
    kraus_ops = [
        np.eye(2),
        np.array([[1, 0], [0, -1]])
    ]

    probabilities = [np.sqrt(1 - p_phase_flip), np.sqrt(p_phase_flip)]

    return kraus_ops, probabilities


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
        # TODO make a function to set these params 
        self.heruistic_LW_seed1 = 50
        self.heruistic_LW_seed2 = 20
        self.opt_angles = None
        self.exe_time = None
        self.opt_iter = None
        self.q_energy = None
        self.q_error = None
        self.f_state = None
        self.olap = None
        self.log = None
    
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

        Args:
            angles (List[float]): _description_

        Returns:
            ndarray: _description_
        """        
        state = plus_state(self.n_qubits)
        p = self.p
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
        p = self.p
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
        ex = np.vdot(state, state * self.H)
        return np.real(ex)
    

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
            for q in range(self.n_qubits):
                noise_ind = np.random.choice(len(noise_prob),size=1,p=noise_prob)[0]
                # Apply noise by randomly selecting a Kraus operator
                kraus_op = kraus_ops[noise_ind]
                
                # Generate Kraus operators for each qubit
                kraus_operator_q = reduce(np.kron, [I]*q + [kraus_op] + [I]*(self.n_qubits-q-1))
                state = np.dot(kraus_operator_q, state)

        state = state/(np.linalg.norm(state))
        return state

    def expectation_noise(self, angles: List[float], noise_prob: List[float], kraus_ops: List[np.ndarray], num_samples: int) -> float:
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
        total_expectation = 0.0
        for _ in range(num_samples):
            noisy_state = self.apply_ansatz_noise(angles, noise_prob, kraus_ops)
            ex = np.vdot(noisy_state, noisy_state * self.H)
            total_expectation += np.real(ex)
        return total_expectation / num_samples
    


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
            state = self.qaoa_ansatz(angles)
            f1 = np.vdot(state, state * self.H)

            angles[i] -= 2*delta
            state = self.qaoa_ansatz(angles)
            f2 = np.vdot(state, state * self.H)

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
        

    def run_heuristic_LW(self):
        """Runs the QAOA using the heuristic L-BFGS-B optimization method with layer-wise learning approach.
            
        Returns:
            _type_: _description_
        """        
        initial_guess = lambda x: (
            [random.uniform(0, 2 * np.pi) for _ in range(x)] + [random.uniform(0, np.pi) for _ in range(x)]
        )
        bds = lambda x: [(0.1, 2 * np.pi)] * x + [(0.1, 1 * np.pi)] * x

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

        temp = []
        t_start = time.time()

        # find a good starting point for layer one
        for _ in range(self.heruistic_LW_seed1):
            initial_guess_p1 = initial_guess(1)
            res = minimize(
                self.expectation,
                initial_guess_p1,
                method='L-BFGS-B',
                jac=None,
                bounds=bds(1),
                options={'maxfun': 150000}
            )

            temp.append([self.expectation(res.x), initial_guess_p1])

        temp = np.asarray(temp, dtype=object)
        idx = np.argmin(temp[:, 0])
        opt_angles = temp[idx][1]

        t_state = np.ones((2 ** self.n_qubits, 1), dtype='complex128') * (1 / np.sqrt(2 ** self.n_qubits)) #|+>

        # optimize untill we find all params
        while len(opt_angles) < 2 * self.p: 
            print('LW point now:', len(opt_angles) / 2)

            t_state = self.qaoa_ansatz(opt_angles)

            # function to find optimum with respect to the fixed found angles
            ex = lambda x: np.real(np.vdot(
                self.apply_ansatz(x, t_state),
                self.apply_ansatz(x, t_state) * self.H
            ))
            temp = []

            for _ in range(self.heruistic_LW_seed2):
                res = minimize(
                    ex,
                    initial_guess(1),
                    method='L-BFGS-B',
                    jac=None,
                    bounds=bds(1),
                    options={'maxfun': 150000}
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
                bounds=bds(int(len(opt_angles) / 2)),
                options={'maxfun': 150000}
            )
            opt_angles = res.x

        self.opt_angles = opt_angles

        t_end = time.time()
        self.exe_time = float(t_end - t_start)
        self.opt_iter = float(res.nfev)
        self.q_energy = self.expectation(self.opt_angles)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)
        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')
        
    def run(self):
        """Runs the QAOA using the heuristic L-BFGS-B optimization method
            
        Returns:
            _type_: _description_
        """           

        initial_angles = np.random.uniform(0, np.pi, 2*self.p)
        bds = [(0.0, 2 * np.pi)] * self.p + [(0.0, 2 * np.pi)] * self.p

        t_start = time.time()
        res = minimize(
            self.expectation,
            initial_angles,
            method='L-BFGS-B',
            jac=None,
            bounds=bds,
            options={'maxiter': 15000}
        )
        t_end = time.time()

        self.opt_angles = res.x
        self.exe_time = float(t_end - t_start)
        self.opt_iter = float(res.nfev)
        self.q_energy = self.expectation(res.x)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(res.x)
        self.olap = self.overlap(self.f_state)

        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')

        
    def qaoa_qfi_matrix(self, params, state_ini, return_grad=False):
        """Computes the Quantum Fisher Information (QFI) matrix for parameter sensitivity analysis.
            on state ini will be applied .ravel()
            The order of the params is [gamma_1,...,gamma_p,beta_1,...beta_p]
        Args:
            params (_type_): _description_
            state_ini (_type_): _description_
            return_grad (Bool): If true returns the gradient of circuit at the point (params)
        Returns:
            _type_: Quantum Fisher Information matrix and a gradient (if return_grad=True)
        """        
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
            return QFI_matrix, gradient_list

        return QFI_matrix

