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
from numpy import array, asarray, arange, copy, vdot, kron, eye, zeros, diag, sqrt, cos, sin, exp, pi


def Graph_to_Hamiltonian(G, n):
    def tensor(k):
        t = k[0]
        i = 1

        while i < len(k):
            t = np.kron(t, k[i])
            i += 1
        return t

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
    """Constructs the Ising Hamiltonian.

    Args:
        n_qubits (_type_): _description_
        bc (str, optional): _description_. Defaults to "closed".

    Returns:
        _type_: _description_
    """    
    # Function that constructs the Ising Hamiltonian
    Z = np.array([1., -1.],)
    I = np.array([1.,1.])
    
    Hzz = np.zeros(2**n_qubits)
    for q in range(n_qubits - 1):
        Hzz = Hzz + reduce(kron, [I]*q + [Z, Z] + [I]*(n_qubits-q-2))
    if bc == "closed":
        Hzz = Hzz + reduce(kron, [Z] + [I]*(n_qubits-2) + [Z])

    return Hzz

def H_sat(n, k, alpha):
    """Constructs the k-SAT Hamiltonian from random clauses.
            A slightly naive approach without checking that all terms\clauses persist 
    Args:
        n (_type_): _description_
        k (_type_): _description_
        alpha (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Function that constructs the k-SAT Hamiltonian from a list of random clauses

    I_prime = np.array([1, 1])
    rho_0_prime = np.array([1, 0])
    rho_1_prime = np.array([0, 1])

    h = np.zeros(2**n)

    for i in range(int(alpha) * n):
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
        return statevector * np.exp(-1j * gamma * self.H.reshape(2**self.n_qubits,1))

    def apply_Hx(self, statevector: ndarray) -> ndarray:
        """Applies the Hx operator to the state vector.

        Args:
            statevector (ndarray): _description_

        Returns:
            ndarray: _description_
        """        
        statevector_new = np.zeros(len(statevector), dtype=complex)
        for i in range(self.n_qubits):
            statevector_swap = statevector[self.x_list[i]]
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
        n_qubits = self.n_qubits
        c = np.cos(beta)
        s = np.sin(beta)
        statevector_new = statevector.copy()
        for i in range(n_qubits):
            statevector_swap = statevector_new[self.x_list[i]]
            statevector_new = -1j * s * statevector_swap + c * statevector_new
        return statevector_new
    
    def qaoa_ansatz(self, angles: List[float]) -> ndarray:
        """ Generates the QAOA ansatz state for a given set of angles.

        Args:
            angles (List[float]): _description_

        Returns:
            ndarray: _description_
        """        
        state = np.ones((2**self.n_qubits, 1), dtype='complex128') * (1 / np.sqrt(2**self.n_qubits))
        p = int(len(angles) / 2)
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
        p = int(len(angles) / 2)
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
        ex = np.vdot(state, state * (self.H).reshape((2**self.n_qubits, 1)))
        return np.real(ex)

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
        
    def run_RI(self):
        """Runs the QAOA using the Richardson Iteration (RI) optimization method.
        """        
        t_start = time.time()
        initial_angles = []
        bds = [(0, 2 * np.pi)] * self.p + [(0, 1 * np.pi)] * self.p
        for i in range(2 * self.p):
            if i < self.p:
                initial_angles.append(random.uniform(0, 2 * np.pi))
            else:
                initial_angles.append(random.uniform(0, np.pi))

        res = minimize(
            self.expectation,
            initial_angles,
            method='L-BFGS-B',
            jac=None,
            bounds=bds,
            options={'maxfun': 150000}
        )

        t_end = time.time()
        self.opt_angles = res.x
        self.exe_time = float(t_end - t_start)
        self.opt_iter = float(res.nfev)
        self.q_energy = self.expectation(res.x)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(res.x)
        self.olap = self.overlap(self.f_state)[0]
        self.log = (
            f'Depth: {self.p} \n'
            f'Error: {self.q_error} \n'
            f'QAOA_Eg: {self.q_energy} \n'
            f'Exact_Eg: {self.min} \n'
            f'Overlap: {self.olap} \n'
            f'Exe_time: {self.exe_time} \n'
            f'Iterations: {self.opt_iter}'
        )

    def run_heuristic_LW(self):
        """Runs the QAOA using the heuristic L-BFGS-B optimization method.

        Returns:
            _type_: _description_
        """        
        initial_guess = lambda x: (
            [random.uniform(0, 2 * np.pi) for _ in range(x)] + [random.uniform(0, np.pi) for _ in range(x)]
        )
        bds = lambda x: [(0.1, 2 * np.pi)] * x + [(0.1, 1 * np.pi)] * x

        def combine(a, b):
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

        t_state = np.ones((2 ** self.n_qubits, 1), dtype='complex128') * (1 / np.sqrt(2 ** self.n_qubits))

        while len(opt_angles) < 2 * self.p:
            print('LW point now:', len(opt_angles) / 2)
            ts1 = time.time()
            t_state = self.qaoa_ansatz(opt_angles)

            ex = lambda x: np.real(np.vdot(
                self.apply_ansatz(x, t_state),
                self.apply_ansatz(x, t_state) * (self.H).reshape((2 ** self.n_qubits, 1))
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
        self.olap = self.overlap(self.f_state)[0]
        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')
        
    def run(self):
        initial_angles = []
        #TODO the problem that I found while was implementing this one is that Akshay and Ernesto have different angles order
        bds = [(0.1, 2 * np.pi)] * self.p + [(0.1, 1 * np.pi)] * self.p
        for i in range(2 * self.p):
            if i < self.p:
                initial_angles.append(random.uniform(0, 2 * np.pi))
            else:
                initial_angles.append(random.uniform(0, np.pi))

        t_start = time.time()
        res = minimize(
            self.expectation,
            initial_angles,
            method='L-BFGS-B',
            jac=None,
            bounds=bds,
            options={'maxiter': 5000}
        )
        t_end = time.time()

        self.opt_angles = res.x
        self.exe_time = float(t_end - t_start)
        self.opt_iter = float(res.nfev)
        self.q_energy = self.expectation(res.x)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(res.x)
        self.olap = self.overlap(self.f_state)[0]

        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')

        
    def qaoa_qfi_matrix(self, pars, state_ini):
        """Computes the Quantum Fisher Information (QFI) matrix for parameter sensitivity analysis.
            on state ini will be applied .ravel()
        Args:
            pars (_type_): _description_
            state_ini (_type_): _description_

        Returns:
            _type_: _description_
        """        
        state_ini = state_ini.ravel()
        n_pars = len(pars)
        p = self.p
        QFI_matrix = np.zeros((n_pars, n_pars), dtype=complex)

        statevector = np.copy(state_ini)
        statevectors_der = []

        for i in range(p):
            n_pars_i = 2 * (i + 1)

            statevector_der_gamma = -1j *( self.H * statevector )
            statevector_der_gamma = self.apply_gamma( pars[2 * i], statevector_der_gamma)
            statevector_der_gamma = self.apply_beta(pars[2 * i + 1], statevector_der_gamma)

            statevector = self.apply_gamma(pars[2 * i], statevector)
            statevector = self.apply_beta(pars[2 * i + 1], statevector)

            statevector_der_beta = -1j * self.apply_Hx(statevector)

            statevectors_der.append(statevector_der_gamma)
            statevectors_der.append(statevector_der_beta)

            for j in range(n_pars_i - 2):
                statevectors_der[j] = self.apply_gamma( pars[2 * i], statevectors_der[j])
                statevectors_der[j] = self.apply_beta(pars[2 * i + 1], statevectors_der[j])

            for a in range(n_pars_i):
                for b in range(n_pars_i - 2, n_pars_i):
                    term_1 = np.vdot(statevectors_der[a], statevectors_der[b])
                    term_2 = np.vdot(statevectors_der[a], statevector) * np.vdot(statevector, statevectors_der[b])
                    QFI_ab = 4 * (term_1 - term_2).real
                    QFI_matrix[a][b] = QFI_matrix[b][a] = QFI_ab

        return QFI_matrix
