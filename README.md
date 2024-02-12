# Quantum Approximate Optimization Algorithm (QAOA)

This repository contains a Python implementation of the Quantum Approximate Optimization Algorithm (QAOA) for solving various optimization problems. It includes functions to construct Ising Hamiltonians, k-SAT Hamiltonians, and solve the Max Cut problem. The QAOA logic can be used to find approximate solutions to these optimization problems.
```markdown


## Getting Started

These instructions will help you understand the implementation and run the provided examples.

### Prerequisites

- Python 3
- NumPy
- SciPy

### Installing

You can install the required dependencies using pip:

pip install numpy scipy

### Using
from qaoa import *

```

## Usage
Detailed usage examples can be found in the examples.ipynb file.
### Example 1: Creating an Ising Hamiltonian and Finding Fisher Information Matrix
```python
# Define the number of qubits and connectivity type ("open" or "closed")
n_qubits = 4
connectivity = "closed"

# Create an Ising Hamiltonian
ising_hamiltonian = H_zz_Ising(n_qubits, connectivity)

depth = 1
# Initialize QAOA with a specific depth and the Ising Hamiltonian
qaoa = QAOA(depth=depth, H=ising_hamiltonian)

# Define a set of parameters
angles = [0.5, 0.2]*depth  # Adjust these parameters as needed

# Calculate the Fisher Information Matrix
qfi_matrix = qaoa.qaoa_qfi_matrix(angles, state_ini=plus_state(n_qubits)) #|+>

print("Fisher Information Matrix:")
with np.printoptions(precision=3, suppress=True):
    print(qfi_matrix)
```
### Example 2: Creating a k-SAT Hamiltonian and Optimizing it with QAOA
```python
# Define the number of variables (n), clauses (k), and alpha for k-SAT Hamiltonian
n = 5
k = 3
alpha = 2.0

# Create a k-SAT Hamiltonian
k_sat_hamiltonian = H_sat(n, k, alpha)

depth = 2
# Initialize QAOA with a specific depth and the k-SAT Hamiltonian
qaoa = QAOA(depth=depth, H=k_sat_hamiltonian)


# Run the QAOA optimization using the L-BFGS-B method
qaoa.run()
# Print the optimization results
with np.printoptions(precision=3, suppress=True):
    print("QAOA Parameters:", qaoa.opt_angles)
    print("Optimized QAOA Energy:", qaoa.q_energy)
```
### Example 3: Solving the Max Cut Problem
### Example 4: Solving QAOA in the presence of noise

## Full description
The given code is an implementation of the Quantum Approximate Optimization Algorithm (QAOA) to solve optimization problems. It includes functions to construct maxCUT Hamiltonian, Ising Hamiltonian, k-SAT Hamiltonian, QAOA ansatz, and optimization methods.

Let's understand the code in detail:

1. The code starts with importing necessary libraries such as `numpy`, `scipy.optimize`, `random`, `time`, `functools`, and `List` from `typing`.

2. It then defines functions to convert a graph into a Hamiltonian, construct the Hamiltonian for the ZZ model (simpliest Ising example), and construct the Hamiltonian for k-SAT problems from random clauses.

3. Next, it defines functions to create different types of noise channels such as depolarization, amplitude damping, and phase flip.

4. After that, it defines a class `QAOA` which encapsulates the logic for the Quantum Approximate Optimization Algorithm. The class includes methods to apply gamma, Hx, and beta operators, generate QAOA ansatz state, apply ansatz to the state vector, compute expectation value, apply ansatz with noise, calculate the overlap of a state with the ground state, run the QAOA using heuristic L-BFGS-B optimization method, and compute the Quantum Fisher Information (QFI) matrix for parameter sensitivity analysis.

5. The class also includes a method to generate a list of indices for state vector swapping in the mixer.

6. Finally, the class includes methods to run the QAOA using the heuristic L-BFGS-B optimization method with layer-wise learning approach.
## License

This project is licensed under the MIT License.

## Acknowledgments

- Ernesto Luis Campos Espinoza
- Akshay Vishwanathan
- Andrey Kardashin

## Contact 
  For any comments or questions feel free to contact
  
  Dmitry Guskov, MSc student: dmitry.guskov@deepquantum.ai
© 2023 Deep Quantum Lab. All rights reserved.
