# Quantum validation and API corrections

The ideal statevector uses H = sum ZiZj, chronological layers exp(-i beta B) exp(-i gamma H), and an angle vector containing all gammas followed by all betas. Qubit zero is the most significant tensor factor. Energies minimize this Ising H; for an unweighted graph the expected cut is (number of edges minus energy)/2.

Noise is applied independently on every qubit after each complete cost-plus-mixer layer. Channel factories return (operators, probabilities). Phase-flip and depolarizing factories supply unweighted unitary operators with actual probabilities. Depolarization is E(rho) = (1-p)rho + p I/2 on a single qubit. Amplitude damping supplies the standard Kraus operators with probabilities=None because its branch probabilities depend on the input state. Custom operators with probabilities=None must already be Kraus weighted and satisfy completeness.

Trajectories sample Born probabilities and normalize each branch. expectation_noise averages scalar energies. With return_state=True it now returns (energy, empirical_density_matrix), not an averaged statevector. Callers relying on the old vector must migrate; old noisy outputs must be regenerated. Pass rng=np.random.default_rng(seed) for reproducibility. The default uses NumPy's global random generator for compatibility. Trajectory count is a Monte Carlo sampling budget, not hardware shots; estimate sampling uncertainty when using results in research.

run_QFI now performs damped natural-gradient descent with a feasible Armijo line search. It no longer passes F inverse times gradient as an L-BFGS-B derivative. It returns a scipy OptimizeResult and records qfi_evaluations separately from energy calls. estimate_QFI_iter remains accepted but no longer invents cost-history entries. Bounds remain finite angle boxes, defaulting to [0, 2*pi]. A stationary result is not a certificate of a global optimum. PROTES and CMA-ES are optional imports loaded only by their optimizer methods.

The one-qubit mixer now works. Where exposed, construct_QAOA_operator_term returns the full Heisenberg matrix U† H U and accepts an explicitly zero diagonal observable. qaoa_operator retains its historical row-vector return convention U.T; use its transpose for a column-vector circuit matrix.

Run `python -m pip install -r requirements-test.txt` then `python -m pytest tests -q` from this repository. Tests import the actual modules and compare states, QFI and gradients to independent dense evolution and finite differences; test full Heisenberg matrices, Kraus completeness and endpoints, density-matrix agreement, local ordering, and singular-metric optimization. Historical notebooks and optional optimizers outside these tests are not comprehensively validated.


## Further optimizer and input corrections

Layerwise optimization now selects each optimized restart's angles, including at depth one, and counts all restart, appended-layer, and joint-refinement energy calls. It accepts a deterministic seed and returns an OptimizeResult with the actual stopping depth. `lw_log=None` means the numerical ground-energy tolerance was never reached. If early stopping selects fewer than the requested layers, `result.x` contains the shorter circuit while `opt_angles` pads it with identity layers to remain a valid requested-depth warm start. A first-hit depth is an upper bound from finite optimization, not a minimum-depth proof.

CMA-ES now saves its best evaluated sample where exposed, accepts an explicit seed, and restores cost tracking if its backend raises. The MCTS method in QAOA_class likewise saves its best evaluated complete circuit and tracks the actual evaluation history. Previously these methods could report a last sample or an unrelated random completion. Old optimization results from these paths require regeneration before research use.

Hamiltonian input must be a finite real diagonal vector of power-of-two length, with positive integer depth. A valid all-zero angle warm start is preserved. `H_zz_Ising` uses the indexed periodic bond sum: one closed site gives the constant identity term, two closed sites count the bond twice, and one open site has no bonds. This convention is explicit so small-chain comparisons do not silently disagree.
