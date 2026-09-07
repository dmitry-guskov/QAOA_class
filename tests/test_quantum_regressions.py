import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm

FILES = ['src/qaoa.py']

@pytest.fixture(params=FILES)
def module(request):
    path = Path(__file__).resolve().parents[1] / request.param
    spec = importlib.util.spec_from_file_location("qaoa_under_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])

def local(op, q, n):
    out = np.ones((1, 1))
    for i in range(n):
        out = np.kron(out, op if i == q else np.eye(2))
    return out

def dense_u(h, angles):
    n, p = int(np.log2(len(h))), len(angles)//2
    b = sum(local(X, q, n) for q in range(n))
    u = np.eye(len(h), dtype=complex)
    for layer in range(p):
        u = expm(-1j*angles[p+layer]*b) @ np.diag(np.exp(-1j*angles[layer]*h)) @ u
    return u

@pytest.mark.parametrize('n,p', [(1,1), (2,2), (3,3)])
def test_state_qfi_and_full_heisenberg(module, n, p):
    rng = np.random.default_rng(100+n+p)
    h = rng.normal(size=2**n)  # asymmetric ordering and general diagonal H
    a = rng.normal(size=2*p)
    q = module.QAOA(p, h)
    plus = np.ones(2**n)/np.sqrt(2**n)
    u = dense_u(h, a)
    state = u @ plus
    np.testing.assert_allclose(q.qaoa_ansatz(a), state, atol=2e-14)
    eps = 1e-6
    derivatives = np.array([(dense_u(h, a+eps*row)@plus-dense_u(h,a-eps*row)@plus)/(2*eps) for row in np.eye(2*p)])
    overlaps = derivatives.conj() @ state
    fisher = 4*np.real(derivatives.conj()@derivatives.T-np.outer(overlaps, overlaps.conj()))
    gradient = 2*np.real(derivatives.conj()@(h*state))
    got_f, got_g = q.qaoa_qfi_matrix(a, return_grad=True)
    np.testing.assert_allclose(got_f, fisher, atol=2e-8)
    np.testing.assert_allclose(got_g, gradient, atol=2e-8)
    if hasattr(q, 'construct_QAOA_operator_term'):
        np.testing.assert_allclose(q.construct_QAOA_operator_term(a), u.conj().T@np.diag(h)@u, atol=2e-14)
        np.testing.assert_allclose(q.construct_QAOA_operator_term(a, np.zeros_like(h)), 0, atol=2e-14)

@pytest.mark.parametrize('factory', ['create_phase_flip_kraus','create_depolarization_kraus','create_amplitude_damping_kraus'])
@pytest.mark.parametrize('p', [0, .37, 1])
def test_channel_completeness(module, factory, p):
    ops, probabilities = getattr(module, factory)(p)
    ks = module._channel_kraus(probabilities, ops)
    np.testing.assert_allclose(sum(k.conj().T@k for k in ks), np.eye(2), atol=1e-14)
    for invalid in [-.01, 1.01, np.nan]:
        with pytest.raises(ValueError):
            getattr(module, factory)(invalid)

@pytest.mark.parametrize('factory', ['create_phase_flip_kraus','create_depolarization_kraus','create_amplitude_damping_kraus'])
def test_trajectories_match_exact_density(module, factory):
    # No phase symmetry: two unequal layers and a nonsymmetric diagonal observable.
    h = np.array([.3, -.7, 1.1, -.2])
    q = module.QAOA(2, h)
    angles = np.array([.23, -.41, .37, .18])
    ops, probabilities = getattr(module, factory)(.38)
    ks = module._channel_kraus(probabilities, ops)
    rho = np.ones((4,4), complex)/4
    for l in range(2):
        u = dense_u(h, [angles[l], angles[2+l]])
        rho = u@rho@u.conj().T
        for qubit in range(2):
            rho = sum(local(k,qubit,2)@rho@local(k,qubit,2).conj().T for k in ks)
    energy, estimate = q.expectation_noise(angles, probabilities, ops, 3500, True, np.random.default_rng(79))
    # Each density entry is bounded; this broad deterministic MC tolerance is
    # complemented by the exact channel and endpoint tests below.
    np.testing.assert_allclose(estimate, rho, atol=.045)
    assert abs(energy-np.trace(np.diag(h)@rho).real) < .055
    assert abs(np.trace(estimate)-1) < 1e-13
    assert np.linalg.eigvalsh(estimate).min() >= -1e-13
    assert abs(energy-np.dot(h, np.diag(estimate)).real) < 1e-13

def test_phase_flip_and_amplitude_endpoints(module):
    h = np.array([1., -1., -1., 1.])
    q = module.QAOA(1, h)
    ops, probs = module.create_phase_flip_kraus(.5)
    actual = q.expectation_noise([.23,.37], probs, ops, 31, rng=np.random.default_rng(9))
    assert abs(actual-.4421194156957086) < 1e-13
    ops, probs = module.create_amplitude_damping_kraus(1)
    actual, rho = q.expectation_noise([.23,.37], probs, ops, 17, True, np.random.default_rng(8))
    np.testing.assert_allclose(rho, np.diag([1,0,0,0]), atol=1e-14)
    # Check local ordering on |11>, independently of a symmetric final state.
    state = module._local_channel_trajectory(np.array([0,0,0,1], complex), ops, 0, 2, np.random.default_rng(3))
    np.testing.assert_allclose(state, [0,1,0,0], atol=1e-14)
    with pytest.raises(ValueError):
        q.expectation_noise([.23,.37], probs, ops, 0)
    with pytest.raises(ValueError):
        q.apply_ansatz_noise([.23,.37], None, [np.zeros((2,2))])

def test_natural_gradient_descent_and_singular_metric(module):
    q = module.QAOA(1, np.array([1.,-1.]))
    start = np.array([.23,-.37])
    original = start.copy()
    result = q.run_QFI(initial_params=start, bds=[(-np.pi,np.pi)]*2, maxiter=150)
    np.testing.assert_array_equal(start, original)
    assert result.fun < -.999999
    assert np.isfinite(result.x).all()
    assert q.qfi_evaluations == result.njev
    flat = module.QAOA(2, np.zeros(4))
    result = flat.run_QFI(initial_params=np.zeros(4), maxiter=5, track_energy=True)
    assert result.success and result.fun == 0
    assert flat.track_cost is False


@pytest.mark.parametrize('depth,stop', [(1,False),(3,True),(2,False)])
def test_layerwise_selected_angles_and_total_accounting(module, depth, stop):
    q = module.QAOA(depth, np.array([1., -1.]))
    before = q.eval_num
    result = q.run_heuristic_LW(True, 3, 2, stop_on_min=stop, seed=314)
    assert q.eval_num-before == result.nfev == len(q.tracked_cost)
    assert q.track_cost is False
    assert result.fun < -.9999
    np.testing.assert_allclose(result.fun, np.vdot(q.f_state, q.H*q.f_state).real, atol=1e-14)
    np.testing.assert_allclose(result.fun, q.expectation(result.x), atol=1e-14)
    assert len(result.x) == 2*result.depth
    if stop:
        assert result.depth == 1
        assert len(q.opt_angles) == 2*q.p
        assert abs(q.expectation(q.opt_angles)-result.fun) < 1e-13
        continued = q.run_QFI(maxiter=0)
        assert len(continued.x) == 2*q.p
    repeated = module.QAOA(depth, q.H)
    other = repeated.run_heuristic_LW(False, 3, 2, stop_on_min=stop, seed=314)
    np.testing.assert_array_equal(result.x, other.x)


def test_layerwise_non_hit_and_bounds(module):
    q = module.QAOA(2, np.array([1., -1.]))
    result = q.run_heuristic_LW(False, 1, 1, bds=[(0.,0.)]*4, seed=2)
    assert q.lw_log is None and result.depth == 2
    assert abs(result.fun) < 1e-14
    np.testing.assert_array_equal(result.x, np.zeros(4))


def test_zero_initial_angles_are_preserved(module, monkeypatch):
    from scipy.optimize import OptimizeResult
    q = module.QAOA(1, np.array([1., -1.]))
    q.opt_angles = np.zeros(2)
    def optimizer(fun, x, **kwargs):
        np.testing.assert_array_equal(x, [0,0])
        return OptimizeResult(x=np.asarray(x), fun=fun(x), nfev=1, nit=0)
    monkeypatch.setattr(module, 'minimize', optimizer)
    q.run()
    np.testing.assert_array_equal(q.opt_angles, [0,0])


def test_cmaes_saves_best_sample(module, monkeypatch):
    if not hasattr(module.QAOA, 'run_cmaes'):
        return  # qaoa_old intentionally exposes no CMA implementation.
    import sys, types
    samples = [np.array([np.pi/4, 3*np.pi/4]), np.zeros(2), np.array([np.pi/4,np.pi/4])]
    class FakeCMA:
        population_size = 3
        def __init__(self, **kwargs): self.index=0
        def ask(self):
            value=samples[self.index]; self.index+=1; return value
        def tell(self, solutions): pass
        def should_stop(self): return False
    monkeypatch.setitem(sys.modules, 'cmaes', types.SimpleNamespace(CMA=FakeCMA))
    q = module.QAOA(1, np.array([1.,-1.]))
    expected = [q.expectation(a) for a in samples]
    q.run_cmaes(generations=1, initial_params=np.zeros(2))
    assert q.q_energy == min(expected) and q.opt_iter == 3
    np.testing.assert_array_equal(q.opt_angles, samples[int(np.argmin(expected))])
    assert abs(q.expectation(q.opt_angles)-q.q_energy) < 1e-14


def test_mcts_saves_best_evaluated_circuit(module):
    if not hasattr(module.QAOA, 'run_mcts'):
        return
    q = module.QAOA(2, np.array([.1,-.7,.4,1.]))
    np.random.seed(7)
    q.run_mcts(track_energy=True, b=7, simulations=31)
    assert q.q_energy == min(q.tracked_cost)
    assert q.opt_iter == len(q.tracked_cost) == 31
    assert q.expectation(q.opt_angles) == q.q_energy
    assert q.track_cost is False


def test_diagonal_validation_and_periodic_ising_convention(module):
    for bad in [[], [1,2,3], [[1,2],[3,4]], [1,np.nan], [1,1j]]:
        with pytest.raises(ValueError): module.QAOA(1, bad)
    for p in [0, 1.5]:
        with pytest.raises(ValueError): module.QAOA(p, [1,-1])
    np.testing.assert_array_equal(module.H_zz_Ising(1), [1,1])
    np.testing.assert_array_equal(module.H_zz_Ising(1, 'open'), [0,0])
    np.testing.assert_array_equal(module.H_zz_Ising(2), [2,-2,-2,2])


def test_cmaes_restores_tracking_on_backend_failure(module, monkeypatch):
    import sys, types
    class BrokenCMA:
        population_size=2
        def __init__(self, **kwargs): pass
        def ask(self): raise RuntimeError('backend failure')
    monkeypatch.setitem(sys.modules,'cmaes',types.SimpleNamespace(CMA=BrokenCMA))
    q=module.QAOA(1,[1.,-1.])
    with pytest.raises(RuntimeError):
        q.run_cmaes(generations=1,track_energy=True,initial_params=[0.,0.],seed=1)
    assert q.track_cost is False
