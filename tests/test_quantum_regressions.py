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
