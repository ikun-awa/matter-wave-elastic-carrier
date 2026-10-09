"""Deterministic regression tests for the new numerical core."""
from pathlib import Path
import importlib.util, numpy as np
p=Path(__file__).with_name('numerical_core.py')
assert p.exists(), 'Numerical core has not been implemented'
spec=importlib.util.spec_from_file_location('core',p)
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
g=c.Grid(12,12.,workers=1)
x,y,z=g.xyz; mode=np.cos(2*np.pi*x/12)+0*y+0*z
assert np.max(abs(g.lap(mode)+(2*np.pi/12)**2*mode))<1e-12
u=np.exp(2j*np.pi*(x+y)/12)+0*z
v=g.free(u,.037)
assert np.max(abs(v-u*np.exp(-2j*(2*np.pi/12)**2*.037)))<1e-12
rng=np.random.default_rng(775); u=.1*(rng.normal(size=g.shape)+1j*rng.normal(size=g.shape))
v=c.scalar_step(g,u.copy(),.002)
assert abs(np.sum(abs(v)**2)/np.sum(abs(u)**2)-1)<2e-14
assert np.max(abs(c.scalar_step(g,v,-.002)-u))<2e-13
u=np.ones(g.shape,dtype=complex)*.7
v=c.scalar_step(g,u,.013)
assert np.max(abs(v-.7*np.exp(1j*.013*.49/1.49)))<2e-14
u=.1*(rng.normal(size=g.shape)+1j*rng.normal(size=g.shape)); e=.1+.01*rng.normal(size=g.shape); p=.01*rng.normal(size=g.shape)
u1,e1,p1=c.coupled_step(g,u.copy(),e.copy(),p.copy(),.001,.5,.2)
u2,e2,p2=c.coupled_step(g,u1,e1,p1,-.001,.5,.2)
assert max(np.max(abs(u2-u)),np.max(abs(e2-e)),np.max(abs(p2-p)))<1e-12
assert abs(np.sum(abs(u1)**2)/np.sum(abs(u)**2)-1)<2e-14
try:c.coupled_step(g,u,np.ones(g.shape),p,.001,.5,.2)
except ValueError:pass
else:raise AssertionError('Singular response must be rejected, never clipped')
print('CORE_REGRESSION_PASS: dispersion, mass, reversibility, nonlinear phase, response-domain guard')

# Hamiltonian response force is tested independently against an energy derivative.
z=rng.normal(size=g.shape); eps=1e-6
dE=(c.energy(g,u,e+eps*z,p,.5,.2)-c.energy(g,u,e-eps*z,p,.5,.2))/(2*eps)
expected=-.5*g.dv*np.sum(c.response_force(g,u,e,.5,.2)*z)
assert abs(dE-expected)/max(1,abs(expected))<1e-7
print("HAMILTONIAN_FORCE_DERIVATIVE_PASS")
