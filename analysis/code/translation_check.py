"""End-state translation/phase fits; distances are certified only as upper bounds."""
from pathlib import Path
import json,numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize
from numerical_core import Grid,phase_distance
B=Path(__file__).resolve().parents[1]; D=B/'data'; RAW=B.parent/'raw'
rows=[]
for name in ['scalar_generic96','full_generic96']:
    cfg=json.loads((RAW/name/'parameters.json').read_text()); g=Grid(cfg['n'],cfg['L']); x,y,z=g.xyz;r=np.sqrt(x*x+y*y+z*z)
    v=np.load(RAW/name/'final.npz')['u']; a=np.load(D/(cfg['profile']+'.npz')); q=CubicHermiteSpline(a['r'],a['Q'],a['Qp'])(r)
    f=g.ft(v); fq=g.ft(q); weight=1+g.k2; prod=weight*np.conj(fq)*f
    k=2*np.pi*np.fft.fftfreq(g.n,d=g.h); ks=[k[:,None,None],k[None,:,None],k[None,None,:]]
    Nq=np.sum(weight*abs(fq)**2); Nv=np.sum(weight*abs(f)**2)
    center=np.array([np.sum(c*abs(v)**2)/np.sum(abs(v)**2) for c in [x,y,z]])
    def objective(shift):
        phase=np.exp(1j*sum(ki*si for ki,si in zip(ks,shift))); inner=np.sum(prod*phase)
        value=(Nv+Nq-2*abs(inner))/Nq
        derivative=np.array([-2*np.real(np.conj(inner)*np.sum(1j*ki*prod*phase))/(abs(inner)*Nq) for ki in ks])
        return float(value),derivative
    opt=minimize(objective,center,jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':60})
    distance=np.sqrt(max(0.,objective(opt.x)[0])); original=phase_distance(g,v,q)[1]
    row={'name':name,'T':cfg['T'],'phase_only_H1':original,'translation_phase_H1_upper_bound':float(distance),'translation':opt.x.tolist(),'centroid':center.tolist(),'optimizer_success':bool(opt.success),'gradient_norm':float(np.linalg.norm(opt.jac)),'qualification':'A computed alignment upper bound, not a proof of global optimizer uniqueness or nonlinear orbital stability.'}
    rows.append(row);print(json.dumps(row),flush=True)
(D/'translation_checks.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
