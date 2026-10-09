"""Exact algebra plus diagnostics on frozen CQ data; does not solve a new PDE."""
from pathlib import Path
import json, hashlib, platform, datetime
import sympy as sp
import numpy as np
import pandas as pd
from scipy.integrate import simpson
BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data'; QA=BASE/'qa'
checks={}
def check(name,expression):
    result=sp.simplify(expression)
    if result != 0: raise AssertionError((name,result))
    checks[name]='exact zero'
s,q,e,K,A,B,gamma,T,N,Q4,Q6,w,lam=sp.symbols('s q e K A B gamma T N Q4 Q6 w lam',positive=True)
a1=gamma/K; a2=-A*gamma**2/K**3; a3=2*A**2*gamma**3/K**5-B*gamma**3/K**4
ee=a1*q+a2*q**2+a3*q**3
check('implicit_coefficients_through_q3',sp.series(K*ee+A*ee**2+B*ee**3-gamma*q,q,0,4).removeO())
F=-gamma**2*q**2/(2*K)+A*gamma**3*q**3/(3*K**3)+gamma**4*(B/(4*K**4)-A**2/(2*K**5))*q**4
check('reduced_energy_derivative',sp.diff(F,q)+gamma*ee)
W=K*(-sp.log(1-e)-e); eq=gamma*q/(K+gamma*q)
check('exact_stationary_elastic_equation',sp.diff(W,e).subs(e,eq)-gamma*q)
Fs=K*sp.log(1+gamma*q/K)-gamma*q
check('exact_reduced_derivative',sp.diff(Fs,q)+gamma*eq)
check('exact_reduced_value',sp.expand_log((W-gamma*e*q).subs(e,eq)-Fs,force=True))
fs=-s/(1+s); fcq=-s+s**2; bfun=s**3/(1+s)
check('nonlinear_coefficient_remainder',fcq-fs-bfun)
check('relative_response_error',sp.cancel((fcq-fs)/(-fs))-s**2)
check('energy_remainder_derivative',sp.diff(-s**2/2+s**3/3-sp.log(1+s)+s,s)-bfun)
rad=s*(s+3)/(1+s)**2
check('radial_jacobian_derivative',sp.diff(rad,s)-(3-s)/(1+s)**3)
check('global_lipschitz_max',rad.subs(s,3)-sp.Rational(9,8))
H=2*s-3*sp.log(1+s)+s/(1+s)
check('saturating_pohozaev_positive_integrand',sp.diff(H,s)-s*(1+2*s)/(1+s)**2)
Escale=lam**2*T-lam**3*Q4/2+lam**6*Q6/3
scale_T=sp.Rational(3,4)*Q4-Q6
check('cq_scale_second_derivative',sp.diff(Escale,lam,2).subs({lam:1,T:scale_T})+3*Q4/2-8*Q6)
check('cq_energy_on_scaling_constraint',(T-Q4/2+Q6/3).subs(T,scale_T)-Q4/4+2*Q6/3)
check('cq_energy_square',-s**2/2+s**3/3-(s*(s-sp.Rational(3,4))**2/3-sp.Rational(3,16)*s))
check('cq_stationarity_defect',(s-s**2)-s/(1+s)+bfun)
check('pohozaev_cq_relation',(T+w*N-Q4+Q6)-(T/2+3*w*N/2-3*Q4/4+Q6/2)*2+2*w*N-Q4/2)
checks['energetic_scale_floor']=str(sp.Rational(3,16)**2)
checks['negative_energy_peak_floor']=str(sp.Rational(3,8)**2)
rows=[]
branch=pd.read_csv(DATA/'branch.csv')
assert len(branch)==33 and (branch.status==0).all()
profiles=[(row.tag,float(row.omega),'branch') for row in branch.itertuples()]
profiles.append(('fine_0.100',.1,'fine'))
for tag,omega,kind in profiles:
    z=np.load(DATA/f'bvp_{tag}.npz'); r=z['r']; Q=z['Q']; Qp=z['Qp']; ss=Q**2
    assert np.all(np.diff(r)>0) and np.isfinite(Q).all()
    def integ(y):return float(4*np.pi*simpson(y*r*r,x=r))
    norm=integ(ss); kinetic=integ(Qp**2); i4=integ(ss**2); i6=integ(ss**3)
    b=ss**3/(1+ss); mean_b=integ(b*ss)/norm
    residual_fixed=np.sqrt(integ(b*b*ss)/norm)
    residual_opt=np.sqrt(integ((b-mean_b)**2*ss)/norm)
    energy_cq=kinetic-i4/2+i6/3
    energy_sat=kinetic+integ(np.log1p(ss)-ss)
    rows.append(dict(tag=tag,kind=kind,omega=omega,Q0=float(Q[0]),N=norm,E_CQ=energy_cq,E_sat_evaluated_on_CQ=energy_sat,eta_peak=float(ss.max()),response_error_peak=float(ss.max()**2),frequency_shift_optimal=mean_b,residual_fixed_frequency=residual_fixed,residual_best_frequency=residual_opt,scale_curvature=-1.5*i4+8*i6,Q6_over_Q4=i6/i4))
    assert energy_cq>=energy_sat-1e-7
    assert residual_opt<=residual_fixed and residual_opt>0
pd.DataFrame(rows).to_csv(DATA/'cq_saturation_diagnostics.csv',index=False)
fine=[row for row in rows if row['kind']=='fine'][0]
# Independent quadrature refinement for diagnostic values at omega=0.1.
z=np.load(DATA/'bvp_fine_0.100.npz'); rr=z['r']; qq=z['Q']; convergence=[]
for stride in [4,2,1]:
    r=rr[::stride]; Q=qq[::stride]; ss=Q**2; b=ss**3/(1+ss)
    norm=4*np.pi*simpson(ss*r*r,x=r); mu=4*np.pi*simpson(b*ss*r*r,x=r)/norm
    std=np.sqrt(4*np.pi*simpson((b-mu)**2*ss*r*r,x=r)/norm)
    convergence.append(dict(stride=stride,grid_step=float(r[1]-r[0]),N=float(norm),mean=float(mu),std=float(std)))
pd.DataFrame(convergence).to_csv(DATA/'diagnostic_quadrature_check.csv',index=False)
report={'version':'elastic-feedback-focused-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'algebra_checks':checks,'algebra_count':len(checks)-2,'fine_profile_diagnostics':fine,'quadrature':convergence,'derived_rows':len(rows),'new_PDE_simulation':False,'environment':{'python':platform.python_version(),'sympy':sp.__version__,'numpy':np.__version__},'qualification':'Symbolic identities do not replace functional-analytic proofs. Diagnostics evaluate equation mismatch on CQ profiles; not solution-distance error or exact-saturation evolution.'}
(QA/'focused_mathematical_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
manifest={}
for p in sorted(DATA.iterdir()):
    if p.is_file():manifest[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(QA/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
cp=json.loads((BASE/'CHECKPOINT.json').read_text(encoding='utf-8'))
cp.update(stage='proof_algebra_and_new_diagnostics_verified',next='Write focused manuscript and synchronize new document',utc=report['utc'])
(BASE/'CHECKPOINT.json').write_text(json.dumps(cp,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
