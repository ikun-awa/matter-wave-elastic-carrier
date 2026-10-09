from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.fft import dst

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data'; QA=BASE/'qa'
branch=pd.read_csv(DATA/'branch.csv')
spec=pd.read_csv(DATA/'spectral_branch.csv')
runs=pd.read_csv(DATA/'evolution_runs.csv')

assert len(branch)==43 and (branch.status==0).all()
assert branch.point_residual.max()<2e-8
assert branch[['rel_identity1','rel_identity2']].to_numpy().max()<5e-11
imin=branch.M.idxmin(); mrow=branch.loc[imin]
assert .049 <= mrow.omega <= .052

assert (spec.Lplus_l0_eig0<0).all()
assert (spec.Lplus_l0_eig1>0).all()
assert np.max(np.abs(spec.Lminus_l0_eig0))<3e-5
assert np.max(np.abs(spec.Lplus_l1_eig0))<5e-5
assert (spec.Lplus_l2_eig0>0).all()
def final_state(label):
    full=DATA/f'evolution_{label}.npz'
    if full.exists():
        z=np.load(full); return z['r'],z['v'][-1]
    z=np.load(DATA/f'evolution_{label}_final.npz')
    return z['r'],z['v_final']

def onto(r,v,target):
    L=r[-1]+r[0]
    sp=CubicSpline(np.r_[0,r,L],np.r_[0j,v,0j])
    return sp(target)

def compare(base_label,test_label):
    rb,vb=final_state(base_label); rt,vt=final_state(test_label)
    vt=onto(rt,vt,rb)
    overlap=np.vdot(vb,vt)
    vt*=np.exp(-1j*np.angle(overlap))
    h=rb[0]; L=h*(len(rb)+1)
    k=np.pi*np.arange(1,len(rb)+1)/L
    d=vt-vb
    dc=dst(d,type=1,norm='ortho'); bc=dst(vb,type=1,norm='ortho')
    rel_l2=np.linalg.norm(d)/np.linalg.norm(vb)
    rel_h1=np.sqrt((np.sum(abs(d)**2)+np.sum(k*k*abs(dc)**2))/
                   (np.sum(abs(vb)**2)+np.sum(k*k*abs(bc)**2)))
    return dict(base=base_label,test=test_label,
                relative_L2=float(rel_l2),relative_H1=float(rel_h1))
rows=[]
for family in ['stable','unstable']:
    base=f'{family}_plus05'
    for suffix in ['time_coarse','time_fine','space_coarse','space_fine','domain480']:
        rows.append(compare(base,f'{family}_{suffix}'))
pd.DataFrame(rows).to_csv(DATA/'evolution_field_convergence.csv',index=False)

left_df=branch[(branch.omega>=.03)&(branch.omega<=.045)]
right_df=branch[(branch.omega>=.06)&(branch.omega<=.1)]
left=np.polyfit(left_df.omega,left_df.M,1)[0]
right=np.polyfit(right_df.omega,right_df.M,1)[0]
assert left<0 and right>0

stable=runs[runs.label=='stable_plus05'].iloc[0]
unstable=runs[runs.label=='unstable_plus05'].iloc[0]
assert stable.max_H1<0.03
assert unstable.max_H1>0.3

report={
 'result':'PASS',
 'branch_points':int(len(branch)),
 'sampled_mass_minimum':{'omega':float(mrow.omega),'M':float(mrow.M),'Q0':float(mrow.Q0)},
 'max_point_residual':float(branch.point_residual.max()),
 'max_integral_identity_relative_residual':float(branch[['rel_identity1','rel_identity2']].to_numpy().max()),
 'sampled_slope_signs':{'lower_branch_fit':float(left),'upper_branch_fit':float(right)},
 'spectral_structure':{
   'Lplus_l0_negative_count_sampled':1,
   'phase_zero_max_abs':float(np.max(np.abs(spec.Lminus_l0_eig0))),
   'translation_zero_max_abs':float(np.max(np.abs(spec.Lplus_l1_eig0))),
   'l2_lowest_min':float(spec.Lplus_l2_eig0.min())},
 'radial_dynamics':{
   'omega_0.10_plus5_max_H1':float(stable.max_H1),
   'omega_0.10_plus5_final_H1':float(stable.final_H1),
   'omega_0.03_plus5_max_H1':float(unstable.max_H1),
   'omega_0.03_plus5_final_H1':float(unstable.final_H1)},
 'qualification':'Numerical evidence for a VK/GSS-type transition and finite-time radial behavior. Not a rigorous orbital-stability proof and not a nonradial 3D evolution test.'
}
(QA/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
