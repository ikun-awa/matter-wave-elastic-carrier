"""Compare full saved fields without deleting fine-grid modes."""
from pathlib import Path
import json,numpy as np,pandas as pd
from scipy.signal import resample
from numerical_core import Grid
B=Path(__file__).resolve().parents[1]; D=B/'data'
# Relative portable default; callers may explicitly route R and D.
R=B.parent/'raw'
def prolong(a,n):
    for axis in range(3):a=resample(a,n,axis=axis)
    return a
# Exact interpolation regression, including shifted centered grids.
x=np.arange(12)/12.; a=np.cos(2*np.pi*x)[:,None,None]*np.ones((1,12,12))
b=prolong(a,18)
assert np.max(abs(b-np.cos(2*np.pi*np.arange(18)/18.)[:,None,None]))<2e-14
# Nyquist split is handled by scipy.signal.resample; no pointwise subsampling.
def compare(coarse,ref):
    ca=json.loads((R/coarse/'parameters.json').read_text()); cb=json.loads((R/ref/'parameters.json').read_text())
    A=np.load(R/coarse/'final.npz'); Z=np.load(R/ref/'final.npz')
    if ca['T']!=cb['T']:raise ValueError('Observation times differ')
    samebox=ca['L']==cb['L']; out=dict(run=coarse,reference=ref,T=ca['T'],comparison='full-fine-grid' if samebox else 'aligned-common-cube plus exterior reference norm')
    for key in ['u','e','p']:
        aa=A[key]; zz=Z[key]
        if aa.size==0:continue
        if samebox:
            aa=prolong(aa,cb['n']) if ca['n']!=cb['n'] else aa
            zz=zz.copy(); g=Grid(cb['n'],cb['L']); exterior=0.
        else:
            assert abs(ca['h']-cb['h'])<1e-12 and cb['n']>ca['n']
            start=(cb['n']-ca['n'])//2; stop=start+ca['n']; zz=zz[start:stop,start:stop,start:stop].copy()
            g=Grid(ca['n'],ca['L']); exterior=max(0.,float(np.sum(abs(Z[key])**2)-np.sum(abs(zz)**2)))
        phase=1.
        if key=='u':
            inner=np.vdot(zz,aa); phase=inner/abs(inner) if abs(inner)>0 else 1.
        diff=aa-phase*zz; den=float(np.sum(abs(Z[key])**2))
        out[key+'_L2_relative']=float(np.sqrt((np.sum(abs(diff)**2)+exterior)/den)) if den>0 else None
        out[key+'_exterior_squared_fraction']=exterior/den if den>0 else None
        out[key+'_L2_absolute']=float(np.sqrt(g.dv*(np.sum(abs(diff)**2)+exterior)))
        if samebox and key in ['u','e']:
            fd=g.ft(diff); fz=g.ft(zz)
            out[key+'_H1_relative']=float(np.sqrt(np.sum((1+g.k2)*abs(fd)**2)/np.sum((1+g.k2)*abs(fz)**2)))
        if not samebox:
            out[key+'_common_cube_relative']=float(np.linalg.norm(diff.ravel())/np.linalg.norm(zz.ravel()))
    a=pd.read_csv(D/(coarse+'.csv')); b=pd.read_csv(D/(ref+'.csv'))
    assert np.array_equal(a.t,b.t)
    for key in ['phase_H1','core','rms']:
        out['max_'+key+'_difference']=float(np.max(abs(a[key]-b[key])))
    return out
if __name__=='__main__':
    pairs=[('scalar_mix96_T20','scalar_mix128'),('scalar_mix96_T20','scalar_time96'),('scalar_mix96_T20','scalar_box144'),('full_mix96','full_mix128'),('full_mix96','full_time96'),('full_mix96','full_box144')]
    results=[]; missing=[]
    for a,b in pairs:
        if not (R/a/'final.npz').exists() or not (R/b/'final.npz').exists():missing.append((a,b));continue
        result=compare(a,b); results.append(result); print('FIELD_COMPARISON',json.dumps(result),flush=True)
    pd.DataFrame(results).to_csv(D/'full_field_comparisons.csv',index=False)
    (B/'qa'/'comparison_status.json').write_text(json.dumps({'completed':len(results),'missing':missing,'fine_modes_discarded':False,'phase_rotation_real_fields':False},indent=2),encoding='utf-8')
    if missing:print('INCOMPLETE_COMPARISON_SET',missing,flush=True)
