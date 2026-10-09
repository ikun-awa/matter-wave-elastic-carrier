"""Unrestricted Cartesian finite-time tests; all fields evolved independently."""
from pathlib import Path
import argparse,json,time,datetime,hashlib
import numpy as np,pandas as pd
from scipy.interpolate import CubicHermiteSpline
from numerical_core import Grid,scalar_step,coupled_step,response_force,energy,phase_distance
BASE=Path(__file__).resolve().parents[1]; DATA=BASE/'data'; QA=BASE/'qa'
RAW=BASE.parent/'raw'
def perturb(g,Q,kind):
    x,y,z=g.xyz; r2=x*x+y*y+z*z; env=np.exp(-r2/100)
    if kind=='quadrupole':
        a=env*(x*x-y*y)/16; b=np.zeros(g.shape)
    else:
        rng=np.random.default_rng(20261005); ks=rng.normal(0,.35,(12,3)); weights=rng.normal(size=(2,12))
        a=np.zeros(g.shape); b=np.zeros(g.shape)
        for i,(kx,ky,kz) in enumerate(ks):
            mode=np.cos(kx*x+ky*y+kz*z+(float(rng.uniform(-np.pi,np.pi)) if kind=="generic" else 0.))
            a+=weights[0,i]*mode; b+=weights[1,i]*mode
        a*=env; b*=env
    s=Q*Q; den=np.sum(s)
    a-=np.sum(s*a)/den; b-=np.sum(s*b)/den
    norm=np.sqrt(np.sum(s*(a*a+b*b))/den)
    if norm<=0:raise ValueError('Degenerate perturbation')
    return a/norm,b/norm

def run(name,n=96,L=64.,dt=.04,T=40.,kind='mixed',delta=.05,full=False,alpha=1.,beta=.2):
    target=RAW/name
    if target.exists():raise FileExistsError(target)
    target.mkdir(); g=Grid(n,L,workers=4); x,y,z=g.xyz; r2=x*x+y*y+z*z; r=np.sqrt(r2)
    tag='coupled_b02' if full else 'scalar_w010'; src=DATA/(tag+'.npz')
    ss=np.load(src); assert r.max()<ss['r'][-1]
    Q=CubicHermiteSpline(ss['r'],ss['Q'],ss['Qp'])(r)
    E=CubicHermiteSpline(ss['r'],ss['E'],ss['Ep'])(r)
    a,b=perturb(g,Q,kind); u=(Q*(1+delta*(a+1j*b))).astype(complex)
    u*=np.sqrt(g.norm(Q)/g.norm(u))
    e=E*(1+(0.02 if delta else 0)*a); p=(.01 if delta else 0)*E*b
    cfg=dict(name=name,n=n,L=L,h=g.h,dt=dt,T=T,kind=kind,delta=delta,full=full,alpha=alpha,beta=beta,seed=20261005,profile=tag,profile_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),initial_e_amplitude=.02 if delta else 0.,initial_p_amplitude=.01 if delta else 0.,symmetry=('none imposed' if kind=='generic' else 'inversion-even but nonradial; no symmetry projection during evolution'),dtype='complex128/float64',save_interval=1.)
    (target/'parameters.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    np.savez_compressed(target/'initial.npz',u=u,e=e if full else np.empty(0),p=p if full else np.empty(0))
    rows=[]; start=time.time(); every=round(1/dt); steps=round(T/dt)
    assert abs(every*dt-1)<1e-12 and abs(steps*dt-T)<1e-12
    def record(t):
        norm=g.norm(u); l2,h1=phase_distance(g,u,Q)
        H=float(energy(g,u,e if full else None,p if full else None,alpha,beta))
        s=abs(u)**2; rms=np.sqrt(g.dv*np.sum(r2*s)/norm)
        centroid=[float(g.dv*np.sum(v*s)/norm) for v in [x,y,z]]
        row=dict(t=t,M=norm,H=H,phase_L2=l2,phase_H1=h1,rms=float(rms),core=float(g.dv*np.sum(s[r<=15])/norm),outer=float(g.dv*np.sum(s[r>=.4*L])/norm),peak=float(s.max()),centroid_norm=float(np.linalg.norm(centroid)))
        if full:row.update(e_min=float(e.min()),e_max=float(e.max()),p_L2=float(np.sqrt(g.norm(p))),response_defect_L2=float(np.sqrt(g.norm(e-s/(1+s)))))
        rows.append(row)
        if len(rows)%10==1:
            pd.DataFrame(rows).to_csv(DATA/(name+'.csv'),index=False)
            print('PROGRESS',name,t,'Mdrift',norm/rows[0]['M']-1,'H1',h1,'seconds',round(time.time()-start,1),flush=True)
    record(0.)
    force=response_force(g,u,e,alpha,beta) if full else None
    for j in range(1,steps+1):
        if full:u,e,p,force=coupled_step(g,u,e,p,dt,alpha,beta,force,True)
        else:u=scalar_step(g,u,dt)
        if j%every==0:record(j*dt)
        if not np.isfinite(u.flat[0]):raise RuntimeError('Nonfinite solution')
    df=pd.DataFrame(rows); df.to_csv(DATA/(name+'.csv'),index=False)
    np.savez_compressed(target/'final.npz',u=u,e=e if full else np.empty(0),p=p if full else np.empty(0))
    np.savez_compressed(DATA/(name+'_slice.npz'),axis=(np.arange(n)-n//2)*g.h,density=abs(u[:,:,n//2])**2,e=e[:,:,n//2] if full else np.empty(0))
    out=dict(cfg,completed=True,seconds=time.time()-start,initial_L2=float(df.phase_L2.iloc[0]),initial_H1=float(df.phase_H1.iloc[0]),max_H1=float(df.phase_H1.max()),max_L2=float(df.phase_L2.max()),max_mass_drift=float(np.max(abs(df.M/df.M.iloc[0]-1))),max_energy_drift=float(np.max(abs(df.H-df.H.iloc[0]))/max(1,abs(df.H.iloc[0]))),min_core=float(df.core.min()),max_outer=float(df.outer.max()),max_centroid=float(df.centroid_norm.max()),max_rms_change=float(np.max(abs(df.rms/df.rms.iloc[0]-1))),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if full:out.update(e_min=float(df.e_min.min()),e_max=float(df.e_max.max()),p_final_L2=float(df.p_L2.iloc[-1]))
    (DATA/(name+'.json')).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('COMPLETED',json.dumps(out),flush=True)
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--name',required=True); ap.add_argument('--n',type=int,default=96); ap.add_argument('--L',type=float,default=64); ap.add_argument('--dt',type=float,default=.04); ap.add_argument('--T',type=float,default=40); ap.add_argument('--kind',default='mixed'); ap.add_argument('--delta',type=float,default=.05); ap.add_argument('--full',action='store_true'); ap.add_argument('--alpha',type=float,default=1.)
    a=vars(ap.parse_args()); run(**a)
