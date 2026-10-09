"""Reproducible 3D radial benchmarks for the exact saturating model.
Equation: i u_t = -Delta u - |u|^2/(1+|u|^2) u.
Numerical results are radial finite-time evidence, not an orbital-stability theorem.
"""
from pathlib import Path
import json, platform, datetime, hashlib
import numpy as np, pandas as pd, scipy
from scipy.integrate import solve_bvp, simpson
from scipy.fft import dst, idst
from scipy.linalg import eigh_tridiagonal
BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data'; QA=BASE/'qa'
for p in (DATA,QA): p.mkdir(parents=True,exist_ok=True)
def dump(p,o): p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def bvp(w,L=120.,n=1201,tol=1e-8,guess=None,A=.9,R=5.):
    r=np.linspace(0,L,n)
    if guess is None:
        q=A*np.exp(-(r/R)**2); y=np.vstack((q,-2*r*q/R**2))
    else: y=guess.sol(r)
    def f(r,y):
        q=y[0]; return np.vstack((y[1],w*q-q**3/(1+q**2)))
    def bc(a,b): return np.array([a[1],b[1]+(np.sqrt(w)+1/L)*b[0]])
    sol=solve_bvp(f,bc,r,y,S=np.array([[0.,0.],[0.,-2.]]),
                  tol=tol,bc_tol=tol,max_nodes=80000)
    if not sol.success or sol.y[0,0] <= 0: raise RuntimeError((w,sol.message))
    return sol
def measures(sol,w,tag,dr=.01):
    L=float(sol.x[-1]); r=np.arange(0,L+dr/2,dr); q,p=sol.sol(r)
    def I(y): return float(4*np.pi*simpson(y*r*r,x=r))
    M=I(q*q); T=I(p*p); H=T+I(np.log1p(q*q)-q*q)
    A1=I(q*q/(1+q*q)); B1=I(np.log1p(q*q))
    id1=T+(w-1)*M+A1; id2=T+3*(w-1)*M+3*B1
    z=sol.x[:-1]+0.314159265*np.diff(sol.x)
    qz,pz=sol.sol(z); ppz=sol.sol(z,1)[1]
    rr=ppz+2*pz/z-w*qz+qz**3/(1+qz**2)
    sc=1+abs(ppz)+abs(2*pz/z)+w*abs(qz)+abs(qz**3/(1+qz**2))
    row=dict(tag=tag,omega=w,L=L,nodes=len(sol.x),status=int(sol.status),
             Q0=float(q[0]),M=M,H=H,T=T,
             point_residual=float(np.max(abs(rr)/sc)),
             identity1=id1,identity2=id2,
             rel_identity1=abs(id1)/(T+abs(w-1)*M+A1),
             rel_identity2=abs(id2)/(T+3*abs(w-1)*M+3*B1))
    np.savez_compressed(DATA/f'bvp_{tag}.npz',r=r,Q=q,Qp=p,mesh=sol.x,mesh_y=sol.y)
    return row
def branch():
    freqs=list(np.round(np.arange(.01,.201,.005),5))+[.25,.3,.4,.5,.6,.7,.8,.9]
    sols={}; rows=[]; last=None
    for w in freqs:
        L=180. if w<=.02 else 120.; n=1801 if L>120 else 1201
        sol=bvp(w,L=L,n=n,tol=1e-8,guess=last); last=sol; sols[w]=sol
        rows.append(measures(sol,w,f'branch_{w:.5f}'))
    df=pd.DataFrame(rows); df.to_csv(DATA/'saturating_branch.csv',index=False)
    return sols,df
def convergence(sols):
    rows=[]
    for w in [.03,.1]:
        seed=sols[w]
        configs=[('coarse',120.,601,1e-6),('medium',120.,1201,1e-8),
                 ('fine',120.,2401,1e-10),('domain80',80.,1601,1e-10),
                 ('domain160',160.,3201,1e-10)]
        block=[]
        for name,L,n,tol in configs:
            sol=bvp(w,L=L,n=n,tol=tol,guess=seed)
            row=measures(sol,w,f'{name}_{w:.3f}'); row.update(tolerance=tol,seed_nodes=n)
            block.append(row)
        ref=block[-1]
        for x in block:
            x['rel_M_to_ref']=abs(x['M']-ref['M'])/ref['M']
            x['rel_H_to_ref']=abs(x['H']-ref['H'])/max(1,abs(ref['H']))
        rows.extend(block)
    pd.DataFrame(rows).to_csv(DATA/'saturating_bvp_convergence.csv',index=False)
def radial_eigs(sol,w,L=80.,dr=.025,nev=6):
    r=np.arange(dr,L,dr); q=sol.sol(r)[0]; s=q*q
    out={}
    for ell in [0,1,2]:
        centrifugal=ell*(ell+1)/r**2
        for kind in ['minus','plus']:
            V=w-s/(1+s) if kind=='minus' else w-s*(s+3)/(1+s)**2
            d=2/dr**2+V+centrifugal; e=-np.ones(len(r)-1)/dr**2
            vals=eigh_tridiagonal(d,e,select='i',select_range=(0,nev-1),
                                  check_finite=False)[0]
            out[f'L{kind}_l{ell}']=vals.tolist()
    return out
def evolution(sol,w,label,L=320.,n=4095,dt=.02,delta=.05,Tend=100.,save_field=True):
    h=L/(n+1); r=h*np.arange(1,n+1); k=np.pi*np.arange(1,n+1)/L
    q0=sol.sol(np.minimum(r,sol.x[-1]))[0]
    m=r>sol.x[-1]
    if np.any(m): q0[m]=sol.y[0,-1]*sol.x[-1]/r[m]*np.exp(-np.sqrt(w)*(r[m]-sol.x[-1]))
    base=(r*q0).astype(complex); target=4*np.pi*h*np.sum(abs(base)**2)
    v=base*(1+delta*np.exp(-(r/4)**2)); v*=np.sqrt(target/(4*np.pi*h*np.sum(abs(v)**2)))
    kin=np.exp(-1j*k*k*dt); every=round(1/dt); steps=round(Tend/dt)
    rec=[]; raw=[]
    def record(t):
        q=abs(v/r)**2; c=dst(v,type=1,norm='ortho')
        N=4*np.pi*h*np.sum(abs(v)**2)
        E=4*np.pi*h*(np.sum(k*k*abs(c)**2)+np.sum((np.log1p(q)-q)*r*r))
        ph=np.vdot(base,v); ph=ph/abs(ph); d=v-ph*base
        dc=dst(d,type=1,norm='ortho'); bc=dst(base,type=1,norm='ortho')
        l2=np.sqrt(np.sum(abs(d)**2)/np.sum(abs(base)**2))
        h1=np.sqrt((np.sum(abs(d)**2)+np.sum(k*k*abs(dc)**2))/
                   (np.sum(abs(base)**2)+np.sum(k*k*abs(bc)**2)))
        rms=np.sqrt(4*np.pi*h*np.sum(r*r*abs(v)**2)/N)
        core=4*np.pi*np.trapezoid(np.r_[0,abs(v[r<=15])**2],x=np.r_[0,r[r<=15]])/N
        rec.append([t,N,E,l2,h1,rms,core,float(q.max())])
        if save_field: raw.append(v.copy())
    record(0.)
    for j in range(1,steps+1):
        q=abs(v/r)**2; v*=np.exp(.5j*dt*q/(1+q))
        v=idst(kin*dst(v,type=1,norm='ortho'),type=1,norm='ortho')
        q=abs(v/r)**2; v*=np.exp(.5j*dt*q/(1+q))
        if j%every==0: record(j*dt)
    a=np.asarray(rec); cols=['t','N','H','phase_aligned_L2','phase_aligned_H1','rms_radius','core_fraction','peak']
    pd.DataFrame(a,columns=cols).to_csv(DATA/f'sat_evolution_{label}.csv',index=False)
    if save_field: np.savez_compressed(DATA/f'sat_evolution_{label}.npz',r=r,t=a[:,0],v=np.asarray(raw),base=base)
    meta=dict(label=label,omega=w,L=L,n=n,dr=h,dt=dt,delta=delta,T=Tend,
              max_N_drift=float(np.max(abs(a[:,1]/a[0,1]-1))),
              max_H_drift=float(np.max(abs((a[:,2]-a[0,2])/max(1,abs(a[0,2]))))),
              max_L2=float(np.max(a[:,3])),max_H1=float(np.max(a[:,4])),
              max_rms_change=float(np.max(abs(a[:,5]/a[0,5]-1))),
              min_core_fraction=float(np.min(a[:,6])))
    dump(DATA/f'sat_evolution_{label}.json',meta); return meta
def main():
    sols,df=branch(); convergence(sols)
    specs={str(w):radial_eigs(sols[w],w) for w in [.03,.1,.2]}
    dump(DATA/'saturating_radial_spectra.json',specs)
    metas=[]
    for w in [.03,.1]:
        for delta in [-.05,.05]:
            metas.append(evolution(sols[w],w,f'w{int(w*100):02d}_d{int((delta+0.05)*100):02d}',
                                   delta=delta))
    # convergence for the stable representative, +5% shape perturbation
    metas.append(evolution(sols[.1],.1,'w10_d05_timefine',dt=.01,delta=.05,save_field=False))
    metas.append(evolution(sols[.1],.1,'w10_d05_spacefine',n=8191,dt=.02,delta=.05,save_field=False))
    metas.append(evolution(sols[.1],.1,'w10_d05_domain480',L=480.,n=6143,dt=.02,delta=.05,save_field=False))
    pd.DataFrame(metas).to_csv(DATA/'saturating_evolution_runs.csv',index=False)
    i=int(df.M.idxmin()); turn=df.loc[i].to_dict()
    report=dict(version='exact-saturation-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                branch_points=len(df),sampled_turning_point=turn,
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                scope='3D radial exact saturating model; finite-time radial dynamics; not a full orbital-stability theorem')
    dump(QA/'summary.json',report)
    manifest={}
    for p in sorted(DATA.iterdir()):
        if p.is_file(): manifest[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    dump(QA/'data_manifest.json',manifest); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
