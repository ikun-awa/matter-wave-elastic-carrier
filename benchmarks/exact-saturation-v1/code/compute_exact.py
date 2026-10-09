"""Exact-saturation stationary branch, spectra, and radial dynamics.
Model: i u_t = -Delta u - |u|^2/(1+|u|^2) u.
Deterministic numerical study; not experimental data and not a proof of orbital stability.
"""
from pathlib import Path
import json, platform, hashlib, time
import numpy as np
import pandas as pd
import scipy
from scipy.integrate import solve_bvp, simpson
from scipy.linalg import eigh_tridiagonal
from scipy.fft import dst, idst

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data'; QA=BASE/'qa'
DATA.mkdir(exist_ok=True); QA.mkdir(exist_ok=True)

def dump(p,x): p.write_text(json.dumps(x,indent=2),encoding='utf-8')

def bvp(w,L=120.0,n=1201,tol=1e-8,guess=None,A=1.0,R=5.0):
    r=np.linspace(0,L,n)
    if guess is None:
        q=A*np.exp(-(r/R)**2); y=np.vstack((q,-2*r*q/R**2))
    else:
        y=guess.sol(r)
    def f(r,y):
        q=y[0]
        return np.vstack((y[1],w*q-q**3/(1+q*q)))
    def bc(a,b):
        return np.array([a[1],b[1]+(np.sqrt(w)+1/L)*b[0]])
    sol=solve_bvp(f,bc,r,y,S=np.array([[0.,0.],[0.,-2.]]),
                  tol=tol,bc_tol=tol,max_nodes=80000)
    if not sol.success or sol.y[0,0] <= 0.02:
        raise RuntimeError(f'BVP failure w={w}: {sol.message}; Q0={sol.y[0,0]}')
    return sol

def measures(sol,w,L,nseed,tol,tag,dr=.01,save=True):
    r=np.arange(0,L+dr/2,dr); q,p=sol.sol(r); s=q*q
    def integ(y): return float(4*np.pi*simpson(y*r*r,x=r))
    T=integ(p*p); M=integ(s); A1=integ(s/(1+s)); B1=integ(np.log1p(s))
    Fint=B1-M; H=T+Fint
    p1=T+(w-1)*M+A1
    p2=T+3*(w-1)*M+3*B1
    z=sol.x[:-1]+0.2718281828*np.diff(sol.x)
    qz,pz=sol.sol(z); ppz=sol.sol(z,1)[1]
    resid=ppz+2*pz/z-w*qz+qz**3/(1+qz*qz)
    scale=1+np.abs(ppz)+np.abs(2*pz/z)+w*np.abs(qz)+np.abs(qz**3/(1+qz*qz))
    row=dict(tag=tag,omega=w,L=L,seed_nodes=nseed,nodes=len(sol.x),
             tolerance=tol,status=int(sol.status),Q0=float(q[0]),M=M,H=H,T=T,
             boundary=float(abs(p[-1]+(np.sqrt(w)+1/L)*q[-1])),
             collocation_max=float(max(sol.rms_residuals)),
             point_residual=float(max(abs(resid)/scale)),
             identity1=p1,identity2=p2,
             rel_identity1=abs(p1)/(T+abs((w-1)*M)+A1),
             rel_identity2=abs(p2)/(T+3*abs((w-1)*M)+3*B1))
    if save:
        np.savez_compressed(DATA/f'bvp_{tag}.npz',r=r,Q=q,Qp=p,mesh=sol.x,mesh_y=sol.y)
    return row

def stationary_branch():
    freqs=sorted(set(
        [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045] +
        [round(x,4) for x in np.arange(.046,.0551,.0005)] +
        [0.0575,0.06,0.065,0.07,0.08,0.1,0.12,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
    seed=bvp(.1)
    sols={.1:seed}
    last=seed
    for w in sorted([x for x in freqs if x<.1],reverse=True):
        L=160. if w<.03 else 120.; n=1601 if L>120 else 1201
        last=bvp(w,L=L,n=n,guess=last); sols[w]=last
    last=seed
    for w in sorted([x for x in freqs if x>.1]):
        last=bvp(w,guess=last); sols[w]=last
    rows=[]
    for w in freqs:
        sol=sols[w]; L=float(sol.x[-1])
        rows.append(measures(sol,w,L,1601 if L>120 else 1201,1e-8,f'branch_{w:.4f}'))
    df=pd.DataFrame(rows).sort_values('omega')
    df.to_csv(DATA/'branch.csv',index=False)
    sampled=df.loc[df.M.idxmin()]
    dump(DATA/'branch_summary.json',dict(points=len(df),sampled_min_omega=float(sampled.omega),
         sampled_min_mass=float(sampled.M),sampled_min_Q0=float(sampled.Q0)))
    return sols,df

def bvp_convergence(sols):
    rows=[]
    for w in [.03,.1,.5]:
        guess=sols[w]
        configs=[(120.,601,1e-6,'coarse'),(120.,1201,1e-8,'medium'),
                 (120.,2401,1e-10,'fine'),(80.,1601,1e-10,'domain80'),
                 (160.,3201,1e-10,'domain160')]
        block=[]
        for L,n,tol,name in configs:
            sol=bvp(w,L,n,tol,guess)
            block.append(measures(sol,w,L,n,tol,f'{name}_{w:.3f}'))
        ref=block[-1]
        for row in block:
            row['rel_M_to_ref']=abs(row['M']-ref['M'])/ref['M']
            row['rel_H_to_ref']=abs(row['H']-ref['H'])/max(1,abs(ref['H']))
        rows.extend(block)
    pd.DataFrame(rows).to_csv(DATA/'bvp_convergence.csv',index=False)

def spectrum(sol,w,L=80.,dr=.025,nev=5):
    out={}
    for ell in [0,1,2]:
        r=np.arange(dr,L,dr); q=sol.sol(r)[0]; s=q*q
        for name in ['minus','plus']:
            V=w-s/(1+s) if name=='minus' else w-s*(s+3)/(1+s)**2
            V=V+ell*(ell+1)/r**2
            d=2/dr**2+V
            e=-np.ones(len(r)-1)/dr**2
            vals=eigh_tridiagonal(d,e,select='i',select_range=(0,nev-1),
                                  check_finite=False)[0]
            out[f'L{name}_l{ell}']=vals.tolist()
    return out

def spectral_branch(sols,df):
    records=[]
    for row in df.itertuples():
        sp=spectrum(sols[float(row.omega)],float(row.omega))
        rec={'omega':float(row.omega),'M':float(row.M),'Q0':float(row.Q0)}
        for k,v in sp.items():
            for j,val in enumerate(v):
                rec[f'{k}_eig{j}']=val
        records.append(rec)
    pd.DataFrame(records).to_csv(DATA/'spectral_branch.csv',index=False)
    conv=[]
    for w in [.03,.1]:
        sol=sols[w]
        for L,dr in [(60.,.05),(80.,.05),(80.,.025),(120.,.025),(80.,.0125)]:
            sp=spectrum(sol,w,L=L,dr=dr,nev=3)
            rec={'omega':w,'L':L,'dr':dr}
            for k,v in sp.items():
                for j,val in enumerate(v):
                    rec[f'{k}_eig{j}']=val
            conv.append(rec)
    pd.DataFrame(conv).to_csv(DATA/'spectral_convergence.csv',index=False)
def radial_evolution(sol,w,label,L=320.,n=4095,dt=.02,delta=.05,Tend=100.,save_field=True):
    h=L/(n+1)
    r=h*np.arange(1,n+1)
    k=np.pi*np.arange(1,n+1)/L
    q0=sol.sol(np.minimum(r,sol.x[-1]))[0]
    mask=r>sol.x[-1]
    if np.any(mask):
        q0[mask]=sol.y[0,-1]*sol.x[-1]/r[mask]*np.exp(-np.sqrt(w)*(r[mask]-sol.x[-1]))
    base=(r*q0).astype(complex)
    N0=4*np.pi*h*np.sum(abs(base)**2)
    v=base*(1+delta*np.exp(-(r/4)**2))
    v*=np.sqrt(N0/(4*np.pi*h*np.sum(abs(v)**2)))
    v0=v.copy()
    kin=np.exp(-1j*k*k*dt)
    steps=round(Tend/dt)
    every=max(1,round(1/dt))
    records=[]
    frames=[]
    times=[]
    def diag(t):
        q=abs(v/r)**2
        coeff=dst(v,type=1,norm='ortho')
        norm=4*np.pi*h*np.sum(abs(v)**2)
        energy=4*np.pi*h*(np.sum(k*k*abs(coeff)**2)+np.sum((np.log1p(q)-q)*r*r))
        overlap=np.vdot(base,v)
        phase=overlap/abs(overlap)
        diff=v-phase*base
        dc=dst(diff,type=1,norm='ortho')
        bc=dst(base,type=1,norm='ortho')
        l2=np.sqrt(np.sum(abs(diff)**2)/np.sum(abs(base)**2))
        h1=np.sqrt((np.sum(abs(diff)**2)+np.sum(k*k*abs(dc)**2))/
                   (np.sum(abs(base)**2)+np.sum(k*k*abs(bc)**2)))
        rms=np.sqrt(4*np.pi*h*np.sum(r*r*abs(v)**2)/norm)
        records.append([norm,energy,l2,h1,rms,float(q.max())])
        times.append(t)
        if save_field:
            frames.append(v.copy())
    diag(0.)
    for j in range(1,steps+1):
        q=abs(v/r)**2
        v*=np.exp(.5j*dt*q/(1+q))
        v=idst(kin*dst(v,type=1,norm='ortho'),type=1,norm='ortho')
        q=abs(v/r)**2
        v*=np.exp(.5j*dt*q/(1+q))
        if j%every==0:
            diag(j*dt)
    a=np.asarray(records)
    dn=(a[:,0]-a[0,0])/a[0,0]
    de=(a[:,1]-a[0,1])/max(1,abs(a[0,1]))
    pd.DataFrame(dict(t=times,N=a[:,0],H=a[:,1],phase_aligned_L2=a[:,2],
             phase_aligned_H1=a[:,3],rms_radius=a[:,4],peak=a[:,5],
             relative_N_drift=dn,relative_H_drift=de)).to_csv(DATA/f'evolution_{label}.csv',index=False)
    if save_field:
        np.savez_compressed(DATA/f'evolution_{label}.npz',
                            r=r,t=np.asarray(times),v=np.asarray(frames),v_initial=v0)
    else:
        np.savez_compressed(DATA/f'evolution_{label}_final.npz',
                            r=r,v_final=v,v_initial=v0)
    meta=dict(label=label,omega=w,L=L,n=n,dr=h,dt=dt,delta=delta,T=Tend,
              max_L2=float(max(a[:,2])),
              max_H1=float(max(a[:,3])),
              final_L2=float(a[-1,2]),
              final_H1=float(a[-1,3]),
              max_abs_rms_change=float(max(abs(a[:,4]/a[0,4]-1))),
              final_rms_change=float(a[-1,4]/a[0,4]-1),
              max_N_drift=float(max(abs(dn))),
              max_H_drift=float(max(abs(de))))
    dump(DATA/f'evolution_{label}.json',meta)
    return meta

def run_evolutions(sols):
    metas=[]
    for w,name in [(.03,'unstable'),(.1,'stable')]:
        for delta,suffix in [(0.,'control'),(.05,'plus05'),(-.05,'minus05')]:
            metas.append(radial_evolution(sols[w],w,f'{name}_{suffix}',
                                          delta=delta,save_field=True))
    for w,name in [(.03,'unstable'),(.1,'stable')]:
        for L,n,dt,suffix in [
            (320.,4095,.04,'time_coarse'),
            (320.,4095,.01,'time_fine'),
            (320.,2047,.02,'space_coarse'),
            (320.,8191,.02,'space_fine'),
            (480.,6143,.02,'domain480')]:
            metas.append(radial_evolution(
                sols[w],w,f'{name}_{suffix}',
                L=L,n=n,dt=dt,delta=.05,save_field=False))
    pd.DataFrame(metas).to_csv(DATA/'evolution_runs.csv',index=False)

def main():
    t0=time.time()
    sols,df=stationary_branch()
    bvp_convergence(sols)
    spectral_branch(sols,df)
    run_evolutions(sols)
    env=dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        model='i u_t=-Delta u-|u|^2/(1+|u|^2)u',
        claim_boundary='numerical exact-saturation branch/spectrum/radial finite-time dynamics; not experimental data or rigorous orbital-stability proof',
        runtime_seconds=time.time()-t0)
    dump(QA/'environment.json',env)
    manifest={}
    for p in sorted(DATA.iterdir()):
        if p.is_file():
            manifest[p.name]={
                'bytes':p.stat().st_size,
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    dump(QA/'data_manifest.json',manifest)
    print('EXACT_SATURATION_COMPLETE',json.dumps(env),flush=True)

if __name__=='__main__':
    main()
