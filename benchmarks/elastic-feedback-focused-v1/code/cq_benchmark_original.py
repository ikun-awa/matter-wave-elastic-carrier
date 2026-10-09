"""Deterministic CQ benchmarks; no experimental electron data."""
from pathlib import Path
import sys, json, time, platform, hashlib
import numpy as np
import pandas as pd
import scipy, sympy as sp
from scipy.integrate import solve_bvp, simpson
from scipy.optimize import brentq
from scipy.fft import dst, idst
BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data'; QA=BASE/'qa'
for p in (DATA,QA): p.mkdir(exist_ok=True)
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def symbolic():
    x,R,N,D,g,b=sp.symbols('eta R N D g b',positive=True)
    exact=-x/(1+x); trunc=-x+x**2
    err=sp.factor((trunc-exact)/(-exact))
    assert sp.simplify(err-x**2)==0
    E=3*D*N/(2*R**2)-g*N**2/(2*(2*sp.pi)**sp.Rational(3,2)*R**3)+b*N**3/(3**sp.Rational(5,2)*sp.pi**3*R**6)
    Ef=sp.lambdify(R,E.subs({N:300,D:1,g:1,b:1}),'numpy')
    dEf=sp.lambdify(R,sp.diff(E,R).subs({N:300,D:1,g:1,b:1}),'numpy')
    ddEf=sp.lambdify(R,sp.diff(E,R,2).subs({N:300,D:1,g:1,b:1}),'numpy')
    roots=[brentq(dEf,3,6,xtol=1e-13),brentq(dEf,7,12,xtol=1e-13)]
    eta=np.linspace(0,2,2001); rad=np.linspace(2.6,20,3481)
    pd.DataFrame(dict(eta=eta,exact=-eta/(1+eta),cq=-eta+eta**2,relative_error=eta**2)).to_csv(DATA/'response.csv',index=False)
    pd.DataFrame(dict(R=rad,E=Ef(rad))).to_csv(DATA/'energy.csv',index=False)
    dump(DATA/'extrema.json',[dict(R=r,E=float(Ef(r)),curvature=float(ddEf(r)),derivative=float(dEf(r))) for r in roots])
    dump(QA/'symbolic.json',dict(relative_error=str(err),error_at_zero=str(sp.limit(err,x,0)),energy=sp.sstr(E),derivative=sp.sstr(sp.diff(E,R)),checks='exact identities passed'))
def bvp(w,L=120.,n=1201,tol=1e-8,guess=None):
    r=np.linspace(0,L,n)
    if guess is None:
        q=.8*np.exp(-(r/5)**2); y=np.vstack((q,-2*r*q/25))
    else: y=guess.sol(r)
    def f(r,y): return np.vstack((y[1],w*y[0]-y[0]**3+y[0]**5))
    def bc(a,b): return np.array([a[1],b[1]+(np.sqrt(w)+1/L)*b[0]])
    sol=solve_bvp(f,bc,r,y,S=np.array([[0.,0.],[0.,-2.]]),tol=tol,bc_tol=tol,max_nodes=60000)
    if not sol.success or sol.y[0,0]<.01: raise RuntimeError(f'BVP failure w={w}: {sol.message}; Q0={sol.y[0,0]}')
    return sol

def measures(sol,w,L,nseed,tol,tag):
    r=np.linspace(0,L,int(L/.01)+1); q,p=sol.sol(r)
    T=4*np.pi*simpson(p*p*r*r,x=r); N=4*np.pi*simpson(q*q*r*r,x=r)
    Q4=4*np.pi*simpson(q**4*r*r,x=r); Q6=4*np.pi*simpson(q**6*r*r,x=r)
    E=T-Q4/2+Q6/3
    p1=T+w*N-Q4+Q6; p2=T/2+1.5*w*N-.75*Q4+.5*Q6
    z=sol.x[:-1]+0.2718281828*np.diff(sol.x); qz,pz=sol.sol(z); ppz=sol.sol(z,1)[1]
    resid=ppz+2*pz/z-w*qz+qz**3-qz**5
    scale=1+np.abs(ppz)+np.abs(2*pz/z)+w*np.abs(qz)+np.abs(qz)**3+np.abs(qz)**5
    row=dict(tag=tag,omega=w,L=L,seed_nodes=nseed,nodes=len(sol.x),tolerance=tol,status=int(sol.status),N=N,E=E,Q0=q[0],Q_L=q[-1],boundary=abs(p[-1]+(np.sqrt(w)+1/L)*q[-1]),collocation_max=float(max(sol.rms_residuals)),point_residual=float(max(abs(resid)/scale)),identity1=p1,identity2=p2,relative_identity1=abs(p1)/(T+w*N+Q4+Q6),relative_identity2=abs(p2)/(T/2+1.5*w*N+.75*Q4+.5*Q6),min_Q=float(min(q)),positive_core=bool(np.min(q)>=-1e-12 and q[0]>.01),monotone_core=bool(np.max(p[q>1e-8])<1e-7))
    if min(q)<-1e-9 or not row['monotone_core']: raise RuntimeError('Non-ground-state candidate')
    np.savez_compressed(DATA/f'bvp_{tag}.npz',r=r,Q=q,Qp=p,mesh=sol.x,mesh_y=sol.y,collocation_residuals=sol.rms_residuals)
    return row

def stationary():
    sols={.05:bvp(.05)}
    for seq in [np.arange(.045,.009,-.005),np.arange(.055,.1451,.005)]:
        last=sols[.05]
        for w in seq:
            w=round(float(w),5); last=bvp(w,guess=last); sols[w]=last
    for w in np.arange(.0175,.0401,.0025):
        w=round(float(w),5)
        if w not in sols: sols[w]=bvp(w,guess=sols[min(sols,key=lambda t:abs(t-w))])
    rows=[]
    for w,sol in sorted(sols.items()):
        rows.append(measures(sol,w,120.,1201,1e-8,f'branch_{w:.5f}'))
    pd.DataFrame(rows).to_csv(DATA/'branch.csv',index=False)
    print('BVP BRANCH',len(rows),'solutions; max relative identity',max(max(a['relative_identity1'],a['relative_identity2']) for a in rows),flush=True)
    conv=[]
    for w in [.025,.1,.145]:
        for L,n,tol,name in [(120.,601,1e-6,'coarse'),(120.,1201,1e-8,'medium'),(120.,2401,1e-10,'fine'),(80.,1601,1e-10,'domain80'),(160.,3201,1e-10,'domain160')]:
            sol=bvp(w,L,n,tol,sols[w]); conv.append(measures(sol,w,L,n,tol,f'{name}_{w:.3f}'))
        ref=conv[-1]
        for row in conv[-5:]:
            row['rel_N_to_ref']=abs(row['N']-ref['N'])/ref['N']; row['rel_E_to_ref']=abs(row['E']-ref['E'])/max(1,abs(ref['E']))
    pd.DataFrame(conv).to_csv(DATA/'bvp_convergence.csv',index=False)
    sol=bvp(.1,160.,3201,1e-10,sols[.1]); print('BVP CONVERGENCE DONE',flush=True)
    return sol

def evolution(sol,label,L,n,dt,delta,Tend=200.):
    """Radial v=r*psi, sine spectral kinetic operator, Strang splitting."""
    h=L/(n+1); r=h*np.arange(1,n+1); k=np.pi*np.arange(1,n+1)/L
    qinit=sol.sol(np.minimum(r,sol.x[-1]))[0]; outer_init=r>sol.x[-1]; qinit[outer_init]=sol.y[0,-1]*sol.x[-1]/r[outer_init]*np.exp(-np.sqrt(.1)*(r[outer_init]-sol.x[-1])); v=r*qinit; Ntarget=4*np.pi*h*np.sum(abs(v)**2)
    v=v*(1+delta*np.exp(-(r/4)**2)); v=v*np.sqrt(Ntarget/(4*np.pi*h*np.sum(abs(v)**2)))
    v=v.astype(complex); v_initial=v.copy(); kin=np.exp(-1j*k*k*dt)
    steps=round(Tend/dt); every=round(1./dt)
    assert abs(steps*dt-Tend)<1e-10 and abs(every*dt-1.)<1e-10
    times=[]; records=[]; raw=[]
    def record(t):
        q=abs(v/r)**2; coeff=dst(v,type=1,norm='ortho')
        norm=4*np.pi*h*np.sum(abs(v)**2)
        energy=4*np.pi*h*(np.sum(k*k*abs(coeff)**2)+np.sum((-q/2+q*q/3)*abs(v)**2))
        width=np.sqrt(4*np.pi*h*np.sum(r*r*abs(v)**2)/norm)
        core=4*np.pi*np.trapezoid(np.r_[0,abs(v[r<=15])**2],x=np.r_[0,r[r<=15]])/norm
        outer=4*np.pi*h*np.sum(abs(v[r>=.8*L])**2)/norm
        times.append(t); records.append([norm,energy,width,core,float(q.max()),outer])
        raw.append(v.copy())
    record(0.)
    for j in range(1,steps+1):
        q=abs(v/r)**2; v*=np.exp(.5j*dt*(q-q*q))
        v=idst(kin*dst(v,type=1,norm='ortho'),type=1,norm='ortho')
        q=abs(v/r)**2; v*=np.exp(.5j*dt*(q-q*q))
        if j%every==0: record(j*dt)
    arr=np.asarray(records); raw=np.asarray(raw)
    if not np.all(np.isfinite(raw)): raise RuntimeError('Non-finite evolution output')
    dn=(arr[:,0]-arr[0,0])/arr[0,0]; de=(arr[:,1]-arr[0,1])/max(1,abs(arr[0,1]))
    np.savez_compressed(DATA/f'evolution_{label}.npz',r=r,t=times,v=raw,v_initial=v_initial,diagnostics=arr)
    pd.DataFrame(dict(t=times,N=arr[:,0],E=arr[:,1],rms_radius=arr[:,2],core_fraction=arr[:,3],peak=arr[:,4],outer_fraction=arr[:,5],relative_N_drift=dn,relative_E_drift=de)).to_csv(DATA/f'evolution_{label}.csv',index=False)
    meta=dict(label=label,L=L,n=n,dr=h,dt=dt,delta=delta,T=Tend,steps=steps,saved_frames=len(times),finite=True,N_initial=arr[0,0],E_initial=arr[0,1],max_N_drift=float(max(abs(dn))),max_E_drift=float(max(abs(de))),max_width_change=float(max(abs(arr[:,2]/arr[0,2]-1))),min_core_fraction=float(min(arr[:,3])),max_outer_fraction=float(max(arr[:,5])))
    dump(DATA/f'evolution_{label}.json',meta)
    print('EVOLUTION',label,'dN',meta['max_N_drift'],'dE',meta['max_E_drift'],flush=True)
    return meta

def compare(label,reference):
    a=np.load(DATA/f'evolution_{label}.npz'); b=np.load(DATA/f'evolution_{reference}.npz')
    ra=a['r']; rb=b['r']; va=a['v'][-1]; vb=b['v'][-1]
    assert np.all(np.diff(rb)>0)
    from scipy.interpolate import CubicSpline
    vb=CubicSpline(np.r_[0,rb,b['r'][-1]+b['r'][0]],np.r_[0j,vb,0j])(ra)
    overlap=np.vdot(vb,va); vb*=np.exp(1j*np.angle(overlap))
    state=np.linalg.norm(va-vb)/np.linalg.norm(vb)
    density=np.linalg.norm(abs(va)**2-abs(vb)**2)/np.linalg.norm(abs(vb)**2)
    return dict(run=label,reference=reference,final_phase_aligned_L2=float(state),final_radial_density_L2=float(density),max_rms_difference=float(max(abs(a['diagnostics'][:,2]-b['diagnostics'][:,2]))))

def main():
    symbolic(); sol=stationary()
    configs=[('control',640.,8191,.01,0.),('perturb01',640.,8191,.01,.01),('perturb03',640.,8191,.01,.03),('perturb05',640.,8191,.01,.05),('time_coarse',640.,8191,.04,.05),('time_medium',640.,8191,.02,.05),('time_fine',640.,8191,.005,.05),('space_coarse',640.,4095,.01,.05),('space_fine',640.,16383,.01,.05),('domain960',960.,12287,.01,.05)]
    metas=[]
    for cfg in configs: metas.append(evolution(sol,*cfg))
    pd.DataFrame(metas).to_csv(DATA/'evolution_runs.csv',index=False)
    pairs=[('time_coarse','time_fine'),('time_medium','time_fine'),('perturb05','time_fine'),('space_coarse','space_fine'),('perturb05','space_fine'),('perturb05','domain960')]
    comp=[compare(a,b) for a,b in pairs]
    pd.DataFrame(comp).to_csv(DATA/'evolution_convergence.csv',index=False)
    dump(QA/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sympy=sp.__version__,platform=platform.platform()))
    dump(DATA/'model_parameters.json',dict(model='i u_t = -Delta u - |u|^2 u + |u|^4 u',stationary_omega=.1,gaussian_N=300,BVP_origin='singular matrix S=diag(0,-2); Qprime(0)=0',BVP_outer='Qprime+(sqrt(omega)+1/L)Q=0',time_boundary='v=r*u=0 at r=0,L',time_solver='sine-pseudospectral Strang splitting',perturbation='Q(r)*(1+delta*exp(-(r/4)^2)); rescale to unperturbed norm',core_radius=15.,conservation='spectral kinetic energy consistent with the discretized Laplacian',claims='radial finite-time model benchmark only; not full 3D stability or electron observations'))
    print('ALL COMPUTATIONS FINISHED',flush=True)
if __name__=='__main__': main()
