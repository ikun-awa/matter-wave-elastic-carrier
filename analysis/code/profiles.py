"""Direct scalar/coupled radial stationary profiles and independent residuals."""
from pathlib import Path
import numpy as np, json
from scipy.integrate import solve_bvp, simpson
BASE=Path(__file__).resolve().parents[1]; DATA=BASE/'data'
def scalar(w=.1,L=120.,tol=1e-10):
    r=np.linspace(0,L,2401); q=1.25*np.exp(-(r/4)**2)
    def f(r,y):return np.vstack((y[1],w*y[0]-y[0]**3/(1+y[0]**2)))
    def bc(a,b):return np.array([a[1],b[1]+(np.sqrt(w)+1/L)*b[0]])
    sol=solve_bvp(f,bc,r,np.vstack((q,-r*q/8)),S=np.diag([0.,-2.]),tol=tol,max_nodes=50000)
    if not sol.success or sol.y[0,0]<.1:raise RuntimeError(sol.message)
    return sol

def coupled(s, beta=.2,L=120.,tol=1e-9):
    r=np.linspace(0,L,2401); q,p=s.sol(r); e=q*q/(1+q*q); ep=2*q*p/(1+q*q)**2
    def f(r,y):
        q,p,e,z=y
        if np.any(e>=.98):pass  # Newton iterates may overshoot; final domain checked.
        return np.vstack((p,(.1-e)*q,z,(e/(1-e)-q*q)/beta))
    def bc(a,b):return np.array([a[1],a[3],b[1]+(np.sqrt(.1)+1/L)*b[0],b[2]])
    sol=solve_bvp(f,bc,r,np.vstack((q,p,e,ep)),S=np.diag([0.,-2.,0.,-2.]),tol=tol,max_nodes=50000)
    if not sol.success:raise RuntimeError(sol.message)
    rr=np.linspace(0,L,12001); yy=sol.sol(rr)
    if yy[0,0]<.1 or yy[2].max()>=1 or yy[2].min()<-1e-10:raise RuntimeError('Invalid stationary pair')
    return sol
def save_profile(sol,w,name,beta=0.):
    L=float(sol.x[-1]); r=np.linspace(0,L,round(L/.01)+1); a=sol.sol(r)
    q,qp=a[:2]; e=a[2] if len(a)==4 else q*q/(1+q*q)
    ep=a[3] if len(a)==4 else 2*q*qp/(1+q*q)**2
    integ=lambda x:float(4*np.pi*simpson(x*r*r,x=r))
    M=integ(q*q); T=integ(qp*qp); W=-np.log1p(-e)-e
    H=T+integ(W-e*q*q)+.5*beta*integ(ep*ep)
    z=sol.x[:-1]+.318309886*np.diff(sol.x); y=sol.sol(z); yp=sol.sol(z,1)
    ez=y[2] if len(y)==4 else y[0]**2/(1+y[0]**2)
    rq=yp[1]+2*y[1]/z-(w-ez)*y[0]
    re=0. if len(y)==2 else -beta*(yp[3]+2*y[3]/z)+y[2]/(1-y[2])-y[0]**2
    positive=bool(q.min()>-1e-10 and qp[q>1e-7].max()<1e-7)
    assert positive
    report={'name':name,'omega':w,'beta':beta,'L':L,'M':M,'H':H,'Q0':float(q[0]),'E0':float(e[0]),'q_residual_max':float(np.max(abs(rq))),'response_residual_max':float(np.max(abs(re))),'identity_wave':T+w*M-integ(e*q*q),'identity_scale':T+.5*beta*integ(ep*ep)+3*w*M+3*integ(W-e*q*q),'nodes':len(sol.x),'positive_monotone':positive,'status':int(sol.status)}
    np.savez_compressed(DATA/(name+'.npz'),r=r,Q=q,Qp=qp,E=e,Ep=ep)
    (DATA/(name+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
if __name__=='__main__':
    s=scalar(); out=[save_profile(s,.1,'scalar_w010')]
    for beta in [.05,.2]:
        c=coupled(s,beta); out.append(save_profile(c,.1,'coupled_b'+str(beta).replace('.',''),beta))
    (DATA/'stationary_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)
