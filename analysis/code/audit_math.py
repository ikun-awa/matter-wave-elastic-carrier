"""Exact symbolic identities and non-certified floating-point spectral checks."""
from pathlib import Path
import json,numpy as np,pandas as pd,sympy as sp
from scipy.integrate import solve_bvp,simpson
from scipy.linalg import eigh_tridiagonal
from profiles import scalar,save_profile
B=Path(__file__).resolve().parents[1]; D=B/'data'; A={}
x,w,s,k=sp.symbols('x w s k',positive=True)
g=x**3/(1+x*x)-w*x; K=sp.factor(x*sp.diff(g,x)/g)
K_s=1+2*s/((1+s)*((1-w)*s-w))
checks={'K_formula':K-K_s.subs(s,x*x),'K_derivative':sp.diff(K_s,s)+2*((1-w)*s*s+w)/((1+s)**2*((1-w)*s-w)**2),'qgprime_minus_g':x*sp.diff(g,x)-g-2*x**3/(1+x*x)**2,'response_derivative':sp.diff(s/(1+s),s)-1/(1+s)**2,'CQ_defect':-s+s*s+s/(1+s)-s**3/(1+s),'gaussian_lower_bound_derivative':sp.diff(s-sp.log(1+s)-s*s/(2*(1+s)),s)-s*s/(2*(1+s)**2)}
for key,z in checks.items():
    assert sp.simplify(z)==0,key
    A[key]='exact zero'
A['sufficient_negative_energy_mass']=float(27*2**sp.Rational(5,4)*sp.pi**sp.Rational(3,2))
A['warning']='Symbolic checks do not replace operator-domain assumptions or the cited shooting theorem.'
(B/'qa'/'symbolic_checks.json').write_text(json.dumps(A,indent=2),encoding='utf-8')
rows=[]; specs=[]
for omega in [.03,.1,.3]:
    for L in [80.,120.]:
        sol=scalar(omega,L); save_profile(sol,omega,f'check_w{omega:.3f}_L{L:.0f}')
        r=np.linspace(0,L,2401)
        def f(r,y):
            q=sol.sol(r)[0]; s=q*q; V=omega-s*(s+3)/(1+s)**2
            return np.vstack((y[1],V*y[0]+q))
        def bc(a,b):return np.array([a[1],b[1]+(np.sqrt(omega)+1/L)*b[0]+sol.sol(L)[0]/(2*np.sqrt(omega))])
        pp=solve_bvp(f,bc,r,np.zeros((2,len(r))),S=np.diag([0.,-2.]),tol=1e-10,max_nodes=50000)
        assert pp.success
        rr=np.linspace(0,L,round(L/.01)+1); q=sol.sol(rr)[0]; P=pp.sol(rr)[0]
        slope=8*np.pi*simpson(q*P*rr*rr,x=rr)
        rows.append(dict(omega=omega,L=L,method='linearized_BVP',step=0.,dM_domega=slope))
        for dw in [1e-3,5e-4,2.5e-4]:
            masses=[]
            for ww in [omega-dw,omega+dw]:
                def fw(r,y):return np.vstack((y[1],ww*y[0]-y[0]**3/(1+y[0]**2)))
                def bw(a,b):return np.array([a[1],b[1]+(np.sqrt(ww)+1/L)*b[0]])
                sf=solve_bvp(fw,bw,r,sol.sol(r),S=np.diag([0.,-2.]),tol=1e-10,max_nodes=50000)
                assert sf.success
                masses.append(4*np.pi*simpson(sf.sol(rr)[0]**2*rr*rr,x=rr))
            rows.append(dict(omega=omega,L=L,method='centered_mass_difference',step=dw,dM_domega=(masses[1]-masses[0])/(2*dw)))
        for h in [.08,.04,.02]:
            rad=np.arange(h,L,h); qs=sol.sol(rad)[0]**2
            for sign in ['minus','plus']:
                v=omega-qs/(1+qs) if sign=='minus' else omega-qs*(qs+3)/(1+qs)**2
                for ell in [0,1,2]:
                    vals=eigh_tridiagonal(2/h**2+v+ell*(ell+1)/rad**2,-np.ones(len(rad)-1)/h**2,select='i',select_range=(0,2),eigvals_only=True)
                    for j,value in enumerate(vals):specs.append(dict(omega=omega,L=L,h=h,operator=sign,ell=ell,index=j,value=float(value)))
        pd.DataFrame(rows).to_csv(D/'slope_checks.csv',index=False)
        pd.DataFrame(specs).to_csv(D/'spectral_checks.csv',index=False)
        print('SPECTRAL_AND_SLOPE',omega,L,float(slope),flush=True)
print('EXACT_IDENTITIES_AND_SPECTRAL_REFINEMENTS_COMPLETE',flush=True)
