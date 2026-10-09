"""Referee-style exact algebra checks; no new PDE simulation or stability certification."""
from pathlib import Path
import sympy as S, json, datetime, sys
B=Path(__file__).resolve().parent
s,a,R,r,lam,mu,Oc=S.symbols('s a R r lam mu Oc',positive=True)
e=S.symbols('e',real=True); om=S.symbols('om',positive=True)
q4,q6,T=S.symbols('q4 q6 T',positive=True)
w=-S.log(1-e)-e; eq=s/(1+s); h=e/(1-e)
checks={}
def zero(name,expr):
    v=S.simplify(S.expand(expr)); checks[name]={'pass':v==0,'remainder':str(v)}
    assert v==0,(name,v)
zero('elastic_restoring_derivative',S.diff(w,e)-h)
zero('elastic_tangent_stiffness',S.diff(w,e,2)-(1-e)**-2)
zero('stationary_response',h.subs(e,eq)-s)
zero('reduced_energy_derivative',S.diff(S.log(1+s)-s,s)+eq)
z=(1-e)*(1+s)
D=w-e*s-S.log(1+s)+s
zero('excess_energy_e_derivative',S.diff(D-(z-1-S.log(z)),e))
zero('excess_energy_s_derivative',S.diff(D-(z-1-S.log(z)),s))
zero('excess_energy_at_e0',D.subs(e,0)-(s-S.log(1+s)))
fs=-s/(1+s); fc=-s+s*s
zero('cq_coefficient_difference',fc-fs-s**3/(1+s))
zero('cq_relative_error',(fc-fs)/(-fs)-s*s)
FG=-s*s/2+s**3/3-S.log(1+s)+s
zero('cq_energy_difference_derivative',S.diff(FG,s)-s**3/(1+s))
rad=s*(s+3)/(1+s)**2
zero('jacobian_radial_derivative',S.diff(rad,s)-(3-s)/(1+s)**3)
zero('jacobian_peak',rad.subs(s,3)-S.Rational(9,8))
HE=lam*lam*T-lam**3*q4/2+lam**6*q6/3
Tstat=3*q4/4-q6
zero('cq_dilation_curvature',S.diff(HE,lam,2).subs({lam:1,T:Tstat})-(-3*q4/2+8*q6))
zero('cq_stationary_energy',HE.subs({lam:1,T:Tstat})-(q4/4-2*q6/3))
zero('cq_bounds_square1',S.Rational(3,16)**2-S.Rational(9,256))
zero('cq_bounds_square2',S.Rational(3,8)**2-S.Rational(9,64))
q=S.symbols('q',positive=True)
g=q**3/(1+q*q)-om*q
Kq=q*S.diff(g,q)/g
zero('shooting_K',Kq-(1+2*q*q/((1+q*q)*((1-om)*q*q-om))))
Ks=1+2*s/((1+s)*((1-om)*s-om))
zero('shooting_K_slope',S.diff(Ks,s)+2*((1-om)*s*s+om)/((1+s)**2*((1-om)*s-om)**2))
Lminus=om-q*q/(1+q*q); Lplus=om-q*q*(q*q+3)/(1+q*q)**2
zero('hessian_schur_beta0',Lminus-2*q*q/(1+q*q)**2-Lplus)
U=S.sqrt(a)*S.exp(-r*r/(2*R*R)); mass=4*S.pi*S.integrate(U*U*r*r,(r,0,S.oo)); kin=4*S.pi*S.integrate(S.diff(U,r)**2*r*r,(r,0,S.oo))
zero('gaussian_mass',mass-S.pi**S.Rational(3,2)*a*R**3)
zero('gaussian_kinetic',kin-S.Rational(3,2)*S.pi**S.Rational(3,2)*a*R)
R2=3*2**S.Rational(3,2)*(1+a)/a
Mbound=S.pi**S.Rational(3,2)*a*R2**S.Rational(3,2)
zero('gaussian_threshold_stationary_amplitude',S.diff(Mbound,a).subs(a,S.Rational(1,2)))
zero('gaussian_threshold_value',Mbound.subs(a,S.Rational(1,2))-27*2**S.Rational(5,4)*S.pi**S.Rational(3,2))
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.executable,'sympy':S.__version__,'checks':checks,'nchecks':len(checks),'all_pass':all(c['pass'] for c in checks.values()),'mass_sufficient_bound':str(S.N(Mbound.subs(a,S.Rational(1,2)),18)),'scope':'Exact local algebra only. Concentration compactness, theorem hypotheses, uniqueness and nonlinear orbital stability are not certified by these symbolic tests.'}
(B/'SYMBOLIC_REVIEW.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
