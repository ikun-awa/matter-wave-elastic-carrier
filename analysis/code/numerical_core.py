"""Fourier splitting for a scalar saturating wave and its inertial response.
No filtering, damping, clipping, mass renormalization or radial projection in time.
"""
import numpy as np
from scipy.fft import fftn, ifftn, rfftn, irfftn
class Grid:
    def __init__(self,n,L,workers=4):
        if n<8 or L<=0:raise ValueError('Invalid grid')
        self.n,self.L,self.h,self.workers=n,float(L),float(L)/n,workers
        self.shape=(n,n,n); a=(np.arange(n)-n//2)*self.h
        self.xyz=np.meshgrid(a,a,a,indexing='ij',sparse=True)
        k=2*np.pi*np.fft.fftfreq(n,d=self.h)
        self.k2=k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2
        self.kr2=self.k2[:,:,:n//2+1]; self.dv=self.h**3
        self._free_cache={}
    def ft(self,u):return fftn(u,workers=self.workers)
    def inv(self,u):return ifftn(u,workers=self.workers)
    def lap(self,e):
        return irfftn(-self.kr2*rfftn(e,workers=self.workers),s=self.shape,workers=self.workers)
    def free(self,u,dt):
        if dt not in self._free_cache:self._free_cache[dt]=np.exp(-1j*dt*self.k2)
        return self.inv(self._free_cache[dt]*self.ft(u))
    def norm(self,u):return self.dv*float(np.sum(abs(u)**2))
    def kinetic(self,u):return self.dv/np.prod(self.shape)*float(np.sum(self.k2*abs(self.ft(u))**2))
def response_force(g,u,e,alpha,beta):
    if alpha<=0 or beta<0 or not np.isfinite(e).all() or np.max(e)>=1:
        raise ValueError('Invalid coefficients or singular response e >= 1')
    return (beta*g.lap(e)-e/(1-e)+abs(u)**2)/alpha
def scalar_step(g,u,dt):
    s=abs(u)**2; u=u*np.exp(.5j*dt*s/(1+s))
    u=g.free(u,dt); s=abs(u)**2
    return u*np.exp(.5j*dt*s/(1+s))
def coupled_step(g,u,e,p,dt,alpha,beta,force=None,return_force=False):
    # Potential half flow: e fixed, |u| fixed, so phase and kick are exact.
    a=response_force(g,u,e,alpha,beta) if force is None else force
    phalf=p+.5*dt*a
    u=g.free(u*np.exp(.5j*dt*e),dt)
    enew=e+dt*phalf
    anew=response_force(g,u,enew,alpha,beta)
    unew=u*np.exp(.5j*dt*enew); pnew=phalf+.5*dt*anew
    if return_force:return unew,enew,pnew,anew
    return unew,enew,pnew
def energy(g,u,e=None,p=None,alpha=1.,beta=0.):
    s=abs(u)**2; T=g.kinetic(u)
    if e is None:return T+g.dv*np.sum(np.log1p(s)-s)
    if np.max(e)>=1:raise ValueError('Singular energy')
    W=-np.log1p(-e)-e
    return T+g.dv*np.sum(.5*alpha*p*p+W-e*s)+.5*beta*g.kinetic(e)
def phase_distance(g,u,q):
    # Separate exact phase minimizations in L2 and H1; no translation fit.
    f=g.ft(u); fq=g.ft(q); den=np.prod(g.shape)
    l2inner=np.vdot(q,u); h1inner=np.sum((1+g.k2)*np.conj(fq)*f)/den
    ph2=l2inner/abs(l2inner) if abs(l2inner)>0 else 1.
    ph1=h1inner/abs(h1inner) if abs(h1inner)>0 else 1.
    d2=np.linalg.norm((u-ph2*q).ravel())/np.linalg.norm(q.ravel())
    d1=np.sqrt(np.sum((1+g.k2)*abs(f-ph1*fq)**2)/np.sum((1+g.k2)*abs(fq)**2))
    return float(d2),float(d1)
