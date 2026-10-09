"""Journal-width figures from executed calculations; nature-figure Python route."""
from pathlib import Path
import sys,json,subprocess
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import NullLocator
B=Path(__file__).resolve().parents[1]; D=B/'data'; F=B/'figures'; Q=B/'qa'
import os
S=Path(os.environ['NATURE_FIGURE_ROOT']) if os.environ.get('NATURE_FIGURE_ROOT') else None
if S:
    sys.path.insert(0,str(S/'scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
F=B.parent/'figures'
F.mkdir(exist_ok=True);Q.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.3,'legend.frameon':False})
BLUE='#165B96'; GREY='#777777'; TEAL='#358F88'; VIOLET='#955482'; BLACK='#222222'
def layout(rows=1):
    fig=plt.figure(figsize=(150/25.4,(83 if rows==1 else 153)/25.4)); axes=[]
    for y in ([.23] if rows==1 else [.61,.13]):
        for x in [.13,.64]:axes.append(fig.add_axes([x,y,.325,.57 if rows==1 else .275]))
    return fig,axes

def panel(a,label,title):
    a.text(0,1,label,transform=a.transAxes+ScaledTranslation(-18/72,6/72,a.figure.dpi_scale_trans),fontsize=9,fontweight='bold',va='bottom')
    a.set_title(title,loc='left',pad=11); a.tick_params(direction='out',length=3,pad=3)

def save(fig,axs,name):
    ids=list('abcd'[:len(axs)])
    if S:require_matplotlib_panel_alignment(fig,axes=axs,panel_ids=ids,row_groups=[ids[i:i+2] for i in range(0,len(ids),2)],column_groups=[ids[i::2] for i in range(2)] if len(ids)>2 else None,json_out=str(Q/(name+'.alignment.json')),overlay_svg=str(Q/(name+'.alignment.svg')),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(F/(name+'.pdf'),dpi=600); fig.savefig(F/(name+'.svg'),dpi=600)
    fig.savefig(F/(name+'.png'),dpi=600); fig.savefig(F/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    if S:
      for kind,args in [('font',[str(S/'scripts'/'audit_pdf_text.py'),str(F/(name+'.pdf')),'--min-pt','5','--json']),('collision',[str(S/'scripts'/'audit_figure_collisions.py'),str(F/(name+'.pdf')),'--json-out',str(Q/(name+'.collision.json')),'--strict'])]:
          p=subprocess.run([sys.executable,*args],capture_output=True,text=True,encoding='utf-8',errors='replace'); (Q/(name+'.'+kind+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8')
          if p.returncode:raise RuntimeError(name+' '+kind+' failed')
    print('FIGURE_VERIFIED',name,flush=True)
def figure2():
    s=np.linspace(0,1.3,651); fig,ax=layout()
    panel(ax[0],'a','Response coefficients')
    ax[0].plot(s,-s/(1+s),color=BLUE,label='Saturating')
    ax[0].plot(s,-s+s*s,color=GREY,ls='--',label='CQ')
    ax[0].set(xlabel='Intensity s',ylabel='Coefficient f(s)',xlim=(0,1.3),ylim=(-.65,.55)); ax[0].legend(loc='upper left')
    diag=pd.read_csv(D/'legacy_cq_diagnostics.csv'); d=diag[diag.kind=='branch']; assert len(d)==33
    panel(ax[1],'b','Peak mismatch on CQ states')
    ax[1].plot(d.omega,100*d.response_error_peak,color=BLUE,marker='o',ms=2.5)
    ax[1].axhline(1,color=GREY,ls='--',lw=.8)
    ax[1].set(xlabel='CQ frequency ω',ylabel='Peak coefficient error (%)',xlim=(0,.15),ylim=(0,85))
    save(fig,ax,'figure_2')
def figure3():
    d=pd.read_csv(D/'slope_checks.csv'); p=d[(d.L==120)&(d.method=='linearized_BVP')]
    fig,ax=layout(); panel(ax[0],'a','Branch-specific slope evidence')
    ax[0].axhline(0,color=GREY,ls='--',lw=.8)
    ax[0].scatter(p.omega,p.dM_domega,s=20,color=BLUE)
    ax[0].set(xlabel='Frequency ω',ylabel='Norm derivative M′(ω)',xlim=(0,.33),ylim=(-1100,2400))
    panel(ax[1],'b','Expected symmetry zero modes')
    e=pd.read_csv(D/'spectral_checks.csv')
    for op,ell,c,ls,lbl in [('minus',0,BLUE,'-','Phase'),('plus',1,GREY,'--','Translation')]:
        q=e[(e.omega==.1)&(e.L==120)&(e['index']==0)&(e.operator==op)&(e.ell==ell)].sort_values('h')
        assert (abs(q.value)>0).all(); ax[1].loglog(q.h,abs(q.value),color=c,ls=ls,marker='o',ms=3,label=lbl)
    ax[1].set(xlabel='Radial spacing h',ylabel='Absolute eigenvalue',xlim=(.016,.10),ylim=(1e-6,1e-4))
    ax[1].set_xticks([.02,.04,.08],['0.02','0.04','0.08']); ax[1].set_yticks([1e-6,1e-5,1e-4],['10⁻⁶','10⁻⁵','10⁻⁴'])
    ax[1].xaxis.set_minor_locator(NullLocator()); ax[1].yaxis.set_minor_locator(NullLocator()); ax[1].legend(loc='lower right')
    save(fig,ax,'figure_3')
def figure4():
    fig,ax=layout(); panel(ax[0],'a','Nonradial scalar responses'); panel(ax[1],'b','Norm inside the core')
    for name,c,ls,label in [('scalar_mix96_T20',BLUE,'-','Mixed'),('scalar_quad96',TEAL,'--','Quadrupole'),('scalar_generic96',VIOLET,':','Asymmetric'),('scalar_control96',GREY,'-.','Control')]:
        d=pd.read_csv(D/(name+'.csv'))
        ax[0].plot(d.t,100*d.phase_H1,color=c,ls=ls,label=label)
        ax[1].plot(d.t,100*d.core,color=c,ls=ls)
    ax[0].set(xlabel='Time t',ylabel='Phase-only H¹ distance (%)',xlim=(0,20),ylim=(0,15)); ax[0].legend(loc='upper left',ncol=1)
    ax[1].set(xlabel='Time t',ylabel='Norm within r ≤ 15 (%)',xlim=(0,20),ylim=(99.78,100.))
    save(fig,ax,'figure_4')
if __name__=='__main__':
    figure2();figure3();figure4()
