"""Render executed two-field trajectories and both mesh-refinement comparisons."""
from pathlib import Path
import sys,json,subprocess,shutil,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
B=Path(__file__).resolve().parents[1]; ROOT=B.parent
D=B/'data'; F=B.parent/'figures'; Q=B/'qa'
for p in (D,F,Q):p.mkdir(exist_ok=True)
# Input snapshots reside beside the other processed data.
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.3,'legend.frameon':False})
BLUE='#165B96'; GREY='#777777'; TEAL='#358F88'; VIOLET='#955482'
fig=plt.figure(figsize=(160/25.4,165/25.4))
ax=[fig.add_axes([x,y,.335,.27]) for y in [.62,.19] for x in [.12,.635]]
def panel(a,label,title):
    a.text(0,1,label,transform=a.transAxes+ScaledTranslation(-18/72,6/72,a.figure.dpi_scale_trans),fontsize=9,fontweight='bold',va='bottom')
    a.set_title(title,loc='left',pad=11);a.tick_params(direction='out',length=3,pad=3)
for a,l,t in zip(ax,'abcd',['Dynamic coupled responses','Coupled core retention','Fieldwise refinement checks','Independent response velocity']):panel(a,l,t)
for name,c,ls,label in [('full_mix96',BLUE,'-','Mixed'),('full_generic96',VIOLET,':','Asymmetric'),('full_control96',GREY,'--','Control')]:
    d=pd.read_csv(D/(name+'.csv'))
    assert np.all(np.diff(d.t)>0)
    ax[0].plot(d.t,100*d.phase_H1,color=c,ls=ls,label=label)
    ax[1].plot(d.t,100*d.core,color=c,ls=ls)
    ax[3].plot(d.t,d.p_L2,color=c,ls=ls)
ax[0].set(xlabel='Time t',ylabel='Phase-only H¹ distance (%)',xlim=(0,20),ylim=(0,15));ax[0].legend(loc='upper left')
ax[1].set(xlabel='Time t',ylabel='Norm within r ≤ 15 (%)',xlim=(0,20),ylim=(99.78,100.))
comp=pd.read_csv(D/'full_field_comparisons.csv')
fine=next(x for x in json.loads((D/'mesh160_comparisons.json').read_text()) if x['run']=='full_mix128')
rows=[comp[(comp.run=='full_mix96')&(comp.reference==r)].iloc[0].to_dict() for r in ['full_mix128','full_time96','full_box144']]
rows.insert(1,fine)
extra=json.loads((D/'fine_refinement_comparisons.json').read_text())
t=next(x for x in extra if x['purpose']=='half_timestep_128')
b=next(x for x in extra if x['purpose']=='box_at_h0p5')
rows.insert(3,t);rows.append(b)
for key,c,mark,lab,off in [('u',BLUE,'o','Wave u',-.12),('e',TEAL,'s','Response e',0),('p',VIOLET,'^','Velocity p',.12)]:
    vals=np.array([r[key+'_L2_relative'] for r in rows],dtype=float)
    assert np.isfinite(vals).all() and (vals>0).all()
    ax[2].semilogy(np.arange(6)+off,vals,ls='none',marker=mark,color=c,ms=4,label=lab)
ax[2].set_xticks(range(6),['M1','M2','T96','T128','B96','B128'],fontsize=7.5)
ax[2].set(xlim=(-.5,5.5),ylim=(1e-7,10),ylabel='Relative field L² difference')
ax[2].legend(loc='upper right',fontsize=7.5)
ax[3].set(xlabel='Time t',ylabel='Response velocity L² norm',xlim=(0,20))
fig.savefig(F/'figure_5.pdf',dpi=600)
fig.savefig(F/'figure_5.svg',dpi=600)
fig.savefig(F/'figure_5.png',dpi=600)
fig.savefig(F/'figure_5.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
pd.DataFrame(rows).to_csv(D/'figure5_fullfield_comparisons.csv',index=False)
print('FIGURE5_RENDERED_FROM_SAVED_DATA; stored original audit records remain authoritative')
