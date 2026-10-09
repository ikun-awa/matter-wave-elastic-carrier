from pathlib import Path
import json,sys
import compare_fields as c
B=Path(__file__).resolve().parents[1]
pairs=[('full_mix96','full_mix128'),('full_mix128','full_mix160'),('full_mix128','full_time128_retry1'),('full_mix128','full_box192_retry1')]
expected={}
for f in ['fine_refinement_comparisons.json','mesh160_comparisons.json']:
 for r in json.loads((B/'data'/f).read_text()):expected[(r['run'],r['reference'])]=r
out=[]
for a,b in pairs:
 assert (c.R/a/'final.npz').exists() and (c.R/b/'final.npz').exists(),(a,b)
 r=c.compare(a,b);out.append(r)
 if (a,b) in expected:
  for key in ['u_L2_relative','e_L2_relative','p_L2_relative']:
   assert abs(r[key]-expected[(a,b)][key])<1e-10,(a,b,key)
 print('POSTPROCESS_PASS',a,b,flush=True)
(B/'qa'/'portable_comparisons.json').write_text(json.dumps(out,indent=2))
print('PORTABLE_FULL_FIELD_COMPARISONS_PASS')
