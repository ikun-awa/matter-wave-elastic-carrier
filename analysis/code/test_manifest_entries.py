from pathlib import Path
B=Path(__file__).resolve().parents[1]
for p in ['code/profiles.py','code/audit_math.py','code/translation_check.py','code/render_figures234.py','data/legacy_cq_diagnostics.csv','data/spectral_checks.csv']:
 assert (B/p).is_file(), 'Missing current-manuscript reproduction input: '+p
print('CURRENT_MANUSCRIPT_REPRODUCTION_ENTRIES_PASS')
