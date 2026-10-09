# Matter-wave elastic-carrier model: reproducibility materials

Research author: **Chunsheng Li**, independent researcher, Philadelphia, Pennsylvania, USA. Contact: elephantintheroom202ls@gmail.com.

This repository supports *Adding matter to matter waves: a candidate elastic-carrier model, constrained stability and reduction limits*. It contains a specified mathematical model and deterministic numerical results, not experimental evidence for microscopic constituents of space. The paper is prepared for submission; no journal acceptance or independent external peer review is claimed.

## Scope and licenses

Original numerical code: **MIT**. Generated numerical data: **CC BY 4.0**. See LICENSE, LICENSE-DATA.txt and LICENSING.md. The manuscript and conceptual Figure1 are outside this release's license scope.

The 21 completed Cartesian cases are indexed in analysis/data/run_manifest.json. Failed attempts are not counted as completed cases. The wave, response and response time derivative are compared separately. Small spatial differences do not remove the approximately 1.33% wave box sensitivity. Selected finite-time trajectories are not a general orbital-stability proof. Negative-energy scalar minimizers and the computed positive-energy state are different objects.

## Obtain the data

Clone or download the source. From Releases / v1.0.0, download **both** data-part-01.zip and data-part-02.zip and extract both into the repository root. These are two ordinary ZIPs with disjoint data paths, not a split binary stream. Verify their SHA-256 values using RELEASE_ASSETS.json; verify every data file with DATA_MANIFEST.json. No paid services or API keys are required. Large raw NPZ files are release assets, not Git blobs.

## Reproduce (Python 3.12 tested)

```
python -m pip install -r requirements.txt
python analysis/code/test_core.py
python analysis/code/test_manifest_entries.py
python methods/symbolic_review.py
python analysis/code/reproduce_comparisons.py
python analysis/code/render_figures234.py
python analysis/code/render_figure5.py
```

The main comparisons remove one global phase only from the complex wave. Real response fields have no phase freedom. Larger-box differences include the reference norm outside the common cube. These are deterministic solution differences, not confidence intervals or certified continuum-limit errors.

Representative stationary/spectral calculations can be regenerated with `python analysis/code/profiles.py`, `python analysis/code/audit_math.py`, and `python analysis/code/translation_check.py`. They write derived files; run them in a disposable copy to keep the frozen outputs unchanged. The 47-point scalar branch and 33-point CQ branch are separately archived benchmarks. Benchmark solver wrappers retain their explicit model and output-directory rules. Legacy illustration scripts for previous drafts are excluded; the supported quantitative-figure entrypoints are the ones above.

Full PDE reruns take longer and require substantial RAM. Use a **new** case name, e.g. `python analysis/code/run_cartesian.py --name rerun_time128 --n 128 --L 64 --dt .01 --T 20 --full`. Existing raw cases are not overwritten. The coupled CLI is the fixed beta=0.20, omega=0.10 validation implementation, not a general-purpose parameter explorer. Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 to avoid excess threading. Float64/complex128 and four SciPy FFT workers were used.

## Verification and provenance

A fresh directory was used to execute regression tests, exact symbolic checks, four complete-field comparisons, representative stationary/spectral reconstruction, translation fits, and all four quantitative figure renderers. The local original figure workflow used nature-figure 2.8.0 with alignment/font/collision audits; the optional NATURE_FIGURE_ROOT environment hook enables these audits when that separately installed workflow is available. Standalone plotting does not require it and does not silently claim fresh audit certification. Summary: verification/REPRODUCTION.json.

AI assistance was used for mathematical drafting, code and manuscript preparation. The human author is responsible for the work. Reproduction checks do not certify every analytic hypothesis, originality or physical interpretation. No DOI is assigned by this GitHub release. Use CITATION.cff and the exact release tag when citing these materials.
