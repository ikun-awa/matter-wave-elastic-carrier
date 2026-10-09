"""Run the archived CQ implementation only in a fresh external output folder."""
from pathlib import Path
import argparse, importlib.util, json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--run', action='store_true', help='Actually run all archived CQ simulations.')
args = parser.parse_args()
archive = Path(__file__).resolve().parents[1]
out = args.output.resolve()
if out == archive or archive in out.parents:
    parser.error('Output must be outside the frozen manuscript archive.')
if out.exists():
    parser.error('Output already exists. Choose a fresh directory; overwrite is forbidden.')
if not args.run:
    print(json.dumps({'validation': 'PASS', 'would_write_to': str(out), 'simulations_run': False}))
    raise SystemExit(0)
source = archive / 'code' / 'cq_benchmark_original.py'
spec = importlib.util.spec_from_file_location('cq_benchmark', source)
if spec is None or spec.loader is None:
    raise RuntimeError('Cannot load the archived calculation source.')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.BASE = out
module.DATA, module.QA = out / 'data', out / 'qa'
module.DATA.mkdir(parents=True)
module.QA.mkdir()
module.main()
