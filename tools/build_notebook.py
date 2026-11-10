"""Build a self-contained Kaggle notebook from the frozen local experiment inputs."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


def main() -> None:
    """Embed the validated code, official prompts and complete dataset into one notebook."""
    root = Path(__file__).resolve().parents[1]
    paths = [root / name for name in ['experiment.py', 'test_experiment.py', 'prepare_resources.py', 'config.json']]
    for directory in ['data', 'prompts', 'reference']:
        paths.extend(path for path in (root / directory).rglob('*') if path.is_file())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if '__pycache__' in path.parts:
                raise ValueError(f'Frozen inputs contain a bytecode cache: {path}')
            archive.write(path, path.relative_to(root).as_posix())
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode('ascii')
    digest = hashlib.sha256(payload).hexdigest()
    sources = [
        f'''from pathlib import Path
import base64, hashlib, io, json, os, subprocess, sys, zipfile
ROOT = Path('/kaggle/working/cod-reproduction')
ROOT.mkdir(parents=True, exist_ok=True)
PAYLOAD = {encoded!r}
bundle = base64.b64decode(PAYLOAD)
assert hashlib.sha256(bundle).hexdigest() == {digest!r}
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    archive.extractall(ROOT)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'PyYAML==6.0.3', 'transformers==4.57.1', 'accelerate==1.10.1'], check=True)
import torch
print('Python', sys.version.split()[0], 'torch', torch.__version__)
print('GPUs', [(torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1)) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available(), 'Enable GPU T4 x2 and Internet before running'
from prepare_resources import verify_local_resources
verify_local_resources()
''',
        '''import unittest
suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_experiment.py')
test_result = unittest.TextTestRunner(verbosity=2).run(suite)
assert test_result.wasSuccessful(), 'Focused validation failed'
''',
        '''from experiment import run_experiment
pilot_summary = run_experiment(ROOT / 'config.json', 'pilot', ROOT / 'runs/pilot')
print(json.dumps(pilot_summary, ensure_ascii=False, indent=2))
''',
        '''full_summary = run_experiment(ROOT / 'config.json', 'full', ROOT / 'runs/full')
print(json.dumps(full_summary, ensure_ascii=False, indent=2))
''',
        '''with (ROOT / 'runs/requirements-freeze.txt').open('w', encoding='utf-8') as handle:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=handle, check=True)
with zipfile.ZipFile('/kaggle/working/cod-results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in (ROOT / 'runs').rglob('*'):
        if path.is_file():
            archive.write(path, path.relative_to(ROOT))
print('Saved /kaggle/working/cod-results.zip')
'''
    ]
    for i, source in enumerate(sources):
        compile(source, f'cell-{i}', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    cells = [dict(cell_type='code', source=source, metadata={}, outputs=[], execution_count=None) for source in sources]
    notebook = dict(nbformat=4, nbformat_minor=4,
                    metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')), cells=cells)
    (root / 'CoD-Reproduction.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    script = "async page=>{return await page.frame({name:'notebook-editor-cells'}).evaluate(cells=>{const w=window.jupyterapp.shell.currentWidget;const j=w.model.toJSON();j.cells=cells;w.model.fromJSON(j);return {configured:j.cells.length}}," + json.dumps(cells, ensure_ascii=False) + ')}'
    (root / 'tools/configure-notebook.mjs').write_text(script, encoding='utf-8')
    print('Notebook built:', len(cells), 'code cells;', len(payload), 'embedded bytes; bundle SHA256', digest)


if __name__ == '__main__':
    main()
