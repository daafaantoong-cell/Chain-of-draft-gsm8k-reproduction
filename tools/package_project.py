"""Package the reproducible experiment and its evidence in the Downloads directory."""
import hashlib
import json
from pathlib import Path
import zipfile


def main() -> None:
    """Hash selected source and evidence files, then verify the completed ZIP."""
    root = Path(__file__).resolve().parents[1]
    required = ['README.md', 'config.json', 'experiment.py', 'test_experiment.py',
                'prepare_resources.py', 'report_results.py', 'CoD-Reproduction.ipynb',
                'CoD-Reproduction-executed.ipynb', 'report.md', 'verification.md']
    paths = [root / name for name in required]
    for directory in ['data', 'prompts', 'reference', 'runs', 'figures', 'debug']:
        paths.extend(path for path in (root / directory).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts)
    paths.extend(path for path in (root / 'tools').iterdir()
                 if path.suffix in {'.py', '.mjs'} and path.name != 'configure-notebook.mjs')
    paths = sorted(set(paths))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required artifacts missing: ' + ', '.join(missing))
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    manifest_path = root / 'artifact_manifest.json'
    manifest_path.write_text(json.dumps({'algorithm': 'SHA-256', 'files': hashes},
                                       ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    archive_path = root.parent / 'CoD-Reproduction.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths + [manifest_path]:
            archive.write(path, Path(root.name) / path.relative_to(root))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity verification failed')
        for name, digest in hashes.items():
            if hashlib.sha256(archive.read(root.name + '/' + name)).hexdigest() != digest:
                raise ValueError('Archived file hash differs: ' + name)
    print(json.dumps({'archive': str(archive_path), 'files': len(paths) + 1,
                      'bytes': archive_path.stat().st_size,
                      'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest()},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
