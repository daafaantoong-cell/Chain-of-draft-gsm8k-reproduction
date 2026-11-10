"""Decode a Playwright CLI result into an explicitly chosen local directory."""
import argparse
import base64
import io
import json
from pathlib import Path
import zipfile


def main() -> None:
    """Extract an exported runs archive after checking every destination path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    output = args.log.read_text(encoding='utf-8-sig')
    value = json.loads(output.split('### Result\n', 1)[1].split('\n### ', 1)[0])
    archive_bytes = base64.b64decode(value, validate=True)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for name in archive.namelist():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Export path escapes destination: {name}')
        archive.extractall(destination)
        print('Extracted', len(archive.namelist()), 'files to', destination)


if __name__ == '__main__':
    main()
