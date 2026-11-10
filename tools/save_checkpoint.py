"""Save a Jupyter Contents snapshot returned by the browser helper."""
import argparse
import json
from pathlib import Path


def main() -> None:
    """Write captured text files within a chosen checkpoint root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    output = args.log.read_text(encoding='utf-8-sig')
    result = json.loads(output.split('### Result\n', 1)[1].split('\n### ', 1)[0])
    destination = args.destination.resolve()
    for name, content in result['files'].items():
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f'Checkpoint path escapes destination: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode('utf-8'))
    print('Saved', len(result['files']), 'files to', destination)


if __name__ == '__main__':
    main()
