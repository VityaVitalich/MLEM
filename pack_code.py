from pathlib import Path
from typing import List
import json
from argparse import ArgumentParser


EXTENSIONS = {".py", ".yaml", ".txt", ".toml", ".md", ".sh", ".ipynb"}
EXCLUDE_DIRS = {"__pycache__", "data", "logs", "ckpt"}


def encode_dir(d: Path) -> List:
    res = []
    for p in d.iterdir():
        if p.is_dir() and not p.name[0] == "." and p.name not in EXCLUDE_DIRS:
            res.append([p.name, encode_dir(p)])
        elif p.is_file() and p.suffix in EXTENSIONS:
            res.append([p.name, p.read_text()])

    return res


def decode_dir(base_path: Path, d: List) -> None:
    for p, content in d:
        path = base_path / p
        if isinstance(content, list):
            path.mkdir()
            decode_dir(path, content)
        else:
            path.write_text(content)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("action", choices=["c", "x"], help="create or extract the JSON")
    parser.add_argument(
        "--path", help="where to start creation or extraction", default="."
    )
    parser.add_argument(
        "--file", help="JSON file with the representation", default="encoded.json"
    )
    args = parser.parse_args()

    if args.action == "c":
        Path(args.file).write_text(json.dumps(encode_dir(Path(args.path))))
    else:
        path = Path(args.path)
        path.mkdir(parents=True)
        decode_dir(path, json.loads(Path(args.file).read_text()))
