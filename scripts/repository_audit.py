"""Проверяет состав Git без изменения индекса или локальных файлов."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit(root=ROOT):
    def git(*args, data=None):
        return subprocess.run(["git", *args], cwd=root, input=data, capture_output=True, check=False).stdout
    tracked = set(filter(None, git("ls-files", "-z").decode().split("\0")))
    candidates = set(filter(None, git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")))
    ignored = set(filter(None, git("check-ignore", "--no-index", "-z", "--stdin", data="\0".join(sorted(candidates)).encode()).decode().split("\0")))
    included = sorted(p for p in candidates - ignored if (root / p).is_file())
    forbidden = [p for p in included if p.startswith(("data/inputs/", ".aws/", ".agents/", ".codex/", ".venv")) or Path(p).name.startswith(".env") and Path(p).name != ".env.example"]
    oversized = [p for p in included if (root / p).stat().st_size > 20 * 1024**2]
    ignored_tracked = sorted(tracked & ignored)
    return {"included_files": len(included), "included_bytes": sum((root / p).stat().st_size for p in included),
            "ignored_but_indexed": ignored_tracked, "forbidden": forbidden, "over_20_mib": oversized,
            "passed": not (ignored_tracked or forbidden or oversized)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
