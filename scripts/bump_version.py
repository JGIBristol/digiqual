import re
import sys
from datetime import date
from pathlib import Path

VERSION = r"\d+\.\d+\.\d+"

# Where the version appears, other than pyproject.toml. Each entry is a regex
# with the version in a group named `v`, so only these exact spots change (a
# blind find-and-replace would also hit any unrelated "x.y.z" in the file).
VERSION_LOCATIONS = {
    "src/digiqual/__init__.py": [rf'^__version__ = "(?P<v>{VERSION})"'],
    "README.md": [rf"release \(v(?P<v>{VERSION})\)"],
    "docs/install.qmd": [rf"release \(v(?P<v>{VERSION})\)"],
    "app/pyproject.toml": [rf'^version = "(?P<v>{VERSION})"', rf'"digiqual==(?P<v>{VERSION})"'],
    "CITATION.cff": [rf"^version: (?P<v>{VERSION})"],
}


def _replace_group(pattern: str, content: str, new_version: str) -> tuple[str, int]:
    def swap(m: re.Match) -> str:
        start, end = m.span("v")
        return m.group(0)[: start - m.start()] + new_version + m.group(0)[end - m.start():]
    return re.subn(pattern, swap, content, flags=re.MULTILINE)


def bump_version(part):
    # --- 1. Update pyproject.toml ---
    toml_path = Path("pyproject.toml")
    toml_content = toml_path.read_text()

    version_pattern = rf'^version = "({VERSION})"'
    match = re.search(version_pattern, toml_content, flags=re.MULTILINE)

    if not match:
        print("Error: Could not find version in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    major, minor, patch = map(int, match.group(1).split("."))

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    else: # patch
        patch += 1

    new_version = f"{major}.{minor}.{patch}"

    new_toml_content = re.sub(version_pattern, f'version = "{new_version}"', toml_content, count=1, flags=re.MULTILINE)
    toml_path.write_text(new_toml_content)

    # --- 2. Update the other files that carry the version ---
    for filename, patterns in VERSION_LOCATIONS.items():
        file_path = Path(filename)
        if not file_path.exists():
            continue
        content = file_path.read_text()
        n_total = 0
        for pattern in patterns:
            content, n = _replace_group(pattern, content, new_version)
            n_total += n
        if filename == "CITATION.cff":
            content, n = re.subn(r"^date-released: .*$", f"date-released: {date.today().isoformat()}",
                                 content, flags=re.MULTILINE)
            n_total += n
        file_path.write_text(content)
        print(f"Updated {filename} ({n_total} change(s))", file=sys.stderr)

    print(new_version)

if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "patch"
    bump_version(part)
