import sys
import re
from pathlib import Path

def bump_version(part):
    # --- 1. Update pyproject.toml ---
    toml_path = Path("pyproject.toml")
    toml_content = toml_path.read_text()

    version_pattern = r'version = "(\d+)\.(\d+)\.(\d+)"'
    match = re.search(version_pattern, toml_content)

    if not match:
        print("Error: Could not find version in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    major, minor, patch = map(int, match.groups())

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
    old_version = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"

    new_toml_content = re.sub(version_pattern, f'version = "{new_version}"', toml_content)
    toml_path.write_text(new_toml_content)

    # --- 2. Update Documentation Files ---
    files_to_update = ["README.md", "index.qmd","src/digiqual/__init__.py","docs/install.py"]

    for filename in files_to_update:
        file_path = Path(filename)
        if file_path.exists():
            content = file_path.read_text()
            new_content = content.replace(f"{old_version}", f"{new_version}")
            file_path.write_text(new_content)
            print(f"Updated {filename}", file=sys.stderr)

    print(new_version)

if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "patch"
    bump_version(part)
