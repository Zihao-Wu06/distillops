"""Rewrite README.md in place so its links and images work on PyPI.

PyPI shows the README without the repository around it: relative links and images break, and
headings have no anchors. Relative links point at GitHub, images at raw files, and in-page
anchors at the README on GitHub, all at the given git ref (the release tag).

    python .github/scripts/pypi_readme.py README.md v0.1.0
"""
import re
import sys

REPO = "Zihao-Wu06/distillops"


def rewrite(text: str, ref: str) -> str:
    blob = f"https://github.com/{REPO}/blob/{ref}/"
    raw = f"https://raw.githubusercontent.com/{REPO}/{ref}/"
    page = f"https://github.com/{REPO}/tree/{ref}"
    absolute = r"(?![a-z][a-z0-9+.-]*:|/)"  # not http:, https:, mailto:, ... and not site-absolute
    text = re.sub(r'(src=")' + absolute + r"(?!#)([^\"]+)\"", lambda m: f'{m[1]}{raw}{m[2]}"', text)
    text = re.sub(r"(\]\()#([^)]+)\)", lambda m: f"{m[1]}{page}#{m[2]})", text)
    text = re.sub(r"(\]\()" + absolute + r"(?!#)([^)\s]+)\)", lambda m: f"{m[1]}{blob}{m[2]})", text)
    return text


if __name__ == "__main__":
    path, ref = sys.argv[1], sys.argv[2]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(rewrite(text, ref))
