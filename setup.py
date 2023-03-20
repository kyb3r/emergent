from setuptools import setup, find_packages
from pathlib import Path
from sys import platform

CURRENT_DIR = Path(__file__).parent


def get_long_description() -> str:
    readme_md = CURRENT_DIR / "README.md"
    with open(readme_md, encoding="utf8") as ld_file:
        return ld_file.read()


install_requires = ["openai"]

setup(
    name="emergent",
    version="0.0.1",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=install_requires,
    packages=["emergent"], # "emergent/data"
)