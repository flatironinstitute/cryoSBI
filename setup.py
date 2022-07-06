"""Create instructions to build cryo-Bife's path optimization version."""
import setuptools

requirements = []

setuptools.setup(
    name="cryo_em_SBI",
    maintainer=[
        "David Silva-Sánchez",
        "Lars Dingeldein",
        "Roberto Covino",
        "Pilar Cossio",
    ],
    version="0.0.1",
    maintainer_email=[
        "david.silva@yale.edu",
    ],
    description="Simulation-based inference of CryoEM data",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DSilva27/cryo_em_SBI.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
