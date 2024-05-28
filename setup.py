from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Robotoddler Project'
LONG_DESCRIPTION = ''

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="robotoddler",
    version=VERSION,
    author="Paul Rolland, Jingweng Wang, Johannes Kirschner",
    author_email="<youremail@email.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    entry_points={
        'console_scripts': [
            'robotoddler=robotoddler.main:main',
        ]
    }
)