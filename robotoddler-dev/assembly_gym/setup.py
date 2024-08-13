from setuptools import setup

setup(
    name="assembly_gym",
    packages=["assembly_gym"],
    version="0.0.1",
    install_requires=[
        # "pybullet==3.2.6",
        # "compas-fab==0.28.0",
        # "gym==0.26.2",
        # "shapely==2.0.3",
    ],
    entry_points={
        "console_scripts": [
            "assembly_tests=assembly_gym.utils.test_suite:main",
        ]
    }
)