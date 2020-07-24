from setuptools import setup, find_packages

setup(
    name='mundle-ai',
    version='1.1',
    packages=find_packages(),
    description='Model uncertainty and local explanations for AIs',
    url='https://github.com/AlexCardaillac/Mundle-AI',
    author='Alexandre Cardaillac',
    author_email='cardaillac.alexandre@gmail.com',
    license='BSD',
    python_requires='>=3.6.8',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'lime',
        'torch',
    ],
)
