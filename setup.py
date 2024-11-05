from setuptools import setup, find_packages

setup(
    name='CoordinationMetricsToolbox',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'generate_dataset=generate_testing_dataset:main',
            'run_tests=test:main',
        ],
    },
    author='Océane Dubois',
    author_email='',
    description='A toolbox for measuring and analyzing coordination metrics.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/oceanedbs/CoordinationMetricsToolbox',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)