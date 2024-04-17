from setuptools import setup, find_packages

install_requires = [
    'jax==0.3.14',
    'jax-md==0.1.28',
    'optax==0.1.2',
    'dm-haiku==0.0.6',
    'sympy==1.10.1',
    'tree_math',
    'cloudpickle',
    'chex',
    'blackjax==0.3.0',
    'coax==0.1.9',
    'gym==0.24.1'
]

with open('README.md', 'rt') as f:
    long_description = f.read()

setup(
    name='mof_al',
    version='0.1.0',
    license='Apache 2.0',
    description='Active learning for MOFs via GNNs.',
    author='Stephan Thaler',
    author_email='stephan.thaler@tum.de',
    packages=find_packages(exclude='examples'),
    python_requires='>=3.8',
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],
    zip_safe=False,
)
