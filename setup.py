from setuptools import setup, find_packages

setup(
    name='synax',  # This is the name of your package
    version='0.1.0',  # Version number
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=['jax','numpy','scipy','interpax','healpy'],  # List any dependencies your package requires
    author='Kangning Diao, Zack Li, Richard D.P. Grumitt',  # Your name
    author_email='dkn16@foxmail.com',  # Your email
    description='A brief description of your package',
    long_description=open('README.md').read(),  # Long description read from a README file
    long_description_content_type='text/markdown',
    url='https://github.com/dkn16/Synax',  # URL to the package's repository
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum version of Python required
)