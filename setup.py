from setuptools import setup, find_packages

VERSION = '1.01' 
DESCRIPTION = 'Neural network out-of-control signal interpretation in multiple stream processes'

def readme():
    with open('README.md',  encoding="cp437", errors='ignore') as f:
        return f.read()

# Setting up

setup(
        name="NN4OCMSP", 
        version=VERSION,
        author='Antonio Lepore, Biagio Palumbo, Gianluca Sposito',
        author_email='antonio.lepore@unina.it, biagio.palumbo@unina.it, gianluca.sposito@unina.it',
        description=DESCRIPTION,
        long_description=readme(),
        long_description_content_type="text/markdown",
        url='https://github.com/unina-sfere/NN4OCMSP',
        license='GNU General Public License v3.0',
        packages=find_packages(),
        install_requires=['numpy', 'matplotlib', 'sklearn', 'tensorflow', 'keras'],
        keywords=['Out-of-control signal interpretation', 'Artificial neural networks', 'Multiple stream process', 'Multi-label classification', 'Railway HVAC systems'],
        classifiers=[
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Manufacturing",
            "Intended Audience :: Developers",    
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
        include_package_data = True,
        package_data={'': ['data/*.csv']},
)


