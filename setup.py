from setuptools import setup, find_packages

VERSION = '2.01' 
DESCRIPTION = 'Neural network based control charting for multiple stream processes'
LONG_DESCRIPTION = 'A Python package from the paper of Lepore, Palumbo, and Sposito, Neural network based control charting for multiple stream processes with an application to HVAC systems in passenger railway vehicles'

# Setting up

setup(
        name="NN4MSP", 
        version=VERSION,
        author='Antonio Lepore, Biagio Palumbo, Gianluca Sposito',
        author_email='antonio.lepore@unina.it, biagio.palumbo@unina.it, gianluca.sposito@unina.it',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        url='https://github.com/unina-sfere/NN4MSP',
        license='GNU General Public License v3.0',
        packages=find_packages(),
        install_requires=['numpy', 'matplotlib', 'sklearn', 'tensorflow', 'keras'],
        keywords=['Multiple stream process', 'Artificial neural networks', 'Statistical process control', 'Multilayer perceptron', 'Railway HVAC systems'],
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


