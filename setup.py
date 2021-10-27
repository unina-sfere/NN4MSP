from setuptools import setup, find_packages

VERSION = '0.07' 
DESCRIPTION = 'Neural network based control charting for multiple stream processes'
LONG_DESCRIPTION = 'Python package from the paper of Lepore, Palumbo, and Sposito, Neural network based control charting for multiple stream processes with an application to HVAC systems in passenger railway vehicles'

# Setting up

setup(
        name="NNforMSP", 
        version=VERSION,
        author=['Antonio Lepore', 'Biagio Palumbo', 'Gianluca Sposito'],
        author_email=['antonio.lepore@unina.it', 'biagio.palumbo@unina.it', 'gianluca.sposito@unina.it'],
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        url='https://github.com/gianlucasposito/NNforMSP',
        license='GNU General Public License v3.0',
        packages=find_packages(),
        install_requires=['numpy', 'matplotlib', 'sklearn', 'tensorflow', 'keras'],
        keywords=['Multiple stream process', 'Artificial neural networks', 'Statistical process control', 'Multilayer perceptron', 'Railway HVAC systems'],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education, Researchers, Practitioners",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)


