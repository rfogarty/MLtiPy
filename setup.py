from setuptools import find_packages, setup
setup(
    name='MLtiPy',
    packages=find_packages(),
    version='0.1.0',
    description='Python ML Utilities library',
    install_requires=['numpy','tensorflow-gpu','matplotlib','scikit-learn','scikit-image','opencv-python-headless','imutils'],
    author='Ryan Botet Fogarty',
    author_email='ryan.fogarty@gmail.com',
    license='Apache 2.0',
)

