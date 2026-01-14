# VQE Optimizer Comparison


[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Qulacs](https://img.shields.io/badge/Qulacs-Quantum%20Simulator-brightgreen.svg)](https://github.com/qulacs/qulacs)
[![PySCF](https://img.shields.io/badge/PySCF-Quantum%20Chemistry-orange.svg)](https://github.com/pyscf/pyscf)
[![OpenFermion](https://img.shields.io/badge/OpenFermion-Quantum%20Computing-blueviolet.svg)](https://github.com/quantumlib/OpenFermion)
[![OpenFermion-PySCF](https://img.shields.io/badge/OpenFermion--PySCF-Integration-red.svg)](https://github.com/quantumlib/OpenFermion-PySCF)
[![NumPy](https://img.shields.io/badge/NumPy-1.24-lightgrey.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-latest-blue.svg)](https://www.scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange.svg)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue.svg)](https://pandas.pydata.org/)
[![Wurlitzer](https://img.shields.io/badge/Wurlitzer-Linux%20IO-purple.svg)](https://github.com/minrk/wurlitzer)

This project compares several optimization methods for the Variational Quantum Eigensolver (VQE) applied to the hydrogen molecule (H₂).  
The compared optimizers include:

- Velocity Verlet  
- COBYLA  
- L-BFGS-B  
- SLSQP  

The implementation is based on **Qulacs** and **OpenFermion**, and the results include:

- Convergence plots of each method  
- A summary performance table (CSV format)
