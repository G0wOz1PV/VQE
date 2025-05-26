# VQE Optimizer Comparison (Round 006)

This project compares several optimization methods for the Variational Quantum Eigensolver (VQE) applied to the hydrogen molecule (H₂).  
The compared optimizers include:

- Velocity Verlet  
- COBYLA  
- L-BFGS-B  
- SLSQP  

The implementation is based on **Qulacs** and **OpenFermion**, and the results include:

- Convergence plots of each method  
- A summary performance table (CSV format)

## Dependencies

- Python 3.7+
- [`qulacs`](https://github.com/qulacs/qulacs)
- [`pyscf`](https://github.com/pyscf/pyscf)
- [`openfermion`](https://github.com/quantumlib/OpenFermion)
- [`openfermionpyscf`](https://github.com/quantumlib/OpenFermion-PySCF)
- `numpy==1.24`
- `scipy`
- `matplotlib`
- `pandas`
- `wurlitzer`

## Installation

You can install all required packages via pip:

```bash
pip install qulacs pyscf openfermion openfermionpyscf wurlitzer numpy==1.24 scipy matplotlib pandas
