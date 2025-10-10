try:
    import google.colab
    print("Google Colab environment detected. Installing necessary packages...")
    !pip install qulacs pyscf openfermion openfermionpyscf --quiet
    !pip install py-cpuinfo --quiet
    !pip install numpy==1.24 --quiet
except ImportError:
    print("Not in a Google Colab environment.")

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import minimize
import cpuinfo

# PySCF
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner

# Qulacs
from qulacs import Observable, QuantumState, QuantumCircuit
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.gate import CZ, RY, RZ, merge




class EvaluationCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

    def reset(self):
        self.count = 0

def get_cpu_info():
    try:
        info = cpuinfo.get_cpu_info()
        return info['brand_raw']
    except:
        return "Could not determine CPU info."


def setup_molecule_and_hamiltonian(name, distance):
    print(f"Setting up molecule: {name} at distance {distance} Å")
    basis = "sto-3g"
    multiplicity = 1

    if name.lower() == 'h2':
        charge = 0
        geometry = [["H", [0, 0, 0]], ["H", [0, 0, distance]]]
    elif name.lower() == 'lih':
        charge = 0
        geometry = [["Li", [0, 0, 0]], ["H", [0, 0, distance]]]
    else:
        raise ValueError(f"Molecule '{name}' is not supported.")

    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)

    n_qubit = molecule.n_qubits
    n_electron = molecule.n_electrons

    fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    qulacs_hamiltonian = create_observable_from_openfermion_text(str(jw_hamiltonian))

    print(f"Number of qubits: {n_qubit}")
    print(f"Number of electrons: {n_electron}")
    print(f"FCI (exact) energy: {molecule.fci_energy:.10f} Hartree")

    return molecule, qulacs_hamiltonian, n_qubit, n_electron


def he_ansatz_circuit(n_qubit, depth, theta_list):
    circuit = QuantumCircuit(n_qubit)
    num_params_per_layer = 2 * n_qubit

    for d in range(depth):
        for i in range(n_qubit):
            circuit.add_gate(merge(RY(i, theta_list[d * num_params_per_layer + 2 * i]),
                                  RZ(i, theta_list[d * num_params_per_layer + 2 * i + 1])))

        for i in range(n_qubit//2):
            circuit.add_gate(CZ(2*i, 2*i+1))
        for i in range(n_qubit//2-1):
            circuit.add_gate(CZ(2*i+1, 2*i+2))


    d = depth
    for i in range(n_qubit):
        circuit.add_gate(merge(RY(i, theta_list[d * num_params_per_layer + 2 * i]),
                              RZ(i, theta_list[d * num_params_per_layer + 2 * i + 1])))

    return circuit

def create_cost_and_gradient_functions(n_qubit, depth, qulacs_hamiltonian):

    def cost_func(theta_list):
        state = QuantumState(n_qubit)
        circuit = he_ansatz_circuit(n_qubit, depth, theta_list)
        circuit.update_quantum_state(state)
        return qulacs_hamiltonian.get_expectation_value(state).real


    cost_func_counted = EvaluationCounter(cost_func)

    def compute_gradient(theta_list, eps=1e-4):
        gradient = np.zeros_like(theta_list)
        for i in range(len(theta_list)):
            theta_plus = theta_list.copy()
            theta_plus[i] += eps
            theta_minus = theta_list.copy()
            theta_minus[i] -= eps
            grad_i = (cost_func_counted(theta_plus) - cost_func_counted(theta_minus)) / (2 * eps)
            gradient[i] = grad_i
        return gradient

    return cost_func_counted, compute_gradient



def velocity_verlet_optimizer(cost_func, gradient_func, init_params, vv_params, max_iter=50, tol=1e-10):
    params = init_params.copy()
    velocity = np.zeros_like(params)

    cost_history = []
    eval_count_history = []

    initial_cost = cost_func(params)
    cost_history.append(initial_cost)
    eval_count_history.append(cost_func.count)

    for i in range(max_iter):
        start_eval_count = cost_func.count

        force = -gradient_func(params)
        velocity += 10000 * vv_params['dt'] * force / vv_params['mass']
        params += vv_params['dt'] * velocity
        force_new = -gradient_func(params)
        velocity += 100 * vv_params['dt'] * force_new / vv_params['mass']
        velocity *= vv_params['damping']

        current_cost = cost_func(params)
        cost_history.append(current_cost)
        eval_count_history.append(cost_func.count)

        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < tol:
            print(f"Velocity Verlet converged at iteration {i}")
            break

        if i % 5 == 0 or i == max_iter - 1:
            print(f"VV - Iter {i}, Cost: {current_cost:.8f}, Evals: {cost_func.count}")

    while len(cost_history) < max_iter + 1:
        cost_history.append(cost_history[-1])
        eval_count_history.append(eval_count_history[-1])

    return params, cost_history, eval_count_history

def run_scipy_optimizer(method, init_params, cost_func, gradient_func, max_iter):

    cost_history = [cost_func(init_params)]
    eval_counts = [cost_func.count]
    param_history = [init_params.copy()] 
    def callback(xk):
        if len(cost_history) > max_iter:
            return

        cost = cost_func(xk)
        cost_history.append(cost)
        eval_counts.append(cost_func.count)
        param_history.append(xk.copy())
        
        current_iter = len(cost_history) - 1
        if current_iter > 0 and (current_iter % 5 == 0 or current_iter == max_iter):
             print(f"{method} - Iter {current_iter}, Cost: {cost:.8f}, Evals: {cost_func.count}")

    start_time = time.time()
    
    if method == "COBYLA":
        options = {'maxfun': 5000} 
        result = minimize(cost_func, init_params, method=method,
                          options=options, callback=callback, tol=1e-6)
    else:
        options = {'maxiter': max_iter}
        result = minimize(cost_func, init_params, method=method,
                          jac=gradient_func, options=options, callback=callback)

    end_time = time.time()
    optimization_time = end_time - start_time
    
    final_cost_history = cost_history[:max_iter + 1]
    final_eval_counts = eval_counts[:max_iter + 1]
    final_param_history = param_history[:max_iter + 1]

    while len(final_cost_history) < max_iter + 1:
        final_cost_history.append(final_cost_history[-1])
        final_eval_counts.append(final_eval_counts[-1])
        final_param_history.append(final_param_history[-1])
        
    final_params = final_param_history[-1]

    return final_params, final_cost_history, final_eval_counts, optimization_time




def compare_optimizers(cost_func, gradient_func, n_qubit, depth, fci_energy, vv_params, max_iter=40):
    np.random.seed(18)
    num_params = 2 * n_qubit * (depth + 1)
    init_params = np.random.random(num_params) * 1e-1

    results = {}


    print("\n=== Running Velocity Verlet ===")
    cost_func.reset()
    start_time = time.time()
    vv_params_opt, vv_history, vv_evals = velocity_verlet_optimizer(
        cost_func, gradient_func, init_params, vv_params, max_iter=max_iter
    )
    vv_time = time.time() - start_time
    results["Velocity Verlet"] = {
        "params": vv_params_opt, "history": vv_history, "evals": vv_evals, "time": vv_time
    }

    
    methods = ["L-BFGS-B", "SLSQP", "COBYLA"]
    for method in methods:
        print(f"\n=== Running {method} ===")
        cost_func.reset()
        opt_params, opt_history, opt_evals, opt_time = run_scipy_optimizer(
            method, init_params, cost_func, gradient_func, max_iter
        )
        results[method] = {
            "params": opt_params, "history": opt_history, "evals": opt_evals, "time": opt_time
        }

    for method, res in results.items():
        final_energy = res['history'][-1]
        res["final_energy"] = final_energy
        res["error"] = abs(final_energy - fci_energy)

    return results
    
def visualize_results(results, fci_energy, molecule_name, cpu_info):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Noto Serif CJK JP', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
    })

    plt.close('all')

    colors = {"Velocity Verlet": "#636EFA", "L-BFGS-B": "#00CC96", "SLSQP": "#EF553B", "COBYLA": "#FFB000"}

    chemical_accuracy = 1.6e-3

    plt.figure(figsize=(10, 7))
    for method, res in results.items():
        plt.plot(res["history"], label=method, color=colors[method], linewidth=2.5)
    plt.axhline(y=fci_energy, color='black', linestyle='--', label=f'FCI Energy ({fci_energy:.6f})')
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Hartree)")
    plt.title(f"VQE Convergence ({molecule_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{molecule_name}_convergence_vs_iterations.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 7))
    for method, res in results.items():
        errors = [abs(e - fci_energy) for e in res["history"]]
        plt.semilogy(errors, label=method, color=colors[method], linewidth=2.5)
    plt.axhline(y=chemical_accuracy, color='red', linestyle=':', label=f'Chemical Accuracy ({chemical_accuracy:.1e})')
    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error (Hartree) [Log Scale]")
    plt.title(f"VQE Error Convergence ({molecule_name})")
    plt.legend()
    plt.ylim(bottom=1e-7)
    plt.tight_layout()
    plt.savefig(f"{molecule_name}_error_vs_iterations.png", dpi=300)
    plt.show()


    plt.figure(figsize=(10, 7))
    for method, res in results.items():
        plt.semilogy(res["evals"], [abs(e - fci_energy) for e in res["history"]], label=method, color=colors[method], linewidth=2.5)
    plt.axhline(y=chemical_accuracy, color='red', linestyle=':', label=f'Chemical Accuracy ({chemical_accuracy:.1e})')
    plt.xlabel("Number of Energy Evaluations")
    plt.ylabel("Absolute Error (Hartree) [Log Scale]")
    plt.title(f"VQE Cost vs. Accuracy ({molecule_name})")
    plt.legend()
    plt.ylim(bottom=1e-7)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(f"{molecule_name}_error_vs_evaluations.png", dpi=300)
    plt.show()

    data = []
    for method, res in results.items():

        evals_to_chem_acc = "N/A"
        errors = [abs(e - fci_energy) for e in res["history"]]
        for i, error in enumerate(errors):
            if error < chemical_accuracy:
                evals_to_chem_acc = res["evals"][i]
                break

        data.append({
            "Optimizer": method,
            "Final Energy": f"{res['final_energy']:.10f}",
            "Final Error": f"{res['error']:.4e}",
            "Total Evals": res["evals"][-1],
            "Time (s)": f"{res['time']:.2f}",
            "Evals to Chem. Acc.": evals_to_chem_acc
        })
    df = pd.DataFrame(data)

    print("\n" + "="*50)
    print(f"Performance Summary for {molecule_name}")
    print("="*50)
    print(f"CPU Used: {cpu_info}")
    print(f"FCI Energy: {fci_energy:.10f} Hartree")
    print(df.to_string(index=False))
    print("="*50 + "\n")

    df.to_csv(f'{molecule_name}_performance_summary.csv', index=False)
    return df

    molecule, hamiltonian, n_qubit, n_electron = setup_molecule_and_hamiltonian(
        molecule_config['name'], molecule_config['distance']
    )

    depth = molecule_config.get('depth', n_qubit)
    cost_func, gradient_func = create_cost_and_gradient_functions(n_qubit, depth, hamiltonian)

    results = compare_optimizers(
        cost_func, gradient_func, n_qubit, depth,
        molecule.fci_energy, molecule_config['vv_params'], max_iter=molecule_config['max_iter']
    )

    performance_df = visualize_results(results, molecule.fci_energy, molecule_config['name'], cpu_info)

    return performance_df

if __name__ == "__main__":
    # CPU
    cpu_info = get_cpu_info()
    print(f"--- VQE Optimizer Benchmark ---")
    print(f"Running on: {cpu_info}\n")


    molecules_to_run = [
                {
            "name": "H2",
            "distance": 0.977,
            "max_iter": 40,
            "depth": 4,
            "vv_params": {'dt': 0.01, 'mass': 0.8, 'damping': 0.68}
        },
         {
            "name": "LiH",
            "distance": 1.596,
            "max_iter": 40,
            "depth": 4,
            "vv_params": {'dt': 0.01, 'mass': 1.9, 'damping': 0.68}
        }
    ]

    all_results = {}
    for config in molecules_to_run:
        df = run_simulation_for_molecule(config, cpu_info)
        all_results[config['name']] = df

    print("\n--- All simulations completed. ---")
