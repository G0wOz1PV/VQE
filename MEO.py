from jax import numpy as np
import jax
import pennylane as qml
import time
import matplotlib.pyplot as plt
import optax

# JAX 
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

# lightning.qubit
dev = qml.device("lightning.qubit", wires=qubits)


def market_equilibrium_optimizer(params, grad, prev_adjust, alpha=0.1, beta=0.05):
    adjustment = -alpha * grad + beta * prev_adjust
    new_params = params + adjustment
    return new_params, adjustment


molecules = [
    ("H2", ["H", "H"], np.array([[0., 0., 0.], [0., 0., 0.74]])),
    ("LiH", ["Li", "H"], np.array([[0., 0., 0.], [0., 0., 1.60]])),
    ("BeH2", ["Be", "H", "H"], np.array([[0., 0., 0.], [0., 0., 1.326], [0., 0., -1.326]])),
    ("H2O", ["O", "H", "H"], np.array([[0., 0., 0.], [0., -0.757, 0.587], [0., 0.757, 0.587]]))
]

max_iters = 200
tol = 1e-8

results = []
all_param_histories = {} # Dictionary to store histories for each molecule


for name, symbols, coordinates in molecules:
    print(f"\n=== {name} 計算開始 ===")
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, basis="sto-3g")
    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev, interface="jax")
    def cost_fn(params):
        occ = [1 if i < (len(symbols) * 2) // 2 else 0 for i in range(qubits)]
        qml.BasisState(np.array(occ), wires=range(qubits))
        # Assuming the same ansatz structure as in the original code
        if len(params) >= 1:
            qml.SingleExcitation(params[0], wires=[2, 0])
        if len(params) >= 2:
            qml.SingleExcitation(params[1], wires=[3, 1])
        if len(params) >= 3:
            qml.DoubleExcitation(params[2], wires=[2, 3, 0, 1])
        return qml.expval(hamiltonian)


    grad_fn = jax.grad(cost_fn)

    # Initialize parameters based on the number of qubits and the ansatz
    num_params = 0
    if qubits >= 2:
        num_params += 1 # SingleExcitation
    if qubits >= 4:
        num_params += 1 # SingleExcitation
    if qubits >= 4:
        num_params += 1 # DoubleExcitation

    init_params = np.array([0.01] * num_params)


    # --- MEO ---
    params = init_params.copy()
    prev_adjust = np.zeros_like(init_params)
    energies_me = []
    param_history_me = [params.copy()] # Store initial parameters
    start_time_me = time.time()
    for i in range(max_iters):
        energy = cost_fn(params)
        energies_me.append(energy)
        grad = grad_fn(params)
        params, prev_adjust = market_equilibrium_optimizer(params, grad, prev_adjust)
        param_history_me.append(params.copy()) # Store parameters after update
        if i > 0 and np.abs(energies_me[-1] - energies_me[-2]) < tol:
            break
    elapsed_me = time.time() - start_time_me

    # --- Adam ---
    opt = optax.adam(learning_rate=0.1)
    opt_state = opt.init(init_params)
    params_adam = init_params.copy()
    energies_adam = []
    param_history_adam = [params_adam.copy()] # Store initial parameters
    start_time_adam = time.time()
    for i in range(max_iters):
        energy = cost_fn(params_adam)
        energies_adam.append(energy)
        grad = grad_fn(params_adam)
        updates, opt_state = opt.update(grad, opt_state)
        params_adam = optax.apply_updates(params_adam, updates)
        param_history_adam.append(params_adam.copy()) # Store parameters after update
        if i > 0 and np.abs(energies_adam[-1] - energies_adam[-2]) < tol:
            break
    elapsed_adam = time.time() - start_time_adam

    results.append({
        "Molecule": name,
        "ME iters": len(energies_me),
        "ME time (s)": elapsed_me,
        "ME Energy": float(energies_me[-1]),
        "Adam iters": len(energies_adam),
        "Adam time (s)": elapsed_adam,
        "Adam Energy": float(energies_adam[-1])
    })

    all_param_histories[name] = {
        "ME": np.array(param_history_me),
        "Adam": np.array(param_history_adam)
    }


    plt.figure()
    plt.plot(energies_me, label="Market Equilibrium Optimizer")
    plt.plot(energies_adam, label="Adam Optimizer")
    plt.xlabel("Iteration")
    plt.ylabel("Energy [Ha]")
    plt.title(f"{name} - VQE Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()


import pandas as pd
df = pd.DataFrame(results)
print("\n=== ベンチマーク結果 ===")
print(df.to_string(index=False))

# CSV
df.to_csv("vqe_benchmark_results.csv", index=False)

# Combine all parameter histories into a single dataset for PCA
all_params = []
for name in all_param_histories:
    all_params.extend(all_param_histories[name]['ME'])
    all_params.extend(all_param_histories[name]['Adam'])

all_params = np.array(all_params)

from sklearn.decomposition import PCA

# Instantiate PCA with 2 components
pca = PCA(n_components=2)

# Fit and transform the data
pca_transformed_params = pca.fit_transform(all_params)

plt.figure(figsize=(10, 8))
ax = plt.gca()

color_map = plt.cm.get_cmap('viridis', len(molecules))

current_idx = 0
for i, (name, symbols, coordinates) in enumerate(molecules):
    num_me_iters = len(all_param_histories[name]['ME'])
    num_adam_iters = len(all_param_histories[name]['Adam'])

    # Extract ME trajectory
    me_pca_data = pca_transformed_params[current_idx : current_idx + num_me_iters]
    current_idx += num_me_iters

    # Extract Adam trajectory
    adam_pca_data = pca_transformed_params[current_idx : current_idx + num_adam_iters]
    current_idx += num_adam_iters

    # Plot ME trajectory
    ax.plot(me_pca_data[:, 0], me_pca_data[:, 1], marker='o', linestyle='-', color=color_map(i), label=f"{name} - ME")

    # Plot Adam trajectory
    ax.plot(adam_pca_data[:, 0], adam_pca_data[:, 1], marker='x', linestyle='--', color=color_map(i), label=f"{name} - Adam")


plt.title("VQE Parameter Trajectories (PCA-reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
