#コラボラトリー設定
!pip install qulacs pyscf openfermion openfermionpyscf

!pip3 install wurlitzer
%load_ext wurlitzer
!pip install numpy==1.24

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import minimize

# PySCF関連のインポート
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner

# Qulacsのインポート
from qulacs import Observable, QuantumState, QuantumCircuit
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.gate import CZ, RY, RZ, merge

# PySCF によってハミルトニアンを計算
def setup_molecule_and_hamiltonian():
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    distance = 0.977
    geometry = [["H", [0,0,0]],["H", [0,0,distance]]]
    description = "tmp"
    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
    n_qubit = molecule.n_qubits
    n_electron = molecule.n_electrons
    fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)

    # ハミルトニアンを qulacs ハミルトニアンに変換
    qulacs_hamiltonian = create_observable_from_openfermion_text(str(jw_hamiltonian))


    return molecule, qulacs_hamiltonian, n_qubit, n_electron

# Qulacs 上で量子回路を構成　ansatz を構成
def he_ansatz_circuit(n_qubit, depth, theta_list):
    """he_ansatz_circuit
    Returns hardware efficient ansatz circuit.

    Args:
        n_qubit (:class:`int`):
            the number of qubit used (equivalent to the number of fermionic modes)
        depth (:class:`int`):
            depth of the circuit.
        theta_list (:class:`numpy.ndarray`):
            rotation angles.
    Returns:
        :class:`qulacs.QuantumCircuit`
    """
    circuit = QuantumCircuit(n_qubit)
    for d in range(depth):
        for i in range(n_qubit):
            circuit.add_gate(merge(RY(i, theta_list[2*i+2*n_qubit*d]), RZ(i, theta_list[2*i+1+2*n_qubit*d])))
        for i in range(n_qubit//2):
            circuit.add_gate(CZ(2*i, 2*i+1))
        for i in range(n_qubit//2-1):
            circuit.add_gate(CZ(2*i+1, 2*i+2))
    for i in range(n_qubit):
        circuit.add_gate(merge(RY(i, theta_list[2*i+2*n_qubit*depth]), RZ(i, theta_list[2*i+1+2*n_qubit*depth])))


    return circuit

# VQE のコスト関数と勾配計算関数を定義
def create_cost_and_gradient_functions(n_qubit, depth, qulacs_hamiltonian):
    def cost(theta_list):
        state = QuantumState(n_qubit) #|00000> を準備
        circuit = he_ansatz_circuit(n_qubit, depth, theta_list) #量子回路を構成
        circuit.update_quantum_state(state) #量子回路を状態に作用
        return qulacs_hamiltonian.get_expectation_value(state) #ハミルトニアンの期待値を計算

    def compute_gradient(theta_list, eps=1e-4):
        """
        パラメータセットの勾配を計算する関数
        中央差分法を使用
        """
        gradient = np.zeros_like(theta_list)
        for i in range(len(theta_list)):
            theta_plus = theta_list.copy()
            theta_plus[i] += eps


            theta_minus = theta_list.copy()
            theta_minus[i] -= eps


            gradient[i] = (cost(theta_plus) - cost(theta_minus)) / (2 * eps)


        return gradient


    return cost, compute_gradient

# Velocity Verlet最適化アルゴリズムを実装
def velocity_verlet_optimizer(cost_func, gradient_func, init_params, dt=0.01, mass=1.0, damping=0.6, max_iter=50, tol=1e-10):
    """
    Velocity Verlet最適化アルゴリズム

    Args:
        cost_func: コスト関数
        gradient_func: 勾配計算関数
        init_params: 初期パラメータ
        dt: 時間ステップ
        mass: 質量パラメータ
        damping: 減衰係数（0から1の間）
        max_iter: 最大反復回数
        tol: 収束判定閾値


    Returns:
        最適化されたパラメータ、コスト履歴
    """
    params = init_params.copy()
    velocity = np.zeros_like(params)  # 速度を初期化
    cost_history = [cost_func(params)]


    for iter in range(max_iter):
        # 現在の勾配を計算（力に相当）
        force = -gradient_func(params)  # 勾配の符号を反転して力を得る


        # Velocity Verlet法によるアップデート
        # 1. 速度を半ステップ更新
        velocity += 10000 * dt * force / mass


        # 2. 位置を全ステップ更新
        params += dt * velocity


        # 3. 新しい位置での力を計算
        force_new = -gradient_func(params)


        # 4. 速度を残りの半ステップ更新
        velocity += 100 * dt * force_new / mass


        # 5. 減衰を適用（オプション）
        velocity *= damping


        # コスト値を計算して履歴に追加
        current_cost = cost_func(params)
        cost_history.append(current_cost)


        # 収束判定
        if iter > 0 and abs(cost_history[-1] - cost_history[-2]) < tol:
            print(f"Velocity Verlet converged at iteration {iter}")
            # 履歴を最大反復回数まで埋める
            cost_history.extend([current_cost] * (max_iter - iter))
            break


        if iter % 5 == 0:
            print(f"Velocity Verlet - Iteration {iter}, Cost: {current_cost}")


    return params, cost_history

# SciPyオプティマイザ用のラッパー関数（履歴追跡付き）
def run_scipy_optimizer(method, init_params, cost_func, gradient_func, max_iter):
    """
    SciPyオプティマイザを履歴追跡付きで実行する関数

    Args:
        method: 最適化手法名
        init_params: 初期パラメータ
        cost_func: コスト関数
        gradient_func: 勾配計算関数
        max_iter: 最大反復回数


    Returns:
        最適化されたパラメータ、コスト履歴、最適化時間
    """
    history = [cost_func(init_params)]


    def callback(xk):
        current_cost = cost_func(xk)
        history.append(current_cost)
        if len(history) % 5 == 1:  # 5回ごとに表示 (最初の1回も含む)
            print(f"{method} - Iteration {len(history)-1}, Cost: {current_cost:.8f}")


    start_time = time.time()


    if method == "COBYLA":
        options = {'maxiter': max_iter, 'rhobeg': 0.1}
        result = minimize(cost_func, init_params, method=method,
                         options=options,
                         callback=callback)
    else:  # L-BFGS-B と SLSQP の場合
        options = {'maxiter': max_iter}
        result = minimize(cost_func, init_params, method=method,
                         jac=gradient_func,
                         options=options,
                         callback=callback)


    end_time = time.time()
    optimization_time = end_time - start_time


    # 履歴の長さを max_iter+1 に揃える
    if len(history) < max_iter + 1:
        # 最後の値で埋める
        history.extend([history[-1]] * (max_iter + 1 - len(history)))
    elif len(history) > max_iter + 1:
        # 切り詰める
        history = history[:max_iter + 1]


    return result.x, history, optimization_time

# 複数の最適化手法を比較する関数
def compare_optimizers(cost_func, gradient_func, n_qubit, depth, fci_energy, max_iter=40):
    """
    VQEの複数最適化手法を比較する関数

    Args:
        cost_func: コスト関数
        gradient_func: 勾配計算関数
        n_qubit: 量子ビット数
        depth: 回路の深さ
        fci_energy: 参照FCI energy
        max_iter: 最大反復回数


    Returns:
        結果とパフォーマンス指標を含む辞書
    """
    # 全ての手法で同じ初期パラメータを生成
    np.random.seed(18)
    init_params = np.random.random(2*n_qubit*(depth+1))*1e-1


    results = {}


    # Velocity Verlet法を実行
    print("\n=== Velocity Verlet法を実行中 ===")
    start_time = time.time()
    vv_params, vv_history = velocity_verlet_optimizer(
        cost_func=cost_func,
        gradient_func=gradient_func,
        init_params=init_params,
        dt=0.01,
        mass=0.8,
        damping=0.68,
        max_iter=max_iter,
        tol=1e-10
    )
    vv_time = time.time() - start_time
    vv_final_energy = cost_func(vv_params)
    vv_error = abs(vv_final_energy - fci_energy)


    results["Velocity Verlet"] = {
        "params": vv_params,
        "history": vv_history,
        "time": vv_time,
        "final_energy": vv_final_energy,
        "error": vv_error
    }


    # 他の最適化手法を実行
    methods = ["COBYLA", "L-BFGS-B", "SLSQP"]
    for method in methods:
        print(f"\n=== {method}法を実行中 ===")
        opt_params, opt_history, opt_time = run_scipy_optimizer(
            method,
            init_params,
            cost_func,
            gradient_func,
            max_iter
        )
        opt_final_energy = cost_func(opt_params)
        opt_error = abs(opt_final_energy - fci_energy)


        results[method] = {
            "params": opt_params,
            "history": opt_history,
            "time": opt_time,
            "final_energy": opt_final_energy,
            "error": opt_error
        }


    return results

# 結果の可視化関数
def visualize_results(results, fci_energy):
    """
    最適化手法の比較結果を可視化する関数

    Args:
        results: 最適化結果を含む辞書
        fci_energy: 参照FCI energy
    """
    # エネルギー収束のプロット
    plt.figure(figsize=(12, 8))
    colors = {
        "Velocity Verlet": "#636EFA",
        "COBYLA": "#FFB000",
        "L-BFGS-B": "#00CC96",
        "SLSQP": "#EF553B"
    }


    for method, res in results.items():
        plt.plot(res["history"], label=method, color=colors[method], linewidth=2)


    plt.axhline(y=fci_energy, color='black', linestyle='--', label='FCI Energy')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Energy Value", fontsize=14)
    plt.title("VQE Optimization Convergence Comparison", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vqe_convergence_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


    # 誤差収束のプロット（対数スケール）
    plt.figure(figsize=(12, 8))
    for method, res in results.items():
        errors = [abs(e - fci_energy) for e in res["history"]]
        plt.semilogy(errors, label=method, color=colors[method], linewidth=2)


    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Energy Error (Log Scale)", fontsize=14)
    plt.title("VQE Error Convergence Comparison", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vqe_error_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


    # パフォーマンス指標表の作成
    data = {
        "最適化手法": [],
        "最終エネルギー値": [],
        "誤差": [],
        "計算時間 (秒)": [],
        "収束率": []
    }


    for method, res in results.items():
        data["最適化手法"].append(method)
        data["最終エネルギー値"].append(f"{res['final_energy']:.10f}")
        data["誤差"].append(f"{res['error']:.10e}")
        data["計算時間 (秒)"].append(f"{res['time']:.2f}")


        # 収束率の計算（最終誤差減少の90%に達するまでの反復回数）
        errors = [abs(e - fci_energy) for e in res["history"]]
        initial_error = errors[0]
        final_error = errors[-1]
        target_error = initial_error - 0.9 * (initial_error - final_error)


        for i, error in enumerate(errors):
            if error <= target_error:
                conv_rate = i
                break
        else:
            conv_rate = len(errors) - 1


        data["収束率"].append(f"{conv_rate}")


    df = pd.DataFrame(data)
    print("\nパフォーマンス比較:")
    print(df.to_string(index=False))


    # 計算時間と最終誤差のプロット
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    times = [results[m]["time"] for m in methods]
    errors = [results[m]["error"] for m in methods]


    fig, ax1 = plt.subplots(figsize=(12, 8))


    x = np.arange(len(methods))
    width = 0.35


    ax1.bar(x - width/2, times, width, label='Computation Time (s)', color='#4C78A8')
    ax1.set_ylabel('Time (s)', fontsize=14)
    ax1.set_xlabel('Optimization Method', fontsize=14)


    ax2 = ax1.twinx()
    ax2.bar(x + width/2, errors, width, label='Final Error', color='#D62728')
    ax2.set_ylabel('Error (Hartree)', fontsize=14, color='red')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.yaxis.set_major_formatter(ScalarFormatter())


    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')


    plt.title('Computation Time and Final Error by Optimization Method', fontsize=16)
    plt.tight_layout()
    plt.savefig("vqe_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


    # 結果をCSVファイルに保存
    df.to_csv('vqe_optimizer_performance.csv', index=False)


    return df

# メイン関数
def main():
    # 分子とハミルトニアンのセットアップ
    print("分子とハミルトニアンをセットアップ中...")
    molecule, qulacs_hamiltonian, n_qubit, n_electron = setup_molecule_and_hamiltonian()
    print(f"量子ビット数: {n_qubit}")
    print(f"電子数: {n_electron}")

    # 回路の深さを設定
    depth = n_qubit


    # コスト関数と勾配関数を作成
    cost, compute_gradient = create_cost_and_gradient_functions(n_qubit, depth, qulacs_hamiltonian)


    # 最適化手法の比較を実行
    max_iterations = 40  # Velocity Verlet法と同じ反復回数を選択
    print(f"\n複数の最適化手法を比較（最大{max_iterations}回反復）...")
    results = compare_optimizers(cost, compute_gradient, n_qubit, depth, molecule.fci_energy, max_iter=max_iterations)


    # 結果の可視化と分析
    print("\n結果を可視化中...")
    performance_df = visualize_results(results, molecule.fci_energy)


    # FCI energyとの比較を表示
    print(f"\nFCI energy: {molecule.fci_energy}")
    for method, res in results.items():
        print(f"{method} final energy: {res['final_energy']}, error: {res['error']}")


    return results, performance_df

if __name__ == "__main__":
    # 乱数シードを固定して再現性を確保
    np.random.seed(20)

    # メイン処理を実行
    results, performance_df = main()
