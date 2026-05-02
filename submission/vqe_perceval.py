import numpy as np
from scipy.linalg import eigh
from scipy.sparse import linalg
import perceval as pcvl
from openfermionpyscf import generate_molecular_hamiltonian
from openfermion.transforms.opconversions.conversions import get_fermion_operator
from openfermion.transforms.opconversions.remove_symmetry_qubits import symmetry_conserving_bravyi_kitaev
from openfermion.linalg.sparse_tools import get_sparse_operator

# =============================================================================
# CONSTANTES H7
# =============================================================================

_phi = (1 + np.sqrt(5)) / 2
_O_n = abs(np.cos(np.pi * _phi * 1))   # O_n_integrity ≈ 0.36237489
_DRIFT_canonical = 0.48796433           # DRIFT Z₇* de referencia

_W = 60

def _bar(c="═"): return c * _W
def _row(label, val):
    val_str = f"{val:.8f}" if isinstance(val, float) else str(val)
    return f"  │  {label:<28} {val_str}"


# =============================================================================
# VQE SOLVER — Perceval + H7 Metriplectic Optimizer
# =============================================================================

class VQESolver:
    def __init__(self, input_str, basis='sto-3g', multiplicity=1, shots=10000):
        self.input_str       = input_str
        self.basis           = basis
        self.multiplicity    = multiplicity
        self.shots           = shots
        self.loss_values     = []
        self.iteration       = 0
        self.params          = None
        self.measurement_bases = ['ZZ', 'XX']
        self.groups          = None
        self.h_names         = None
        self.h_weights       = None
        self.identity_offset = 0.0
        self.real_energy     = None # Para validación vs Eigensolver
        self.groups          = None # Almacenará los grupos Pauli (ZZ, XX, etc)
        self.num_qubits      = 2

        # ── H7 Metriplectic State ─────────────────────────────────────
        self.learning_rate   = 0.03
        self.covariance      = np.eye(12)
        self.base_epsilon    = 1e-4
        self.target_energy   = -1.137        # referencia H₂ STO-3G

        # Atractor H7: O_n_integrity como punto fijo metripléxico
        self.attractor       = np.full(12, _O_n)

        # Estados fotónicos dual-rail (6 modos)
        self.states = {
            '00': pcvl.BasicState([0, 1, 0, 1, 0, 0]),
            '01': pcvl.BasicState([0, 1, 0, 0, 1, 0]),
            '10': pcvl.BasicState([0, 0, 1, 1, 0, 0]),
            '11': pcvl.BasicState([0, 0, 1, 0, 1, 0]),
        }

    # ── Parsing ───────────────────────────────────────────────────────

    def parse_input(self, input_str):
        lines = input_str.strip().split('\n')
        num_atoms = int(lines[0].strip())
        geometry = []
        for line in lines[2:2 + num_atoms]:
            parts = line.split()
            atom  = parts[0]
            coords = tuple(float(x) for x in parts[1:])
            geometry.append((atom, coords))
        return geometry

    # ── Hamiltoniano ──────────────────────────────────────────────────

    def set_hamiltonian(self, geometry=None):
        if geometry is None:
            geometry = self.parse_input(self.input_str)

        mol_ham   = generate_molecular_hamiltonian(geometry, self.basis, self.multiplicity)
        ferm_op   = get_fermion_operator(mol_ham)
        qubit_ham = symmetry_conserving_bravyi_kitaev(ferm_op, 4, 2)

        h_weights, h_names = [], []
        for term, coeff in qubit_ham.terms.items():
            if term:
                qubit_string = 'II'
                for op in term:
                    qubit_string = qubit_string[:op[0]] + op[1] + qubit_string[op[0] + 1:]
                h_weights.append(coeff)
                h_names.append(qubit_string)
            else:
                self.identity_offset = coeff

        self.h_names  = h_names
        self.h_weights = h_weights
        self.groups   = self._group_hamiltonian_terms(h_names, h_weights)

        sparse_ham = get_sparse_operator(qubit_ham)
        eigs, _    = linalg.eigsh(sparse_ham, k=1, which='SA')
        self.real_energy = float(eigs[0])

        print(_bar())
        print(f"  Hamiltoniano  ·  {self.basis.upper()}  ·  SCBK")
        print(_bar('─'))
        print(_row("E exacta (eigensolver)", self.real_energy))
        print(_row("Offset identidad",       float(np.real(self.identity_offset))))
        print(_row("Términos Pauli",         len(h_names)))
        print(_row("O_n_integrity (prior)",  _O_n))
        print(_bar())

    def _group_hamiltonian_terms(self, h_names, h_weights):
        """
        Agrupa términos por base de medición.
        ZZ_group: solo Z  → medición computacional directa.
        XX_group: X o Y   → requieren rotación de base.
        """
        ZZ_group, XX_group = [], []
        for name, weight in zip(h_names, h_weights):
            if 'X' in name or 'Y' in name:
                XX_group.append((weight, name))
            else:
                ZZ_group.append((weight, name))
        return [ZZ_group, XX_group]

    def exact_eigensolver(self):
        n = 4
        H = np.zeros((n, n), dtype=np.complex128)
        pm = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]]),
        }
        for weight, name in zip(self.h_weights, self.h_names):
            H += weight * np.kron(pm[name[0]], pm[name[1]])
        H += self.identity_offset * np.eye(n)
        eigenvalues, _ = eigh(H)
        return float(np.min(eigenvalues))

    # ── Circuito Perceval ─────────────────────────────────────────────

    def H(self):              return pcvl.BS.H()
    def RY(self, a):          return pcvl.BS.Ry(theta=a)
    def RX(self, a):          return pcvl.BS.Rx(theta=a)
    def RY_basis_change(self): return pcvl.BS.Rx(theta=np.pi / 2)

    def RZ(self, angle):
        circ = pcvl.Circuit(2)
        circ.add(0, pcvl.PS(angle))
        return circ

    def Anzats(self, params):
        circ = pcvl.Circuit(6)
        circ.add(1, self.RX(params[0]))
        circ.add(3, self.RX(params[1]))
        circ.add(1, self.RZ(params[2]))
        circ.add(3, self.RZ(params[3]))
        circ.add(1, self.RX(params[4]))
        circ.add(3, self.RX(params[5]))
        circ.add((0, 1, 2, 3, 4, 5), pcvl.PERM([0, 1, 2, 3, 4, 5]))
        circ.add((0, 1), pcvl.BS())
        circ.add((2, 3), pcvl.BS())
        circ.add((4, 5), pcvl.BS())
        circ.add((0, 1, 2, 3, 4, 5), pcvl.PERM([0, 1, 2, 3, 4, 5]))
        circ.add((3, 4), pcvl.BS())
        circ.add((0, 1), pcvl.BS(pcvl.BS.r_to_theta(1 / 3)))
        circ.add((2, 3), pcvl.BS(pcvl.BS.r_to_theta(1 / 3)))
        circ.add((4, 5), pcvl.BS(pcvl.BS.r_to_theta(1 / 3)))
        circ.add((3, 4), pcvl.BS())
        circ.add((0, 1, 2, 3, 4, 5), pcvl.PERM([0, 1, 2, 3, 4, 5]))
        circ.add(1, self.RX(params[6]))
        circ.add(3, self.RX(params[7]))
        circ.add(1, self.RZ(params[8]))
        circ.add(3, self.RZ(params[9]))
        circ.add(1, self.RX(params[10]))
        circ.add(3, self.RX(params[11]))
        return circ

    def _append_Hadamard(self, idx, circ):
        if idx == 0:
            circ.add((1, 2), self.H())
        elif idx == 1:
            circ.add((3, 4), self.H())

    def _rotate_measurements(self, circuit, pauli_string):
        new_circuit = circuit.copy()
        for i, op in enumerate(pauli_string):
            if op == 'X':
                self._append_Hadamard(i, new_circuit)
            elif op == 'Y':
                new_circuit.add(
                    (1 if i == 0 else 3, 2 if i == 0 else 4),
                    self.RY_basis_change()
                )
        return new_circuit

    def _measured_qubits(self, pauli_string):
        return [i for i, op in enumerate(pauli_string) if op in {'X', 'Y', 'Z'}]

    # ── Evaluación de energía ─────────────────────────────────────────

    def compute_energy_and_entropy(self, params):
        params = np.mod(np.asarray(params, dtype=float), 2 * np.pi)
        averages      = []
        total_entropy = 0.0

        for basis_index, basis in enumerate(self.measurement_bases):
            processor = pcvl.Processor('SLOS')
            new_circ  = self._rotate_measurements(self.Anzats(params), basis)
            processor.set_circuit(new_circ)
            processor.with_input(self.states['00'])
            
            # Implementación del algoritmo de muestreo de Perceval
            sampler    = pcvl.algorithm.Sampler(processor)
            remote_job = sampler.sample_count(self.shots) # algorithm execution

            # ── Conteo con descarte explícito de modos inesperados ────
            output_dict  = {}
            total_counts = 0
            
            # Recuperamos los resultados (compatible con ResultMap y exqalibur.BSCount)
            results_map = remote_job['results']
            
            for bit_label, state_obj in self.states.items():
                # Usamos acceso por llave y comprobación de membresía porque BSCount no tiene .get()
                count = results_map[state_obj] if state_obj in results_map else 0
                output_dict[bit_label] = count
                total_counts += count

            if total_counts == 0:
                return 0.0, 0.0

            # Entropía de Shannon como proxy Von Neumann
            probs = [c / total_counts for c in output_dict.values()]
            total_entropy += -sum(p * np.log(p + 1e-15) for p in probs)

            # Normalizar
            for key in output_dict:
                output_dict[key] /= total_counts

            # Valor esperado del grupo
            avg = 0.0
            for weight, term in self.groups[basis_index]:
                for state_key in self.states:
                    parity = sum(
                        int(state_key[i])
                        for i in self._measured_qubits(term)
                    )
                    sign = -1 if parity % 2 == 1 else +1
                    avg += sign * output_dict[state_key] * float(weight)
            averages.append(avg)

        loss = float(np.real(sum(averages)) + self.identity_offset)
        return loss, total_entropy / len(self.measurement_bases)

    # ── Optimizador Metripléxico H7 ───────────────────────────────────

    def optimize(self, max_iter=100):
        """
        Bucle metripléxico H7.

        Evolución:  dθ = lr·∇H  +  ε·Σ⁻¹·(θ − O_n)
                        ↑              ↑
                    simpléctico     disipación hacia atractor H7
                    {θ, H}             [θ, S]

        La covarianza Σ acumula memoria del gradiente (vacío cuántico).
        El atractor es O_n_integrity ≈ 0.3624, no el origen.
        """
        # Inicialización con prior H7 perturbado
        if self.groups is None:
            self.set_hamiltonian()

        rng   = np.random.default_rng()
        theta = self.attractor + rng.uniform(-0.05, 0.05, 12)

        best_energy = np.inf
        best_theta  = theta.copy()

        print(f"\n{_bar()}")
        print(f"  H7 METRIPLECTIC OPTIMIZER  ·  {max_iter} iteraciones")
        print(_bar('─'))
        print(_row("Atractor O_n_integrity",  _O_n))
        print(_row("learning_rate",           self.learning_rate))
        print(_row("base_epsilon",            self.base_epsilon))
        print(_row("theta_0 ~ O_n + U[-0.1,0.1]", ""))
        print(_bar())

        for i in range(max_iter):
            energy, entropy_val = self.compute_energy_and_entropy(theta)
            self.loss_values.append(energy)

            # Tracking del mínimo real
            if energy < best_energy:
                best_energy = energy
                best_theta  = theta.copy()

            # 1. Gradiente simpléctico — diferencias finitas
            grad    = np.zeros(12)
            eps_g   = 0.05
            for j in range(12):
                t_plus      = theta.copy()
                t_plus[j]  += eps_g
                e_plus, _   = self.compute_energy_and_entropy(t_plus)
                grad[j]     = (e_plus - energy) / eps_g

            # 2. Epsilon dinámico — crece con entropía para evitar colapso
            dynamic_epsilon = self.base_epsilon + 0.1 * entropy_val

            # 3. Componente métrica — disipación hacia O_n_integrity
            try:
                cov_inv = np.linalg.inv(
                    self.covariance + dynamic_epsilon * np.eye(12)
                )
            except np.linalg.LinAlgError:
                cov_inv = np.eye(12)

            dissipation = dynamic_epsilon * np.dot(cov_inv, theta - self.attractor)

            # 4. Evolución metripléctica: {θ,H} + [θ,S]
            theta -= self.learning_rate * grad + dissipation

            # 5. Actualización covarianza — memoria del vacío
            self.covariance = 0.9 * self.covariance + 0.1 * np.outer(grad, grad)

            # ── Log cada 5 iter ───────────────────────────────────────
            if i % 5 == 0:
                gap = abs(energy - self.real_energy) if self.real_energy else float('nan')
                print(
                    f"  iter {i:03d}"
                    f"  │  E={energy:+.6f}"
                    f"  │  S={entropy_val:.4f}"
                    f"  │  ε={dynamic_epsilon:.6f}"
                    f"  │  gap={gap:.6f}"
                )

            self.iteration += 1

            # Convergencia
            if i > 5 and abs(self.loss_values[-1] - self.loss_values[-2]) < 1e-5:
                print(f"\n  [H7] Convergencia en iteración {i}.")
                break

        self.params = best_theta

        # ── Métricas de Torsión y Drift (Sistema Principal) ──────────
        delta      = abs(np.cos(np.pi * _phi * 1) - np.cos(np.pi * _phi * 6))
        drift_muestra = delta  # muestra de un solo nodo activo
        tension    = abs(best_energy - self.real_energy) if self.real_energy else float('nan')

        print(f"\n{_bar()}")
        print(f"  OPTIMIZACIÓN FINALIZADA")
        print(f"  Algoritmo: Perceval Sampler + H7 Phase Governor")
        print(_bar('─'))
        print(_row("E optimizada (best)",    best_energy))
        print(_row("E exacta (eigensolver)", self.real_energy or float('nan')))
        print(_row("Gap |E_opt - E_exact|",  tension))
        print(_bar('─'))
        print(_row("O_n_integrity",          _O_n))
        print(_row("DRIFT canónico Z₇*",     _DRIFT_canonical))
        print(_row("δ torsión (n=1)",        drift_muestra))
        print(_row("δ / DRIFT",              drift_muestra / _DRIFT_canonical))
        print(_bar())

        # Interfaz compatible con Aqora/Scipy (fun, x)
        class Result:
            def __init__(self, energy, params):
                self.fun = energy  # Energía mínima encontrada
                self.x = params    # Parámetros óptimos (theta)
                self.algorithm = "Perceval-Metriplectic-H7"

        return Result(best_energy, best_theta)


# =============================================================================
# EJECUCIÓN
# =============================================================================

def run_sample():
    default_input = '''2
Sample H2 molecule
H 0.3710 0.0 0.0
H -0.3710 0.0 0.0'''

    solver   = VQESolver(default_input, shots=10000)
    geometry = solver.parse_input(default_input)
    solver.set_hamiltonian(geometry)

    result = solver.optimize()

    print(f"\n  output  →  {float(result.fun):.8f} Ha")
    return solver, result


if __name__ == '__main__':
    run_sample()
