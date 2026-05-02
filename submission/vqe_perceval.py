import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from scipy.sparse import linalg
import perceval as pcvl
from openfermionpyscf import generate_molecular_hamiltonian
from openfermion.chem.molecular_data import MolecularData
from openfermion.transforms.opconversions.conversions import get_fermion_operator
import symmetry_conserving_bravyi_kitaev
import get_sparse_operator


class VQESolver:
    def __init__(self, input_str, basis='sto-3g', multiplicity=1, shots=20000):
        self.input = input_str
        self.basis = basis
        self.multiplicity = multiplicity
        self.shots = shots
        self.loss_values = []
        self.iteration = 0
        self.params = None
        self.measured_qs = []
        self.measurement_bases = ['ZZ', 'XX']
        self.groups = None
        self.h_names = None
        self.h_weights = None
        self.identity_offset = 0.0
        self.real_energy = None
        self.num_qubits = 2
        self.states = {
            '00': pcvl.BasicState([0, 1, 0, 1, 0, 0]),
            '01': pcvl.BasicState([0, 1, 0, 0, 1, 0]),
            '10': pcvl.BasicState([0, 0, 1, 1, 0, 0]),
            '11': pcvl.BasicState([0, 0, 1, 0, 1, 0])
        }

    def parse_input(self, input_str):
        lines = input_str.strip().split('\n')
        num_atoms = int(lines[0].strip())
        geometry = []
        for line in lines[2:2 + num_atoms]:
            parts = line.split()
            atom = parts[0]
            coords = tuple(float(x) for x in parts[1:])
            geometry.append((atom, coords))
        return geometry

    def set_hamiltonian(self, geometry=None):
        if geometry is None:
            geometry = self.parse_input(self.input)

        mol_ham = generate_molecular_hamiltonian(geometry, self.basis, self.multiplicity)
        ferm_op = get_fermion_operator(mol_ham)
        qubit_ham = symmetry_conserving_bravyi_kitaev(ferm_op, 4, 2)

        h_weights = []
        h_names = []
        for term, coeff in qubit_ham.terms.items():
            if term:
                pauli_string = ''.join([op[1] for op in term])
                qubit_string = 'II'
                for op in term:
                    qubit_string = qubit_string[:op[0]] + op[1] + qubit_string[op[0] + 1:]
                h_weights.append(coeff)
                h_names.append(qubit_string)
            else:
                self.identity_offset = coeff

        self.h_names = h_names
        self.h_weights = h_weights
        self.groups = self.group_hamiltonian_terms(h_names, h_weights)

        sparse_ham = get_sparse_operator(qubit_ham)
        eigs, _ = linalg.eigsh(sparse_ham, k=1, which='SA')
        self.real_energy = float(eigs[0])

    def exact_eigensolver(self, h_names, h_weights, identity_offset):
        n = 4
        H = np.zeros((n, n), dtype=np.complex128)
        pauli_matrices = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }

        def kron_pauli(term):
            matrices = [pauli_matrices[p] for p in term]
            return np.kron(matrices[0], matrices[1])

        for weight, name in zip(h_weights, h_names):
            H += weight * kron_pauli(name)

        H += identity_offset * np.eye(n)
        eigenvalues, _ = eigh(H)
        return float(np.min(eigenvalues))

    def group_hamiltonian_terms(self, h_names, h_weights):
        ZZ_group = []
        XX_group = []
        for name, weight in zip(h_names, h_weights):
            if 'X' in name or 'Y' in name:
                XX_group.append((weight, name))
            else:
                ZZ_group.append((weight, name))
        return [ZZ_group, XX_group]

    def H(self):
        return pcvl.BS.H()

    def RY(self, angle):
        return pcvl.BS.Ry(theta=angle)

    def RX(self, angle):
        return pcvl.BS.Rx(theta=angle)

    def RZ(self, angle):
        # RZ in dual-rail encoding is a phase shift between the two modes
        circ = pcvl.Circuit(2)
        circ.add(0, pcvl.PS(angle))
        return circ

    def RY_basis_change(self):
        # Rotate Y basis to Z basis: RX(pi/2)
        return pcvl.BS.Rx(theta=np.pi/2)

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

    def append_Hadamard(self, idx: int, circ: pcvl.Circuit) -> None:
        if idx == 0:
            circ.add((1, 2), self.H())
        elif idx == 1:
            circ.add((3, 4), self.H())

    def rotate_measurements(self, circuit: pcvl.Circuit, pauli_string: str):
        new_circuit = circuit.copy()
        for i, pauli_op in enumerate(pauli_string):
            if pauli_op == 'X':
                self.append_Hadamard(i, new_circuit)
            elif pauli_op == 'Y':
                new_circuit.add((1 if i == 0 else 3, 2 if i == 0 else 4), self.RY_basis_change())
        return new_circuit

    def measured_qubits(self, pauli_string: str):
        return [i for i, pauli_op in enumerate(pauli_string) if pauli_op in {'X', 'Y', 'Z'}]

    def minimize_loss(self, params):
        params = np.asarray(params, dtype=float)
        params = np.mod(params, 2 * np.pi)
        averages = []

        for basis_index, basis in enumerate(self.measurement_bases):
            processor = pcvl.Processor('SLOS')
            new_circ = self.rotate_measurements(self.Anzats(params), basis)
            processor.set_circuit(new_circ)
            processor.with_input(self.states['00'])
            sampler = pcvl.algorithm.Sampler(processor)
            shot_num = self.shots
            remote_job = sampler.sample_count(shot_num)

            output_dict = {}
            total_counts = 0
            for a in range(len(self.states)):
                key = str(np.binary_repr(a, self.num_qubits))
                count = remote_job['results'].get(self.states[key], 0)
                output_dict[key] = count
                total_counts += count

            if total_counts == 0:
                return 0.0

            for key in output_dict:
                output_dict[key] /= total_counts

            avg = 0.0
            for weight, term in self.groups[basis_index]:
                for output_state in self.states:
                    parity = sum(int(output_state[i]) for i in self.measured_qubits(term))
                    if parity % 2 == 1:
                        avg -= output_dict[output_state] * float(weight)
                    else:
                        avg += output_dict[output_state] * float(weight)
            averages.append(avg)

        loss = float(sum(averages) + self.identity_offset)
        self.loss_values.append(loss)
        self.iteration += 1
        return loss

    def optimize(self, method='cobyla', options=None):
        if options is None:
            options = {'tol': 1e-4, 'maxiter': 200}
        init_param = [2 * np.pi * np.random.random() for _ in range(12)]
        result = minimize(self.minimize_loss, init_param, method=method, options=options)
        self.params = result.x
        return result


def run_sample():
    default_input = '''2
Sample H2 molecule
H 0.3710 0.0 0.0
H -0.3710 0.0 0.0'''
    solver = VQESolver(default_input, shots=10000)
    geometry = solver.parse_input(default_input)
    solver.set_hamiltonian(geometry)
    result = solver.optimize()
    print('Optimized Energy:', float(result.fun))
    print('Exact energy:', solver.real_energy)
    return solver, result


if __name__ == '__main__':
    run_sample()
