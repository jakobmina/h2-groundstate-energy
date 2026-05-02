# H2 Ground State Energy Calculation

This project is a submission for the `h2-groundstate-energy` competition on Aqora, aimed at calculating the ground state energy of a hydrogen molecule using the Variational Quantum Eigensolver (VQE) algorithm implemented with PennyLane.

## Structure

The codebase is organized within the `src/submission` directory and consists of:

- `__init__.py`: This file orchestrates the calculation of the ground state energy by invoking the VQE algorithm.
- `vqe_pennylane.py`: Contains the implementation of the VQE algorithm, detailing the quantum circuit preparation, optimization process, and computation of the ground state energy.

## Installation

Use the Aqora CLI's install command to set up your environment, which installs all necessary dependencies automatically:

```bash
aqora install
```

## Testing

Testing your submission is straightforward with the `aqora test` command, which replicates the competition environment:

```bash
aqora test
```

This command ensures that your submission is compatible with the competition's evaluation process and meets the expected performance criteria.

## Uploading Results

Once you are satisfied with your submission's performance, you can upload your results to the Aqora platform using the `aqora upload` command. This step is crucial for officially submitting your work for evaluation in the competition:

```bash
aqora upload
```

After a successful submission, you will appear on the leaderboard of the `h2-groundstate-energy` competition on the Aqora platform.