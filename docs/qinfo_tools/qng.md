# The Quantum Natural Gradient optimizer

Qadence-libs provides a set of optimizers based on quantum information tools, in particular based on the [Quantum Fisher Information](https://en.wikipedia.org/wiki/Quantum_Fisher_information). The Quantum Natural Gradient[^1] is a gradient-based optimizer which uses the Quantum Fisher Information matrix to better navigate the optimizer's descent to the minimum. The parameter update rule for the QNG optimizer is written as:

$$
\theta_{t+1} = \theta_t - \eta g^{-1}(\theta_t)\nabla \mathcal{L}(\theta_t)
$$

where $g(\theta)$ is the Fubiny-Study metric tensor (aka Quantum Geometric Tensor), which is equivalent to the Quantum Fisher Information matrix $F(\theta)$ up to a constant factor $F(\theta)= 4 g(\theta)$. The Quantum Fisher Information can be written as the Hessian of the fidelity of a quantum state:

$$
  F_{i j}(\theta)=-\left.2 \frac{\partial}{\partial \theta_i} \frac{\partial}{\partial \theta_j}\left|\left\langle\psi\left(\theta^{\prime}\right) \mid \psi(\theta)\right\rangle\right|^2\right|_{{\theta}^{\prime}=\theta}
$$

However, computing the above expression is a costly operation scaling quadratically with the number of parameters in the variational quantum circuit. It is thus usual to use approximate methods when dealing with the QFI matrix. Qadence provides a SPSA-based implementation of the Quantum Natural Gradient[^2]. The [SPSA](https://www.jhuapl.edu/spsa/) (Simultaneous Perturbation Stochastic Approximation) algorithm is a well known gradient-based algorithm based on finite differences. QNG-SPSA constructs an iterative approximation to the QFI matrix with a constant number of circuit evaluations that does not scale with the number of parameters. Although the SPSA algorithm outputs a rough approximation of the QFI matrix, the QNG-SPSA has been proven to work well while being a very efficient method due to the constant overhead in circuit evaluations (only 6 extra evalutions per iteration).

In this tutorial we use the QNG and QNG-SPSA optimizers with the Quantum Circuit Learning algorithm, a variational quantum algorithm which uses Quantum Neural Networks as universal function approximators. 

```python exec="on" source="material-block" html="1" session="main"
import torch
from torch.utils.data import random_split
import random
import matplotlib.pyplot as plt

import qadence as qd
from qadence_libs.qinfo_tools import QNG, QNG_SPSA
```

First, we prepare the Quantum Circuit Learning data. In this case we will fit a simple one-dimensional sin($x$) function:
```python exec="on" source="material-block" html="1" session="main"
# Ensure reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)

# Create dataset
def qcl_training_data(
    domain: tuple = (0, 2 * torch.pi), n_points: int = 200
) -> tuple[torch.Tensor, torch.Tensor]:
    start, end = domain

    x_rand, _ = torch.sort(torch.DoubleTensor(n_points).uniform_(start, end))
    y_rand = torch.sin(x_rand)

    return x_rand, y_rand


x, y = qcl_training_data()

# random train/test split of the dataset
train_subset, test_subset = random_split(x, [0.75, 0.25])
train_ind = sorted(train_subset.indices)
test_ind = sorted(test_subset.indices)

x_train, y_train = x[train_ind], y[train_ind]
x_test, y_test = x[test_ind], y[test_ind]
```

We now create the base Quantum Circuit that we will use with all the optimizers:
```python exec="on" source="material-block" html="1" session="main"
n_qubits = 4

# create a simple feature map to encode the input data
feature_param = qd.FeatureParameter("phi")
feature_map = qd.kron(qd.RX(i, feature_param) for i in range(n_qubits))
feature_map = qd.tag(feature_map, "feature_map")

# create a digital-analog variational ansatz using Qadence convenience constructors
ansatz = qd.hea(n_qubits, depth=n_qubits)
ansatz = qd.tag(ansatz, "ansatz")

# Observable
observable = qd.hamiltonian_factory(n_qubits, detuning=qd.Z)
```

## Optimizers

We will experiment with three different optimizers: ADAM, QNG and QNG-SPSA. For each of them we create a new instance of the same quantum model to benchmark the optimizers indepently under the same conditions.

```python exec="on" source="material-block" html="1" session="main"
# ADAM
circuit_adam = qd.QuantumCircuit(n_qubits, feature_map, ansatz)
model_adam = qd.QNN(circuit_adam, [observable])

# QNG
circuit_qng = qd.QuantumCircuit(n_qubits, feature_map, ansatz)
model_qng = qd.QNN(circuit_qng, [observable])
circ_params_qng = [param for param in model_qng.parameters() if param.requires_grad]

# QNG-SPSA
circuit_qng_spsa = qd.QuantumCircuit(n_qubits, feature_map, ansatz)
model_qng_spsa = qd.QNN(circuit_qng_spsa, [observable])
circ_params_qng_spsa = [param for param in model_qng_spsa.parameters() if param.requires_grad]
```
 
We can now train each of the models with the corresponding optimizer:

### ADAM 
```python exec="on" source="material-block" html="1" session="main"
# Train with ADAM
n_epochs_adam = 20
lr_adam = 0.1
mse_loss = torch.nn.MSELoss()  # standard PyTorch loss function
optimizer = torch.optim.Adam(model_adam.parameters(), lr=lr_adam)  # standard PyTorch Adam optimizer
loss_adam = []
for i in range(n_epochs_adam):
    optimizer.zero_grad()
    loss = mse_loss(model_adam(values=x_train).squeeze(), y_train.squeeze())
    loss_adam.append(float(loss))
    loss.backward()
    optimizer.step()
```

### QNG
```python exec="on" source="material-block" html="1" session="main"
# Train with QNG
n_epochs_qng = 20
lr_qng = 0.1
mse_loss = torch.nn.MSELoss()  # standard PyTorch loss function
optimizer = QNG(circ_params_qng, lr=lr_qng, circuit=circuit_qng, beta=0.1)
loss_qng = []
for i in range(n_epochs_qng):
    optimizer.zero_grad()
    loss = mse_loss(model_qng(values=x_train).squeeze(), y_train.squeeze())
    loss_qng.append(float(loss))
    loss.backward()
    optimizer.step()
```

### QNG-SPSA

```python exec="on" source="material-block" html="1" session="main"
# Train with QNG-SPSA
n_epochs_qng_spsa = 20
lr_qng_spsa = 0.01
mse_loss = torch.nn.MSELoss()  # standard PyTorch loss function
optimizer = QNG_SPSA(
    circ_params_qng_spsa,
    lr=lr_qng_spsa,
    circuit=circuit_qng_spsa,
    epsilon=0.01,
    beta=0.1,
)

loss_qng_spsa = []
for i in range(n_epochs_qng_spsa):
    optimizer.zero_grad()
    loss = mse_loss(model_qng_spsa(values=x_train).squeeze(), y_train.squeeze())
    loss_qng_spsa.append(float(loss))
    loss.backward()
    optimizer.step()

```

## Plotting

We now plot the losses corresponding to each of the optimizers:
```python exec="on" source="material-block" html="1" session="main"
# Plot losses
fig, _ = plt.subplots()
plt.plot(range(n_epochs_adam), loss_adam, label="Adam optimizer")
plt.plot(range(n_epochs_qng), loss_qng, label="QNG optimizer")
plt.plot(range(n_epochs_qng_spsa), loss_qng_spsa, label="QNG-SPSA optimizer")
plt.legend()
plt.xlabel("Training epochs")
plt.ylabel("Loss")

from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

## References
[^1]: [Stokes et al.](https://quantum-journal.org/papers/q-2020-05-25-269/) - Quantum Natural Gradient
[^2]: [Gacon et al.](https://arxiv.org/abs/2103.09232) - Simultaneous Perturbation Stochastic Approximation of the Quantum Fisher Information