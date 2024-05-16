# The Quantum Natural Gradient optimizer

Qadence-libs provides a set of optimizers based on quantum information tools, in particular based on the Quantum Fisher Information[^1] (QFI). The Quantum Natural Gradient [^2] (QNG) is a gradient-based optimizer which uses the QFI matrix to better navigate the optimizer's descent to the minimum. The parameter update rule for the QNG optimizer is written as:

$$
\theta_{t+1} = \theta_t - \eta g^{-1}(\theta_t)\nabla \mathcal{L}(\theta_t)
$$

where $g(\theta)$ is the Fubiny-Study metric tensor (aka Quantum Geometric Tensor), which is equivalent to the Quantum Fisher Information matrix $F(\theta)$ up to a constant factor $F(\theta)= 4 g(\theta)$. The Quantum Fisher Information can be written as the Hessian of the fidelity of a quantum state:

$$
  F_{i j}(\theta)=-\left.2 \frac{\partial}{\partial \theta_i} \frac{\partial}{\partial \theta_j}\left|\left\langle\psi\left(\theta^{\prime}\right) \mid \psi(\theta)\right\rangle\right|^2\right|_{{\theta}^{\prime}=\theta}
$$

However, computing the above expression is a costly operation scaling quadratically with the number of parameters in the variational quantum circuit. It is thus usual to use approximate methods when dealing with the QFI matrix. Qadence provides a SPSA-based implementation of the Quantum Natural Gradient[^3]. The [SPSA](https://www.jhuapl.edu/spsa/) (Simultaneous Perturbation Stochastic Approximation) algorithm is a well known gradient-based algorithm based on finite differences. QNG-SPSA constructs an iterative approximation to the QFI matrix with a constant number of circuit evaluations that does not scale with the number of parameters. Although the SPSA algorithm outputs a rough approximation of the QFI matrix, the QNG-SPSA has been proven to work well while being a very efficient method due to the constant overhead in circuit evaluations (only 6 extra evaluations per iteration).

In this tutorial, we use the QNG and QNG-SPSA optimizers with the Quantum Circuit Learning algorithm, a variational quantum algorithm which uses Quantum Neural Networks as universal function approximators.

```python exec="on" source="material-block" html="1" session="main"
import torch
from torch.utils.data import random_split
import random
import matplotlib.pyplot as plt

from qadence import QuantumCircuit, QNN, FeatureParameter
from qadence import kron, tag, hea, RX, Z, hamiltonian_factory

from qadence_libs.qinfo_tools import QuantumNaturalGradient
from qadence_libs.types import FisherApproximation
```

First, we prepare the Quantum Circuit Learning data. In this case we will fit a simple one-dimensional sin($x$) function:
```python exec="on" source="material-block" html="1" session="main"
# Ensure reproducibility
seed = 0
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
n_qubits = 3

# create a simple feature map to encode the input data
feature_param = FeatureParameter("phi")
feature_map = kron(RX(i, feature_param) for i in range(n_qubits))
feature_map = tag(feature_map, "feature_map")

# create a digital-analog variational ansatz using Qadence convenience constructors
ansatz = hea(n_qubits, depth=n_qubits)
ansatz = tag(ansatz, "ansatz")

# Observable
observable = hamiltonian_factory(n_qubits, detuning= Z)
```

## Optimizers

We will experiment with three different optimizers: ADAM, QNG and QNG-SPSA. To train a model with the different optimizers we will create a `QuantumModel` and reset the values of their variational parameters before each training loop so that all of them have the same starting point. 

```python exec="on" source="material-block" html="1" session="main"
# Build circuit and model
circuit = QuantumCircuit(n_qubits, feature_map, ansatz)
model = QNN(circuit, [observable])

# Loss function
mse_loss = torch.nn.MSELoss()

# Initial parameter values
initial_params = torch.rand(model.num_vparams)
```

We can now train the model with the different corresponding optimizers:
### ADAM
```python exec="on" source="material-block" html="1" session="main"
# Train with ADAM
n_epochs_adam = 20
lr_adam = 0.1

model.reset_vparams(initial_params)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)  

loss_adam = []
for i in range(n_epochs_adam):
    optimizer.zero_grad()
    loss = mse_loss(model(values=x_train).squeeze(), y_train.squeeze())
    loss_adam.append(float(loss))
    loss.backward()
    optimizer.step()
```

### QNG
```python exec="on" source="material-block" html="1" session="main"
# Train with QNG
n_epochs_qng = 20
lr_qng = 0.1

model.reset_vparams(initial_params)
optimizer = QuantumNaturalGradient(
    model.parameters(),
    lr=lr_qng,
    approximation=FisherApproximation.EXACT,
    model=model,
    beta=0.1,
)

loss_qng = []
for i in range(n_epochs_qng):
    optimizer.zero_grad()
    loss = mse_loss(model(values=x_train).squeeze(), y_train.squeeze())
    loss_qng.append(float(loss))
    loss.backward()
    optimizer.step()
```

### QNG-SPSA
```python exec="on" source="material-block" html="1" session="main"
# Train with QNG-SPSA
n_epochs_qng_spsa = 20
lr_qng_spsa = 0.01

model.reset_vparams(initial_params)
optimizer = QuantumNaturalGradient(
    model.parameters(),
    lr=lr_qng_spsa,
    approximation=FisherApproximation.SPSA,
    model=model,
    beta=0.1,
    epsilon=0.01,
)

loss_qng_spsa = []
for i in range(n_epochs_qng_spsa):
    optimizer.zero_grad()
    loss = mse_loss(model(values=x_train).squeeze(), y_train.squeeze())
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
[^1]: [Meyer J.](https://quantum-journal.org/papers/q-2021-09-09-539/) Fisher Information in Noisy Intermediate-Scale Quantum Applications
[^2]: [Stokes et al.](https://quantum-journal.org/papers/q-2020-05-25-269/) - Quantum Natural Gradient
[^3]: [Gacon et al.](https://arxiv.org/abs/2103.09232) - Simultaneous Perturbation Stochastic Approximation of the Quantum Fisher Information
