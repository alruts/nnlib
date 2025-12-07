# pinnlib

A JAX-based library of neural network architectures for physics-informed
acoustic applications.

## Overview

pinnlib provides a comprehensive suite of neural network architectures and
utilities specifically designed for solving acoustic wave equations using
Physics-Informed Neural Networks (PINNs). Built on top of JAX and Equinox, it
offers high-performance, differentiable implementations for both time-domain
and frequency-domain acoustic problems.

## Features

- **Neural Network Architectures**: Modified MLP, SIREN, PirateNet, and custom architectures
- **Physics-Informed Components**: Wave equation and Helmholtz equation solvers
- **Complex Number Support**: Full support for complex-valued neural networks
- **Custom Activations**: Specialized activation functions for acoustic applications
- **Loss Functions**: Comprehensive suite of PDE and data loss functions
- **Metrics**: Point-wise and aggregated error metrics
- **Feature Maps**: Periodic and random Fourier feature embeddings

## Installation

### Prerequisites

- Python 3.13+
- JAX with appropriate backend (CPU/GPU/TPU)

### Install from source

```bash
git clone <repository-url>
cd pinnlib
uv sync
```

### Dependencies

pinnlib relies on several key libraries:
- **JAX**: Automatic differentiation and XLA compilation
- **Equinox**: Neural network library compatible with JAX
- **Diffrax**: Differential equation solvers
- **Optax**: Optimization library

## Quick Start

### Basic Wave Equation PINN

```python
import jax
import jax.numpy as jnp
from pinnlib import WavePINN, ModifiedMLP

# Create a simple wave PINN
key = jax.random.PRNGKey(0)
pinn = WavePINN.create(
    arch_name="modified_mlp",
    in_size=3,  # x, y, t
    out_size=1,  # pressure
    width_size=32,
    depth=4,
    key=key
)

# Get model parameters
params = pinn.get_parameters()

# Evaluate at a point
x, y, t = 1.0, 2.0, 0.5
pressure = pinn(params, x, y, t)
residual = pinn.residual(params, x, y, t)
```

### Helmholtz Equation PINN

```python
from pinnlib import HelmholtzPINN

# Create a Helmholtz PINN for frequency domain
frequency = 1000.0  # Hz
pinn = HelmholtzPINN.create(
    arch_name="siren",
    in_size=2,  # x, y
    out_size=1,  # complex pressure
    width_size=64,
    depth=6,
    frequency=frequency,
    key=key
)

# Evaluate complex pressure
x, y = 0.5, 0.3
pressure = pinn(params, x, y)  # Complex-valued
residual = pinn.residual(params, x, y)
```

## Architecture Reference

### Available Architectures

| Architecture | Description | Use Case |
|--------------|-------------|----------|
| `modified_mlp` | MLP with learned modulators | General PINN applications |
| `mlp` | Standard MLP | Baseline comparisons |
| `siren` | Sinusoidal representation network | High-frequency details |
| `modified_siren` | SIREN with modulators | Complex acoustic fields |
| `pirate_net` | Adaptive residual network | Challenging PDE problems |

### Custom Activations

```python
from pinnlib.activations import (
    split_tanh, cardioid, rotating_cardioid,
    SinActivation, SplitSinActivation
)

# Use custom activation in architecture
pinn = WavePINN.create(
    arch_name="modified_mlp",
    activation=split_tanh,  # Complex tanh
    key=key
)
```

## Loss Functions and Training

### Computing Losses

```python
from pinnlib.losses import compute_loss, data_loss, hom_pde_loss
from pinnlib.metrics import aggregated_metrics

# Prepare data batch
batch = {
    "data": (coords, pressure_values),
    "pde": pde_coords
}

# Compute total loss
total_loss, loss_dict = compute_loss(
    params, pinn, batch,
    losses={"pde": hom_pde_loss, "data": data_loss},
    criterion=aggregated_metrics["mse"]
)
```

### Weighted Loss Training

```python
from pinnlib.losses import compute_weighted_loss, compute_weights

# Compute gradient-norm-based weights
weights = compute_weights(params, pinn, batch, losses)

# Use weighted loss
total_loss, loss_dict = compute_weighted_loss(
    params, pinn, weights, batch, losses, criterion
)
```

## Advanced Features

### Feature Embeddings

```python
from pinnlib.feature_maps import PeriodicFeatures, RandomFourierFeatures

# Add periodic features
embedding = PeriodicFeatures(sigma=1.0, L=5)
pinn = WavePINN.create(
    arch_name="modified_mlp",
    embedding=embedding,
    key=key
)
```

### Complex Number Support

```python
from pinnlib.complex_utils import split_real_and_imaginary_activation

# Apply activation to real and imaginary parts separately
activation = split_real_and_imaginary_activation(jax.nn.tanh)
```

## API Reference

### Core Classes

- `WavePINN`: Physics-informed neural network for wave equation
- `HelmholtzPINN`: Physics-informed neural network for Helmholtz equation

### Neural Networks

- `ModifiedMLP`: MLP with learned modulators
- `PirateNet`: Adaptive residual network
- `make_siren()`: Factory function for SIREN networks
- `make_modified_siren()`: Factory function for modified SIREN

### Utilities

- `apply_model()`: Apply model with parameters
- `get_parameters()`: Extract trainable parameters
- `args_to_array()`: Convert function arguments to array

## Examples

See the `example_scripts/` directory for complete examples:
- `esm_demo.py`: Basic PINN demonstration
- `train_helmholtz_piston_comsol.py`: Training with COMSOL data
- `mesh.py`: Mesh processing utilities

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add license information here]

## Citation

If you use pinnlib in your research, please cite:

```bibtex
@software{pinnlib,
  title={pinnlib: A JAX-based library for physics-informed acoustic neural networks},
  author={Sturla Njardarson},
  year={2024},
  url={https://github.com/username/pinnlib}
}
```
