# pinnlib

A library of neural network architectures for acoustics-informed applications, built on [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

pinnlib implements Physics-Informed Neural Networks (PINNs) for solving acoustic PDEs — both the time-domain wave equation and the frequency-domain Helmholtz equation. It provides everything needed to go from problem setup to trained model: network architectures, PDE residuals, loss functions, adaptive weighting, data generators, and utilities for complex-valued fields.

## Installation

```bash
uv sync
```

## Quick Start

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import pinnlib as pl
from pinnlib.data import UniformGenerator, DataPointGenerator
from pinnlib.metrics import mse

key = jr.PRNGKey(0)

# Build a PINN for the Helmholtz equation
pinn = pl.HelmholtzPINN.create(
    arch_name="modified_siren",
    embedding=None,
    frequency=1000.0,
    in_size=2,
    out_size="scalar",
    width_size=32,
    depth=3,
    dtype=pl.default_complex_dtype(),
    first_activation=pl.SplitSinActivation(30),
    activation=pl.SplitSinActivation(30),
    final_activation=pl.identity_activation,
    key=key,
)

# Separate trainable parameters from static structure
params, static = eqx.partition(pinn.model, eqx.is_array)

# Evaluate pressure at a point
p = pinn(params, x, y)

# Evaluate PDE residual at a point
r = pinn.residual(params, x, y)
```

---

## Core PINN Classes

### `WavePINN` — Time Domain

Solves the acoustic wave equation: `∇²p − (1/c²) p_tt = 0`

```python
pinn = pl.WavePINN.create(
    arch_name="modified_siren",   # architecture (see below)
    in_size=3,                     # x, y, t → 3 inputs
    out_size="scalar",
    width_size=64,
    depth=3,
    first_activation=pl.SinActivation(30.0),
    activation=pl.SinActivation(30.0),
    final_activation=pl.LearnableTanh(jnp.array(1.0)),
    key=key,
)

p = pinn(params, x, y, t)              # pressure
r = pinn.residual(params, x, y, t)    # wave equation residual
v = pinn.velocity(params, x, y, t, nx, ny)   # particle velocity in direction (nx, ny)
z = pinn.impedance(params, x, y, t, nx, ny)  # normalized acoustic impedance
```

**Physical defaults:** wave speed = 343.2 m/s, medium density = 1.2043 kg/m³. Override via `wave_speed=` and `medium_density=`.

### `HelmholtzPINN` — Frequency Domain

Solves the Helmholtz equation: `∇²p + k²p = 0`, where `k = 2πf/c`

```python
pinn = pl.HelmholtzPINN.create(
    arch_name="modified_siren",
    embedding=None,
    frequency=1000.0,              # Hz — required
    in_size=2,
    out_size="scalar",
    width_size=32,
    depth=3,
    dtype=pl.default_complex_dtype(),
    first_activation=pl.SplitSinActivation(30),
    activation=pl.SplitSinActivation(30),
    final_activation=pl.identity_activation,
    key=key,
)

p = pinn(params, x, y)               # complex pressure field
r = pinn.residual(params, x, y)      # Helmholtz residual
v = pinn.velocity(params, x, y, nx, ny)    # velocity via Euler's equation
z = pinn.impedance(params, x, y, nx, ny)   # normalized impedance
```

The Helmholtz PINN outputs complex values — use `pl.split_real_and_imaginary_loss` and `pl.split_real_and_imaginary_metric` to handle real/imaginary parts separately in losses and metrics.

### Available Architectures

Pass any of these as `arch_name`:

| Name | Description |
|------|-------------|
| `"modified_siren"` | SIREN with learned linear modulators — recommended default |
| `"siren"` | Standard sinusoidal representation network |
| `"modified_mlp"` | MLP with learned linear modulators |
| `"mlp"` | Standard Equinox MLP |
| `"pirate_net"` | Physics-informed residual adaptive network |

---

## Training Loop

The standard training loop uses `compute_weighted_loss` with adaptive gradient-norm weighting:

```python
losses = {"data": pl.data_loss, "pde": pl.hom_pde_loss}
loss_weights = {key: jnp.array(1.0) for key in losses}

optimizer = optax.adam(optax.schedules.exponential_decay(1e-3, 2000, 0.9))
opt_state = optimizer.init(params)

@eqx.filter_jit
def train_step(model, params, opt_state, weights, batch):
    (total, per_term), grads = jax.value_and_grad(
        pl.compute_weighted_loss, has_aux=True
    )(
        params,
        model=model,
        batch=batch,
        weights=weights,
        losses=losses,
        criterion=mse,
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, (total, per_term)

for step in range(n_steps):
    batch = {"data": next(data_loader), "pde": next(domain_loader)}
    params, opt_state, (loss, per_term) = train_step(pinn, params, opt_state, loss_weights, batch)

    # Update adaptive weights every N steps
    if step % 1000 == 0:
        new_weights = pl.compute_weights(params, pinn, batch, losses, criterion=mse)
        loss_weights = pl.update_weights(0.9, loss_weights, new_weights)
```

The `batch` dict must have the same keys as `losses`. `"data"` batches are `PointCloud`s; `"pde"` batches are coordinate tuples from a domain generator.

### For Complex-Valued Fields (Helmholtz)

Wrap the criterion and optimizer for complex support:

```python
criterion = pl.split_real_and_imaginary_loss(mse)
optimizer = optax.contrib.split_real_and_imaginary(soap(learning_rate))

# Conjugate gradients are needed for complex parameters
grads_conj = jax.tree.map(jnp.conj, grads)
updates, opt_state = optimizer.update(grads_conj, opt_state, params)
```

---

## Loss Functions

### `data_loss`
Fits the model to observed pressure measurements at sensor locations.

```python
# pressure_pc is a PointCloud of (coords, measured_pressures)
loss = pl.data_loss(params, model, pressure_pc, criterion=mse)
```

### `hom_pde_loss`
Enforces the homogeneous PDE residual at collocation points.

```python
# coords is a tuple of coordinate arrays (from a domain generator)
loss = pl.hom_pde_loss(params, model, coords, criterion=mse)
```

### `compute_loss` / `compute_weighted_loss`
Aggregate multiple named loss terms.

```python
total, per_term = pl.compute_loss(params, model, batch, losses, criterion=mse)
total, per_term = pl.compute_weighted_loss(params, model, weights, batch, losses, criterion=mse)
```

### Adaptive Gradient-Norm Weighting

Automatically balance loss terms by equalizing their gradient norms:

```python
# Compute new weights based on current gradient norms
new_weights = pl.compute_weights(params, model, batch, losses, criterion=mse)

# Update with exponential moving average (momentum=0.9)
loss_weights = pl.update_weights(0.9, loss_weights, new_weights)
```

---

## Data

### Generators

Generators are infinite iterators that yield batches of training points.

```python
from pinnlib.data import UniformGenerator, SobolGenerator, DataPointGenerator, MeshGenerator

# Collocation points from a rectangular domain
domain_gen = UniformGenerator(
    bounds=[(-0.5, 0.5), (-0.5, 0.5)],
    batch_size=256,
    key=key,
)

# Low-discrepancy collocation (better coverage)
sobol_gen = SobolGenerator(
    bounds=[(-0.5, 0.5), (-0.5, 0.5)],
    batch_size=256,
    key=key,
)

# Batches from observed data
data_gen = DataPointGenerator(point_cloud=dataset, batch_size=64, key=key)

# Points sampled on a mesh surface
mesh_gen = MeshGenerator(mesh=my_mesh, batch_size=128, key=key)

# Use as infinite iterators
for coords in domain_gen:
    ...
```

### `GridDiscretisationND`

Evaluate functions on regular grids and convert to point clouds:

```python
from pinnlib.data import GridDiscretisationND

# Discretize a known function onto a 128×128 grid
field = GridDiscretisationND.discretise_fn(
    fn=lambda x, y: jnp.sin(x) * jnp.cos(y),
    bounds=[(-1.0, 1.0), (-1.0, 1.0)],
    n_points=[128, 128],
)

x, y = field.coordinate_arrays    # meshgrid arrays
vals = field.vals                  # shape (128, 128)
pc = field.as_point_cloud()        # convert to PointCloud
combined = field1 + field2         # element-wise operations supported

# Discretize model predictions back onto a grid for evaluation
from pinnlib.misc import args_to_array
predicted = GridDiscretisationND.discretise_fn(
    fn=args_to_array(lambda *xs: pinn(params, *xs)),
    bounds=field.bounds,
    n_points=field.vals.shape,
)
```

### `PointCloud`

`PointCloud(coords, vals)` is a `NamedTuple` pairing coordinate tuples with values. Most data utilities consume and produce point clouds.

### Point Cloud Utilities (`pinnlib.data.pc_utils`)

```python
from pinnlib.data import pc_utils as pcu

# Subsample to a coarse grid (e.g. sensor locations)
sparse = pcu.grid_sample_points(grid_size=(5, 5))(dataset)

# Random subsample
subset = pcu.sample_points(n=100)(dataset)

# Filter by predicate
interior = pcu.filter_points(lambda coords, vals: coords[0] > 0)(dataset)

# Compose transforms
transform = pcu.pipe(
    pcu.filter_points(lambda c, v: c[0] > 0),
    pcu.sample_points(n=64),
)
result = transform(dataset)

bb = pcu.get_bounding_box(dataset)  # returns list of (min, max) per axis
```

---

## Activation Functions

| Class / Function | Description |
|---|---|
| `SinActivation(omega)` | `sin(ω · x)` — standard SIREN activation |
| `SplitSinActivation(omega)` | Sine applied separately to real and imaginary parts |
| `LearnableTanh(scale)` | Tanh with learnable scale parameter |
| `LearnableSplitTanh(scale_r, scale_i)` | Learnable tanh for real/imaginary separately |
| `WaveletActivation()` | Combines sine and cosine like a wavelet |
| `split_tanh` | Functional version of split tanh |
| `cardioid` | Cardioid-shaped complex activation |
| `rotating_cardioid` | Cardioid with rotation |
| `identity_activation` | No-op (use as `final_activation` for unbounded output) |

---

## Feature Maps (Input Embeddings)

Wrap inputs before the network to improve spectral representation:

```python
from pinnlib.feature_maps import PeriodicFeatures, RandomFourierFeatures

# Trainable per-axis periodic embeddings: (x,y) → (cos(x/T), sin(x/T), cos(y/T), sin(y/T))
embedding = PeriodicFeatures(in_size=2, key=key)

# Random Fourier features (fixed at init)
embedding = RandomFourierFeatures(in_size=2, embed_dim=64, key=key)

# Pass to PINN — in_size is adjusted automatically
pinn = pl.WavePINN.create(..., embedding=embedding, in_size=2, ...)
```

---

## Complex Number Utilities

For Helmholtz problems with complex pressure fields:

```python
# Wrap a real loss function to apply to real and imaginary parts separately
criterion = pl.split_real_and_imaginary_loss(mse)

# Wrap a real metric for complex outputs
metric = pl.split_real_and_imaginary_metric(mse)

# Wrap a real activation for complex inputs
activation = pl.split_real_and_imaginary_activation(jax.nn.tanh)
```

---

## Metrics

### Point-wise

```python
from pinnlib.metrics import abs_error, sq_error, relative, log_error, psnr, diff

e = abs_error(pred, target)
e = sq_error(pred, target)
e = relative(pred, target)
e = psnr(pred, target)
```

### Aggregated (global)

```python
from pinnlib.metrics import mse, rmse, mae, nrmse_range, nrmse_std, mean_rel_error, mag_phase

mse(pred, target)
rmse(pred, target)
mae(pred, target)
nrmse_range(pred, target)   # RMSE normalized by data range
nrmse_std(pred, target)     # RMSE normalized by standard deviation
mean_rel_error(pred, target)
mag_phase(pred, target)     # combined magnitude and phase error (complex fields)
```

---

## Re-parameterization

Tools for modifying model parameter distributions after construction:

```python
from pinnlib.reparametrize import reparam_pytree, reparametrize_linear, filter_tree_map

# Re-initialize weights from a new distribution
model = reparam_pytree(model, weight_init_fn, bias_init_fn)

# Apply a transformation only to selected layers/parameters
model = filter_tree_map(model, transform_fn, filter_fn)

# SIREN-specific initialization
from pinnlib.reparametrize import siren_weight_initializer, siren_bias_initializer
```

---

## Utilities

```python
import pinnlib as pl

pl.default_wave_speed()       # 343.2 m/s
pl.default_medium_density()   # 1.2043 kg/m³
pl.default_floating_dtype()   # float32 or float64 depending on JAX x64 setting
pl.default_complex_dtype()    # complex64 or complex128

# Convert a function f(*args) ↔ f_array(x) where x is a stacked array
from pinnlib.misc import args_to_array, array_to_args
f_array = args_to_array(f)    # useful for GridDiscretisationND.discretise_fn
f_args = array_to_args(f_array, n_args=2)
```

---

## Logging

```python
from pinnlib._logger import TensorboardLogger

logger = TensorboardLogger(log_dir="runs/experiment_1")

logger.log_scalar("loss/total", value=loss, step=step)
logger.log_scalars("loss", {"data": data_loss, "pde": pde_loss}, step=step)
logger.log_histogram("weights/layer0", values=weights, step=step)
logger.log_plot("pressure_field", fig=fig, step=step)
logger.log_hparams({"lr": 1e-3, "width": 64, "depth": 3}, metrics={"final_loss": loss})
logger.flush()
logger.close()
```

---

## Visualization

```python
from pinnlib._plotting import plot_batch

plot_batch(
    mesh=my_mesh,
    point_batches={"sensors": sensor_pc, "collocation": domain_pc},
    ground_truth=gt_field,   # optional volumetric reference
)
```
