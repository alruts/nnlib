# pinnlib

A library of neural network architectures for acoustics-informed applications, built on JAX and Equinox.

## Installation

```bash
uv sync
```

**Dependencies:** JAX, Equinox, Optax, Diffrax, PyVista, TensorFlow (logging), and more — see `pyproject.toml`.

---

## Features

### Core PINN Classes

**`WavePINN`** — time-domain acoustic wave equation PINN
- `create()` — construct with configurable architecture
- `__call__()` — forward pressure prediction
- `residual()` — wave equation PDE residual
- `velocity()` — directional particle velocity
- `impedance()` — normalized acoustic impedance

**`HelmholtzPINN`** — frequency-domain Helmholtz equation PINN
- `create()`, `__call__()`, `residual()`, `velocity()`, `impedance()` — same interface as `WavePINN`

**Standalone PDE functions:**
- `wave_residual()`, `wave_directional_velocity()`, `wave_impedance()`
- `helmholtz_residual()`, `helmholtz_directional_velocity()`, `helmholtz_impedance()`

---

### Neural Network Architectures

- `MLPWithFirstActivation` — MLP with a separate first-layer activation
- `ModifiedMLP` — MLP with learned linear modulators
- `PirateBlock` / `PirateNet` — physics-informed residual adaptive network
- `make_siren()` — SIREN (Sinusoidal Representation Network)
- `make_modified_siren()` — SIREN with custom initialization

---

### Activation Functions

- `SinActivation` — sine scaled by angular frequency
- `SplitSinActivation` — sine for complex inputs (applied to real/imag separately)
- `LearnableSplitTanh` — split tanh with learnable scale for complex inputs
- `LearnableTanh` — tanh with learnable scale
- `WaveletActivation` — sine + cosine wavelet activation
- `split_tanh()`, `cardioid()`, `rotating_cardioid()`, `identity_activation()`

---

### Loss Functions

- `data_loss()` — predicted vs. observed pressure
- `hom_pde_loss()` — homogeneous PDE residual loss
- `compute_loss()` — total loss with per-term breakdown
- `compute_weighted_loss()` — loss with per-term weights
- `compute_weights()` — gradient-norm-based adaptive weighting
- `update_weights()` — running average weight update with momentum
- `compute_mask()` — exponential mask for loss weighting

---

### Metrics

Point-wise: `psnr()`, `abs_error()`, `sq_error()`, `relative()`, `log_error()`, `diff()`

Global: `mse()`, `rmse()`, `mae()`, `nrmse_range()`, `nrmse_std()`, `mean_rel_error()`, `mag_phase()`

---

### Complex Number Utilities

- `split_real_and_imaginary_activation()` — extend real activations to complex inputs
- `split_real_and_imaginary_metric()` — extend real metrics to complex inputs
- `split_real_and_imaginary_loss()` — extend real losses to complex inputs

---

### Feature Maps / Input Embeddings

- `Identity` — no transformation
- `PeriodicFeatures` — cos/sin embeddings with per-axis trainable periods
- `RandomFourierFeatures` — random Gaussian Fourier embeddings

---

### Re-parameterization

- `reparametrize_linear()` — re-parameterize Linear layer weights/biases
- `siren_weight_initializer()` / `siren_bias_initializer()` — SIREN-specific initialization
- `reparam_pytree()` — re-parameterize any model with a new distribution
- `filter_tree_map()` — apply a transformation to selected model parameters
- `make_nd_array_filter()`, `make_is_leaf_of_filter()`

---

### Data Structures

- `PointCloud` — NamedTuple of coordinate arrays and values
- `GridDiscretisationND` — N-dimensional regular grid with `discretise_fn()`, `coordinate_arrays`, `as_point_cloud()`, and element-wise operators (`+`, `-`, `*`)

Type aliases: `Coords`, `Coord`, `Vecs`, `Vec`, `CoordsVecs`, `Vals`

---

### Data Generators

- `SobolGenerator` — low-discrepancy Sobol sequence sampling
- `UniformGenerator` — uniform rectangular domain sampling
- `MeshGenerator` — surface sampling via barycentric coordinates
- `DataPointGenerator` — random batch sampling from a `PointCloud`

---

### Point Cloud Utilities

- `map_coords()`, `map_vals()` — apply functions to coordinates or values
- `filter_points()` — filter points by predicate
- `grid_sample_points()` — subsample using a uniform grid
- `sample_points()` — random n-point sampling
- `pipe()` — compose multiple transforms
- `get_bounding_box()`, `discretise_fn()`

---

### Utilities

- `default_floating_dtype()`, `default_complex_dtype()` — JAX dtype helpers
- `default_wave_speed()` — 343.2 m/s
- `default_medium_density()` — 1.2043 kg/m³
- `apply_model()` — enables gradient computation through model weights
- `get_parameters()`, `args_to_array()`, `array_to_args()`

---

### Visualization

- `plot_batch()` — PyVista 3D plot of mesh, point batches, and optional ground-truth volume

---

### Logging

**`TensorboardLogger`** — TensorBoard integration for JAX experiments
- `log_scalar()`, `log_scalars()`, `log_histogram()`, `log_plot()`, `log_text()`, `log_hparams()`, `flush()`, `close()`
