## Physics/structure-informed experiments

The configurations in `configs/phy_ai/cnn/` add spatial-structure terms to
deterministic, single-channel CNN1 and CNN10 predictions.

These experiments are described as **physics-guided** or
**structure-informed**, rather than as Physics-Informed Neural Networks
(PINNs). They do not impose a governing partial differential equation or a
conservation law. Instead, they encourage the predicted temperature field to
reproduce selected spatial properties of the reference field.

The implemented ideas are based on the following two articles.

---

### 1. Xiong (2025): continuity and change-angle fidelity

**Article**

Minquan Xiong (2025), *Impact of Physical Constraints on Deep
Learning-Based Downscaling Prediction of Temperature*, Journal of
Meteorological Research, 39(4), 904–919.

[https://doi.org/10.1007/s13351-025-4061-1](https://doi.org/10.1007/s13351-025-4061-1)

#### What the article does

Xiong applies a U-Net to downscale ECMWF daily maximum 2-m temperature from
\(0.5^\circ\) to \(0.05^\circ\) over the lower reaches of the Yangtze River.
The model uses physically meaningful predictors, including elevation, 10-m
zonal and meridional winds, and the direct model output temperature.

The article also introduces two hybrid losses. Both retain an RMSE data term
and add one spatial penalty:

1. a **continuity penalty**, based on squared differences between adjacent
   temperature values;
2. a **change-angle fidelity penalty**, based on two-argument arctangents of
   adjacent temperature values.

The equations below reproduce the operations defined in the article, written
with unambiguous \(H \times W\) notation. Let \(\widehat{T}_{i,j}\) denote the
predicted temperature and \(T_{i,j}\) the reference temperature.

#### Continuity error

The continuity error compares the total squared neighbour-difference energy
of the predicted and reference fields:

$$
\begin{aligned}
E_c =
\Bigg|
&\left[
\sum_{i=1}^{H-1}\sum_{j=1}^{W}
\left(\widehat{T}_{i+1,j}-\widehat{T}_{i,j}\right)^2
+
\sum_{i=1}^{H}\sum_{j=1}^{W-1}
\left(\widehat{T}_{i,j+1}-\widehat{T}_{i,j}\right)^2
\right] \\
-&\left[
\sum_{i=1}^{H-1}\sum_{j=1}^{W}
\left(T_{i+1,j}-T_{i,j}\right)^2
+
\sum_{i=1}^{H}\sum_{j=1}^{W-1}
\left(T_{i,j+1}-T_{i,j}\right)^2
\right]
\Bigg|.
\end{aligned}
$$

The hybrid continuity loss is:

$$
L_c = \mathrm{RMSE} + w_c E_c.
$$

This term does not impose the fluid-mechanics continuity equation. It matches
an aggregate measure of spatial smoothness between the prediction and the
target. A prediction can therefore have a low continuity error when its total
neighbour-difference energy is similar to the target, even if some local
gradients differ.

#### Change-angle fidelity error

The article defines a directional or change-angle quantity from adjacent
temperature values. In the implementation, the two-argument arctangent is
evaluated as `atan2(neighbour, current)`:

$$
\begin{aligned}
E_d =
\Bigg|
&\left[
\sum_{i=1}^{H-1}\sum_{j=1}^{W}
\operatorname{atan2}
\left(\widehat{T}_{i+1,j},\widehat{T}_{i,j}\right)
+
\sum_{i=1}^{H}\sum_{j=1}^{W-1}
\operatorname{atan2}
\left(\widehat{T}_{i,j+1},\widehat{T}_{i,j}\right)
\right] \\
-&\left[
\sum_{i=1}^{H-1}\sum_{j=1}^{W}
\operatorname{atan2}
\left(T_{i+1,j},T_{i,j}\right)
+
\sum_{i=1}^{H}\sum_{j=1}^{W-1}
\operatorname{atan2}
\left(T_{i,j+1},T_{i,j}\right)
\right]
\Bigg|.
\end{aligned}
$$

The hybrid directional loss is:

$$
L_d = \mathrm{RMSE} + w_d E_d.
$$

This is the change-angle definition used by Xiong. It should not be confused
with the standard orientation of a finite-difference gradient vector,
\(\operatorname{atan2}(\partial_y T,\partial_x T)\).

#### Adaptation in this repository

The article defines the losses for individual two-dimensional fields. The
repository adapts them to mini-batches by computing the structural error for
each sample and averaging across the batch. A small numerical-stability
constant is also used in the RMSE calculation.

The supplied configurations use:

```yaml
continuity_weight: 1.0e-4
directional_weight: 1.0e-4
```

These values are **project hyperparameters**. Xiong does not prescribe
\(10^{-4}\) as a universal value; the article recommends selecting the weights
experimentally according to validation performance.

Associated configurations:

```text
configs/phy_ai/cnn/
├── config_cnn1_xiong_continuity.yaml
├── config_cnn1_xiong_continuity_test.yaml
├── config_cnn1_xiong_directional.yaml
├── config_cnn1_xiong_directional_test.yaml
├── config_cnn10_xiong_continuity.yaml
├── config_cnn10_xiong_continuity_test.yaml
├── config_cnn10_xiong_directional.yaml
└── config_cnn10_xiong_directional_test.yaml
```

---

### 2. Serifi et al. (2021): value and gradient loss

**Article**

Agon Serifi, Tobias Günther, and Nikolina Ban (2021),
*Spatio-Temporal Downscaling of Climate Data Using Convolutional and
Error-Predicting Neural Networks*, Frontiers in Climate, 3, 656479.

[https://doi.org/10.3389/fclim.2021.656479](https://doi.org/10.3389/fclim.2021.656479)

#### What the article does

Serifi et al. study spatial and temporal reconstruction of temperature and
precipitation fields produced by the COSMO regional climate model. For the
slowly varying temperature field, they use a residual-predicting network that
learns a correction to a conventional interpolation baseline.

The authors note that convolutional networks can generate overly smooth
outputs. They therefore combine an \(L_1\) value error with a gradient error
so that the network is penalized for errors in both field values and
derivatives.

#### Published loss equation

Let \(Y\) denote the reference field and \(\widehat{Y}\) the prediction. The
loss published in the article is:

$$
L\left(Y,\widehat{Y}\right)
=
\left\|Y-\widehat{Y}\right\|_1
+
\lambda
\left\|\nabla Y-\nabla\widehat{Y}\right\|_1.
$$

The first term preserves pointwise accuracy. The second compares derivatives
and is intended to reduce over-smoothing and improve the reconstruction of
higher-frequency details.

Serifi et al. empirically use \(\lambda=1\) in their reported experiments.
This value is also used as the default `gradient_weight` in the supplied
project configurations.

#### Adaptation in this repository

The article writes the derivative term compactly with \(\nabla\). In this
repository, the gradient is implemented with forward finite differences along
the two spatial axes only. The horizontal and vertical gradient errors are
averaged and added to the \(L_1\) data term.

The current implementation therefore reproduces the spatial part of the
published gradient-loss idea. It does not include temporal derivatives, a PDE
residual, or a conservation constraint.

Associated configurations:

```text
configs/phy_ai/cnn/
├── config_cnn1_serifi_gradient.yaml
├── config_cnn1_serifi_gradient_test.yaml
├── config_cnn10_serifi_gradient.yaml
└── config_cnn10_serifi_gradient_test.yaml
```

---

## Full and reduced configurations

Each CNN/loss combination has:

- a full configuration for scientific experiments;
- a matching `_test.yaml` configuration for inexpensive integration tests.

The reduced configurations preserve the selected architecture and loss but
use shorter date ranges, fewer epochs, a smaller batch size, and a separate
test-results directory. They should not be used to report final scientific
results.

Before launching a full experiment, verify:

- input and target paths;
- training, validation, and test periods;
- result directories;
- CNN architecture;
- loss name and weight;
- output channel count;
- random seed.

## Example commands

Run from the repository root.

### Xiong continuity loss

```bash
python temp/era5_mswt/main/train.py \
  temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn10_xiong_continuity.yaml

python temp/era5_mswt/main/eval.py \
  temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn10_xiong_continuity.yaml
```

### Xiong directional loss

```bash
python temp/era5_mswt/main/train.py \
  temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn10_xiong_directional.yaml

python temp/era5_mswt/main/eval.py \
  temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn10_xiong_directional.yaml
```

### Serifi gradient loss

```bash
python temp/era5_mswt/main/train.py \
  temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn10_serifi_gradient.yaml

python temp/era5_mswt/main/eval.py \
  temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn10_serifi_gradient.yaml
```