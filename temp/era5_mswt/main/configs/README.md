## Configuration tree

```text
configs/
├── cnn/
│   ├── config.yaml
│   ├── config_bs32.yaml
│   ├── config_cnn1.yaml
│   ├── config_cnn1_mse.yaml
│   ├── config_cnn10.yaml
│   ├── config_cnn10_bs32.yaml
│   ├── config_cnn10_mse.yaml
│   ├── config_mse.yaml
│   └── test.yaml
└── phy_ai/
    └── cnn/
        ├── config_cnn1_xiong_continuity.yaml
        ├── config_cnn1_xiong_continuity_test.yaml
        ├── config_cnn1_xiong_directional.yaml
        ├── config_cnn1_xiong_directional_test.yaml
        ├── config_cnn1_serifi_gradient.yaml
        ├── config_cnn1_serifi_gradient_test.yaml
        ├── config_cnn10_xiong_continuity.yaml
        ├── config_cnn10_xiong_continuity_test.yaml
        ├── config_cnn10_xiong_directional.yaml
        ├── config_cnn10_xiong_directional_test.yaml
        ├── config_cnn10_serifi_gradient.yaml
        └── config_cnn10_serifi_gradient_test.yaml
```

## Classical CNN experiments

The classical configurations cover CNN1 and CNN10 with either Gaussian
negative log-likelihood or MSE, plus batch-size variants. Gaussian experiments
produce a mean and log-variance; deterministic MSE experiments produce one
temperature channel. `test.yaml` is the reduced classical configuration.

## Physics/structure-informed experiments

The `phy_ai/cnn/` configurations apply one loss at a time to deterministic,
single-channel CNN1 or CNN10 predictions. Their scientific motivation comes
from:

- Minquan Xiong (2025), *Impact of Physical Constraints on Deep
  Learning-Based Downscaling Prediction of Temperature*, Journal of
  Meteorological Research, 39(4), 904-919.
  [https://doi.org/10.1007/s13351-025-4061-1](https://doi.org/10.1007/s13351-025-4061-1)
- Agon Serifi, Tobias Günther, and Nikolina Ban (2021), *Spatio-Temporal
  Downscaling of Climate Data Using Convolutional and Error-Predicting Neural
  Networks*, Frontiers in Climate, 3:656479.
  [https://doi.org/10.3389/fclim.2021.656479](https://doi.org/10.3389/fclim.2021.656479)

The equations below describe the losses implemented in
`src/core/losses.py`. Let \(P_{b,i,j}\) be the predicted temperature,
\(T_{b,i,j}\) the target, \(B\) the batch size, and \(\varepsilon>0\) the
numerical-stability constant.

### Common RMSE data term for the Xiong losses

Both Xiong variants begin with:

$$
L_{\mathrm{RMSE}}
=\sqrt{\frac{1}{BHW}\sum_{b=1}^{B}\sum_{i=1}^{H}\sum_{j=1}^{W}
\left(P_{b,i,j}-T_{b,i,j}\right)^2+\varepsilon}.
$$

This term maintains pointwise predictive accuracy. Each Xiong constraint then
adds a penalty measuring disagreement between a structural quantity computed
from the predicted and reference fields.

### Xiong spatial-continuity loss

For any field \(F\), define its spatial continuity energy as the sum of squared
differences between horizontal and vertical neighbours:

$$
C(F_b)=
\sum_{i=1}^{H-1}\sum_{j=1}^{W}\left(F_{b,i+1,j}-F_{b,i,j}\right)^2
+\sum_{i=1}^{H}\sum_{j=1}^{W-1}\left(F_{b,i,j+1}-F_{b,i,j}\right)^2.
$$

The implemented loss is:

$$
L_{\mathrm{continuity}}
=L_{\mathrm{RMSE}}
+w_c\frac{1}{B}\sum_{b=1}^{B}\left|C(P_b)-C(T_b)\right|,
$$

with `continuity_weight` \(w_c=10^{-4}\) in the supplied configurations.
Following Xiong, its purpose is to constrain the spatial continuity/smoothness
of the downscaled temperature field, discouraging unrealistic neighbouring-grid
variations while retaining the data-fit objective. It matches the target's
aggregate continuity energy; it is not a fluid continuity equation.

### Xiong directional-consistency loss

For any field \(F\), define the aggregate neighbour direction:

$$
D(F_b)=
\sum_{i=1}^{H-1}\sum_{j=1}^{W}
\operatorname{atan2}\!\left(F_{b,i+1,j},F_{b,i,j}\right)
+\sum_{i=1}^{H}\sum_{j=1}^{W-1}
\operatorname{atan2}\!\left(F_{b,i,j+1},F_{b,i,j}\right).
$$

The implemented loss is:

$$
L_{\mathrm{directional}}
=L_{\mathrm{RMSE}}
+w_d\frac{1}{B}\sum_{b=1}^{B}\left|D(P_b)-D(T_b)\right|,
$$

with `directional_weight` \(w_d=10^{-4}\). Its purpose is to make the
prediction reproduce the target field's aggregate spatial direction changes,
thereby improving directional consistency between neighbouring temperatures.
The PyTorch implementation uses `atan2(neighbour, current)`.

### Serifi spatial-gradient loss

Define forward spatial differences:

$$
\Delta_xF_{b,i,j}=F_{b,i,j+1}-F_{b,i,j},\qquad
\Delta_yF_{b,i,j}=F_{b,i+1,j}-F_{b,i,j}.
$$

The three averaged terms are:

$$
L_{\mathrm{data}}=\operatorname{mean}|P-T|,
$$

$$
L_{\nabla x}=\operatorname{mean}|\Delta_xP-\Delta_xT|,\qquad
L_{\nabla y}=\operatorname{mean}|\Delta_yP-\Delta_yT|.
$$

The implemented loss is:

$$
L_{\mathrm{Serifi}}
=L_{\mathrm{data}}+\lambda\left(L_{\nabla x}+L_{\nabla y}\right),
$$

with `gradient_weight` \(\lambda=1\) in the supplied configurations. Serifi
et al. combine an \(L_1\) value loss with a gradient loss because CNN outputs
tend to be overly smooth. Penalizing derivative errors helps reconstruct sharp,
high-frequency spatial details. This implementation uses spatial gradients
only; it does not include temporal gradients or impose a conservation law.

## Full and reduced configurations

Each CNN1/loss and CNN10/loss combination has a full configuration and a
matching `_test.yaml` configuration. The reduced variants preserve the model
and loss settings but use 20 epochs, a batch size of 4, training over
1980-1984, testing on 1985, disabled validation, and
`./temp/results_test/`. They are intended for inexpensive integration runs,
not as final scientific experiments.

## Running an experiment

Run from the repository root so the relative data paths resolve consistently:

```bash
python temp/era5_mswt/main/train.py temp/era5_mswt/main/configs/cnn/config_cnn1.yaml
python temp/era5_mswt/main/eval.py temp/era5_mswt/main/configs/cnn/config_cnn1.yaml
```

For a physics/structure-informed experiment:

```bash
python temp/era5_mswt/main/train.py temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn10_serifi_gradient.yaml
python temp/era5_mswt/main/eval.py temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn10_serifi_gradient.yaml
```

Before starting a full experiment, verify `paths`, date ranges, result
directories, and the desired CNN mode in the selected YAML file.
