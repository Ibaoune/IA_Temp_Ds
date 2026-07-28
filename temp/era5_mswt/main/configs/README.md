# Experiment Configuration Overview

This document distinguishes the 17 completed classical experiments from the
Physics-Informed AI configurations defined for separate experiments.

## Configuration Families

The classical, mainly data-driven or probabilistic configurations remain
organized by model family:

```text
configs/
├── cnn/
├── glm/
└── unet/
```

Configurations that add physical or structural constraints are grouped
separately:

```text
configs/phy_ai/
├── cnn/
├── glm/
└── unet/
    ├── config_arch1_xiong_continuity.yaml
    └── config_arch1_xiong_directional.yaml
```

Currently, `phy_ai/unet/` contains the two Xiong-inspired experiments.
`phy_ai/cnn/` and `phy_ai/glm/` are reserved for future Physics-Informed CNN
and GLM experiments; they do not contain model configurations yet.

## Baseline Models

The project establishes two primary baselines to benchmark performance:

### 1. CNN Baseline (`cnn_temperature` & `cnn1_temperature`)
* **Architecture:** Convolutional Neural Network (`mode: cnn10` and `mode: cnn1`)
* **Loss function:** Gaussian (NLL)
* **Optimizer:** Adam
* **Learning Rate:** 1e-3
* **Batch size:** 64
* **Scheduler:** None
* **Regularization:** Dropout (0.1), Weight Decay (1e-4)

### 2. U-Net Optimized Baseline (`unet1`)
* **Architecture:** U-Net (`unet1`)
* **Loss function:** Gaussian (NLL)
* **Optimizer:** AdamW
* **Learning Rate:** 1e-3
* **Batch size:** 32
* **Scheduler:** Cosine Annealing
* **Regularization:** Dropout (0.1), Weight Decay (1e-4)

*(Note: An initial U-Net port named `unet1_temperature` was trained using the basic CNN parameters before establishing the `unet1` optimized baseline).*

---

## Experimental Campaign

The campaign follows a systematic ablation and hyperparameter tuning strategy to evaluate the impact of different architectural and training choices.

### Architecture Experiments

#### CNN vs U-Net Baseline (`unet1_temperature`)
**Configuration file:** `unet/config_arch1.yaml`

**Modification relative to baseline:** Uses the U-Net architecture instead of CNN, but keeps CNN training parameters (Adam, BS=64, no scheduler).

**Objective:** Isolate the impact of the U-Net architecture's spatial skip-connections.

**Hypothesis being tested:** U-Net provides better high-resolution spatial localization than standard CNNs.

**Expected impact:** Improved spatial correlation metrics and better extreme value capture.

#### Training Regime Upgrade (`unet1`)
**Configuration file:** `unet/unet1.yaml` (and identically `unet/unet1_loss_gaussian.yaml`)

**Modification relative to baseline:** Switched from `unet1_temperature` basic params to advanced params: AdamW, Batch Size 32, Cosine Scheduler.

**Objective:** Maximize U-Net convergence capability.

**Hypothesis being tested:** Modern optimization techniques (AdamW + Cosine Annealing) unlock U-Net's full capacity.

**Expected impact:** Lower training/validation loss and better overall RMSE.

### Loss Function Experiments

#### Mean Squared Error vs Gaussian (`cnn_mse_temperature` / `unet1_temp_mse` / `unet1_loss_mse`)
**Configuration files:** `cnn/config_mse.yaml`, `unet/config_unet1_mse.yaml`, `unet/unet1_loss_mse.yaml`

**Modification relative to baseline:** Replaced the Gaussian Negative Log-Likelihood loss with deterministic MSE.

**Objective:** Compare probabilistic variance-aware predictions against purely deterministic predictions.

**Hypothesis being tested:** Gaussian loss allows the model to capture heteroscedastic uncertainty better than MSE, improving performance on extreme temperature variations.

**Expected impact:** MSE might yield a slightly better pure RMSE, but Gaussian should heavily outperform on extreme distribution metrics (B02, B98, WAMS).

### Physics-Informed AI Experiments

The Physics-Informed configurations use the standard `UNet1` architecture and
add one structural constraint at a time:

- `phy_ai/unet/config_arch1_xiong_continuity.yaml`
  - Architecture: `UNet1`
  - Loss: `xiong_continuity`
  - Constraint: spatial continuity
  - Output: deterministic, one channel
- `phy_ai/unet/config_arch1_xiong_directional.yaml`
  - Architecture: `UNet1`
  - Loss: `xiong_directional`
  - Constraint: directional consistency
  - Output: deterministic, one channel

The continuity and directional constraints are tested in separate experiments.
They are not combined into a single loss.

Run the experiments from the repository root:

```bash
cd temp/era5_mswt/main

python train.py configs/phy_ai/unet/config_arch1_xiong_continuity.yaml
python eval.py configs/phy_ai/unet/config_arch1_xiong_continuity.yaml

python train.py configs/phy_ai/unet/config_arch1_xiong_directional.yaml
python eval.py configs/phy_ai/unet/config_arch1_xiong_directional.yaml
```

### Optimization Experiments

#### Batch Size Variations (`cnn_bs32_temperature` / `unet1_temp_bs32` / `unet_bs_64`)
**Configuration files:** `cnn/config_bs32.yaml`, `unet/config_unet1_bs32.yaml`, `unet/unet_bs_64.yaml`

**Modification relative to baseline:** Inverted the batch sizes (from 64 to 32 for CNN/early-UNet, and from 32 to 64 for optimized UNet).

**Objective:** Evaluate the generalization gap induced by batch size.

**Hypothesis being tested:** Smaller batch sizes provide more stochasticity, acting as implicit regularization and escaping local minima.

**Expected impact:** Improved validation loss and reduced overfitting for Batch Size 32.

#### Learning Rate Decay (`unet_lr_1e4` / `unet_lr_5e4`)
**Configuration files:** `unet/unet_lr_1e4.yaml`, `unet/unet_lr_5e4.yaml`

**Modification relative to baseline:** Reduced base learning rate from 1e-3 to 1e-4 and 5e-4 respectively.

**Objective:** Prevent catastrophic forgetting or divergence during Cosine Annealing.

**Hypothesis being tested:** The default 1e-3 LR might be too aggressive for the U-Net's deep layers.

**Expected impact:** Smoother, albeit slower, convergence curves and potentially better fine-grained spatial accuracy.

#### Scheduler Removal (`unet_no_scheduler`)
**Configuration file:** `unet/unet_no_scheduler.yaml`

**Modification relative to baseline:** Disabled the Cosine Annealing scheduler.

**Objective:** Establish the exact contribution of the learning rate schedule.

**Hypothesis being tested:** A static learning rate causes the model to plateau in suboptimal local minima.

**Expected impact:** Higher final RMSE compared to the baseline `unet1`.

### Regularization Experiments

#### Elevated Dropout (`unet1_temp_reg` / `unet_dropout_02`)
**Configuration files:** `unet/config_unet1_reg.yaml` (Dropout 0.3), `unet/unet_dropout_02.yaml` (Dropout 0.2)

**Modification relative to baseline:** Increased Dropout from 0.1 to 0.2 and 0.3.

**Objective:** Combat overfitting.

**Hypothesis being tested:** The U-Net architecture might be heavily memorizing the training climatology.

**Expected impact:** Reduced gap between Training and Validation loss; potentially worse raw bias due to underfitting if 0.3 is too high.

#### Aggressive Weight Decay (`unet_weight_decay_1e3`)
**Configuration file:** `unet/unet_weight_decay_1e3.yaml`

**Modification relative to baseline:** Increased AdamW weight decay from 1e-4 to 1e-3.

**Objective:** Enforce smoother filters and restrict model capacity.

**Hypothesis being tested:** Stronger L2 regularization prevents the model from relying on noisy, localized artifacts in the ERA5 predictors.

**Expected impact:** More spatially cohesive predictions, avoiding pixel-level 'checkerboard' artifacts.

---

## Experiment Design Logic

The overall strategy of this campaign is structured as a hierarchical grid search focused on empirical climate downscaling:
1. **First phase:** Establish if a complex spatial architecture (U-Net) outperforms a local one (CNN).
2. **Second phase:** Determine the most robust loss function for climate fields. The debate between MSE (mean-seeking) and Gaussian (distribution-seeking) is critical for capturing climate extremes like heatwaves.
3. **Third phase:** Once the architecture and loss are set, tune the optimization landscape. Climate datasets are highly correlated sequentially, so finding the right batch size and learning rate scheduler is key to proper gradient descent.
4. **Final phase:** Apply strict regularization (Dropout, Weight Decay) to ensure the downscaled outputs don't just memorize the topography, but actually learn transferrable thermodynamic mappings.

---

## Completed Classical Experiments Summary Table

| Experiment | Category | Modified Parameter | Objective | Postprocessed |
| ---------- | -------- | ------------------ | --------- | ------------- |
| `cnn_temperature` | Baseline | None (CNN Base) | Reference benchmark | Yes |
| `cnn1_temperature` | Baseline | Mode (cnn1) | Shallow vs Deep CNN | Yes |
| `unet1_temperature` | Arch | UNet (CNN params) | Evaluate architecture | Yes |
| `unet1` | Baseline | AdamW, BS=32, Cosine | Optimized UNet Ref | Yes |
| `cnn_mse_temperature` | Loss | Loss = MSE | CNN deterministic | Yes |
| `unet1_temp_mse` | Loss | Loss = MSE (Basic) | UNet deterministic | Yes |
| `unet1_loss_mse` | Loss | Loss = MSE (Optim) | UNet deterministic | Yes |
| `unet1_loss_gaussian` | Loss | None (Duplicate) | Baseline validation | Yes |
| `cnn_bs32_temperature` | Optim | Batch Size = 32 | Implicit regularization | Yes |
| `unet1_temp_bs32` | Optim | Batch Size = 32 | Implicit regularization | Yes |
| `unet_bs_64` | Optim | Batch Size = 64 | Batch normalization diff | Yes |
| `unet_lr_1e4` | Optim | LR = 1e-4 | Convergence stability | Yes |
| `unet_lr_5e4` | Optim | LR = 5e-4 | Convergence stability | Yes |
| `unet_no_scheduler` | Optim | Sched = None | Scheduler ablation | Yes |
| `unet1_temp_reg` | Reg | Dropout = 0.3 | High dropout combat | Yes |
| `unet_dropout_02` | Reg | Dropout = 0.2 | Mid dropout combat | Yes |
| `unet_weight_decay_1e3`| Reg | WD = 1e-3 | Stronger L2 penalty | Yes |
