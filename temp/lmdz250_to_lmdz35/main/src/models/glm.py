import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from src.core.utils import vprint


def _meshgrid_points(lat, lon):
    """
    Build flattened grid points as array of shape (n_points, 2) = [lat, lon].
    """
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    return np.column_stack([yy.ravel(), xx.ravel()])


def _build_nearest_mapping(lat_in, lon_in, lat_out, lon_out, n_neighbors=1):
    """
    For each fine target grid point, find the nearest coarse predictor grid box(es).

    Returns
    -------
    mapping : list
        mapping[out_flat_idx] = [(iy_in, ix_in), ...]
    """
    src_points = _meshgrid_points(lat_in, lon_in)   # coarse predictor grid
    tgt_points = _meshgrid_points(lat_out, lon_out) # fine target grid

    n_src_lon = len(lon_in)
    mapping = []

    for p in tgt_points:
        d2 = np.sum((src_points - p) ** 2, axis=1)
        nearest_flat = np.argsort(d2)[:n_neighbors]

        neighbors = []
        for idx in nearest_flat:
            iy = idx // n_src_lon
            ix = idx % n_src_lon
            neighbors.append((iy, ix))

        mapping.append(neighbors)

    return mapping


def _extract_features_for_target(x, neighbors):
    """
    Extract predictor features for one target point from nearest coarse boxes.

    Parameters
    ----------
    x : np.ndarray
        Shape (N, C, H_in, W_in)
    neighbors : list of tuples
        [(iy, ix), ...]

    Returns
    -------
    X_feat : np.ndarray
        Shape (N, C * K)
    """
    feats = [x[:, :, iy, ix] for (iy, ix) in neighbors]  # each -> (N, C)
    return np.concatenate(feats, axis=1)


class LocalGaussianGLM:
    """
    One local Gaussian linear model per fine target grid point.

    For each fine point:
    - take nearest coarse predictor box(es)
    - build a local linear model
    - predict temperature on the fine grid

    This is a clean Baño-like baseline for temperature.
    """

    def __init__(self, lat_out, lon_out, lat_in, lon_in, n_neighbors=1):
        self.lat_out = np.asarray(lat_out)
        self.lon_out = np.asarray(lon_out)
        self.lat_in = np.asarray(lat_in)
        self.lon_in = np.asarray(lon_in)
        self.n_neighbors = n_neighbors

        self.n_lat_out = len(lat_out)
        self.n_lon_out = len(lon_out)

        # store one local model per fine-grid point
        self.models = np.full((self.n_lat_out, self.n_lon_out), None, dtype=object)

        # precompute fine -> coarse nearest-neighbor mapping
        self.mapping = _build_nearest_mapping(
            lat_in=self.lat_in,
            lon_in=self.lon_in,
            lat_out=self.lat_out,
            lon_out=self.lon_out,
            n_neighbors=self.n_neighbors,
        )

    def predict(self, x):
        """
        Predict on fine grid.

        Parameters
        ----------
        x : np.ndarray
            Shape (N, C, H_in, W_in)

        Returns
        -------
        predictions : np.ndarray
            Shape (N, H_out, W_out)
        """
        n_samples = x.shape[0]
        predictions = np.full(
            (n_samples, self.n_lat_out, self.n_lon_out),
            np.nan,
            dtype=np.float32,
        )

        for iy in range(self.n_lat_out):
            for ix in range(self.n_lon_out):
                entry = self.models[iy, ix]
                if entry is None:
                    continue

                flat_idx = iy * self.n_lon_out + ix
                neighbors = self.mapping[flat_idx]

                X_feat = _extract_features_for_target(x, neighbors)

                # keep only the features retained during training
                keep_mask = entry["keep_mask"]
                X_feat = X_feat[:, keep_mask]

                X_feat = sm.add_constant(X_feat, has_constant="add")

                try:
                    predictions[:, iy, ix] = X_feat @ entry["params"]
                except Exception:
                    continue

        return predictions


def train_glm(
    cfg,
    x_train,
    y_train,
    lat_in,
    lon_in,
    lat_out,
    lon_out,
    n_neighbors=None,
    min_samples=30,
):
    """
    Train a local Gaussian GLM/OLS family for temperature downscaling.

    Parameters
    ----------
    cfg : Config
    x_train : torch.Tensor
        Shape (N, C, H_in, W_in), coarse predictors
    y_train : torch.Tensor
        Shape (N, 1, H_out, W_out), fine target
    lat_in, lon_in : arrays
        coarse predictor coordinates
    lat_out, lon_out : arrays
        fine target coordinates
    n_neighbors : int
        1 -> GLM1
        4 -> GLM4
    min_samples : int
        Minimum number of valid samples to fit a local model

    Returns
    -------
    last_model, best_model, train_losses, val_losses, best_loss
        Compatible with your current train.py
    """
    if cfg.variable.lower() != "temp":
        raise NotImplementedError(
            "This glm.py implementation is temperature-only. "
            "For precipitation, use a separate Binomial + Gamma implementation."
        )

    if n_neighbors is None:
        n_neighbors = getattr(cfg, "glm_n_neighbors", 4)

    if lat_in is None or lon_in is None or lat_out is None or lon_out is None:
        raise ValueError(
            "GLM training requires lat_in, lon_in, lat_out, and lon_out."
        )

    vprint(f"Starting local Gaussian GLM training with n_neighbors={n_neighbors}...")

    x_train_np = x_train.detach().cpu().numpy()            # (N, C, H_in, W_in)
    y_train_np = y_train.detach().cpu().numpy().squeeze(1) # (N, H_out, W_out)

    glm_wrapper = LocalGaussianGLM(
        lat_out=lat_out,
        lon_out=lon_out,
        lat_in=lat_in,
        lon_in=lon_in,
        n_neighbors=n_neighbors,
    )

    trained_count = 0
    skipped_count = 0

    for iy in tqdm(range(glm_wrapper.n_lat_out), desc="Training Gaussian GLMs"):
        for ix in range(glm_wrapper.n_lon_out):
            flat_idx = iy * glm_wrapper.n_lon_out + ix
            neighbors = glm_wrapper.mapping[flat_idx]

            y_point = y_train_np[:, iy, ix]
            X_feat = _extract_features_for_target(x_train_np, neighbors)

            # valid rows only
            valid = np.isfinite(y_point) & np.all(np.isfinite(X_feat), axis=1)
            y_fit = y_point[valid]
            X_fit = X_feat[valid]

            if len(y_fit) < min_samples:
                skipped_count += 1
                continue

            # skip almost constant targets
            if np.nanstd(y_fit) < 0.05:
                skipped_count += 1
                continue

            # remove constant / near-constant predictors
            feature_std = np.nanstd(X_fit, axis=0)
            keep_mask = feature_std > 1e-8
            X_fit = X_fit[:, keep_mask]

            if X_fit.shape[1] == 0:
                skipped_count += 1
                continue

            X_fit = sm.add_constant(X_fit, has_constant="add")

            try:
                # OLS is more stable here than GLM(Gaussian)
                res = sm.OLS(y_fit, X_fit).fit()

                glm_wrapper.models[iy, ix] = {
                    "params": res.params.astype(np.float32),
                    "keep_mask": keep_mask,
                }
                trained_count += 1

            except Exception:
                skipped_count += 1

    vprint(
        f"Local Gaussian GLM training finished. "
        f"Trained: {trained_count}, Skipped: {skipped_count}"
    )

    # For GLM, last_model == best_model
    return glm_wrapper, glm_wrapper, [], [], None