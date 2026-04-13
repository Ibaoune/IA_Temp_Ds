import numpy as np
import statsmodels.api as sm
import os
import pickle
from tqdm import tqdm
from utils import vprint

class PixelWiseGLM:
    """
    A wrapper for pixel-wise GLM models (Binomial + Gamma for precip, Gaussian for temp).
    """
    def __init__(self, n_lat, n_lon, variable="precip"):
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.variable = variable
        self.models = np.full((n_lat, n_lon), None, dtype=object)

    def predict(self, x):
        """
        x: (N, C, H, W)
        Returns: (N, H, W)
        """
        n_samples, n_channels, n_lat, n_lon = x.shape
        predictions = np.full((n_samples, n_lat, n_lon), np.nan, dtype=np.float32)

        for i in range(n_lat):
            for j in range(n_lon):
                model_entry = self.models[i, j]
                if model_entry is None:
                    continue
                
                x_feat = x[:, :, i, j]
                X_feat_sm = sm.add_constant(x_feat, has_constant='add')
                
                if self.variable == "precip":
                    try:
                        prob = model_entry['occurrence'].predict(X_feat_sm)
                        amount = model_entry['amount'].predict(X_feat_sm)
                        predictions[:, i, j] = prob * amount
                    except Exception:
                        continue
                else:
                    try:
                        predictions[:, i, j] = model_entry.predict(X_feat_sm)
                    except Exception:
                        continue
        
        return predictions

def train_glm(cfg, x_train, y_train):
    """
    Fits GLM models for each pixel.
    x_train: (N, C, H, W) torch.Tensor
    y_train: (N, 1, H, W) torch.Tensor
    """
    vprint(f"Starting GLM training for {cfg.variable}...")
    
    # Convert to numpy for statsmodels
    x_train_np = x_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy().squeeze(1) # (N, H, W)
    
    n_samples, n_channels, n_lat, n_lon = x_train_np.shape
    
    glm_wrapper = PixelWiseGLM(n_lat, n_lon, variable=cfg.variable)
    
    trained_count = 0
    skipped_count = 0
    
    # Create save dirs
    model_save_dir = os.path.join(cfg.exp_dir, "glm_models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    for i in tqdm(range(n_lat), desc="Training GLMs"):
        for j in range(n_lon):
            y_point = y_train_np[:, i, j]
            if np.isnan(y_point).any():
                skipped_count += 1
                continue
                
            x_feat = x_train_np[:, :, i, j]
            # Add constant
            X_feat_sm = sm.add_constant(x_feat, has_constant='add')
            
            try:
                if cfg.variable == "precip":
                    # Binomial for occurrence
                    y_binary = (y_point >= 1.0).astype(float)
                    glm_occ = sm.GLM(y_binary, X_feat_sm, family=sm.families.Binomial())
                    res_occ = glm_occ.fit()
                    
                    # Gamma for quantity
                    rainy_idx = np.where(y_point >= 1.0)[0]
                    if len(rainy_idx) < 10: # Minimum rainy days to fit gamma
                        skipped_count += 1
                        continue
                        
                    y_wet = y_point[rainy_idx]
                    X_wet = X_feat_sm[rainy_idx]
                    glm_amt = sm.GLM(y_wet, X_wet, family=sm.families.Gamma(link=sm.families.links.log()))
                    res_amt = glm_amt.fit()
                    
                    glm_wrapper.models[i, j] = {'occurrence': res_occ, 'amount': res_amt}
                    trained_count += 1
                else:
                    # Gaussian for temperature
                    glm_gauss = sm.GLM(y_point, X_feat_sm, family=sm.families.Gaussian())
                    res_gauss = glm_gauss.fit()
                    glm_wrapper.models[i, j] = res_gauss
                    trained_count += 1
            except Exception as e:
                # vprint(f"Failed at ({i},{j}): {e}")
                skipped_count += 1
                
    vprint(f"GLM training finished. Trained: {trained_count}, Skipped: {skipped_count}")
    
    # We return the wrapper and dummy losses
    return glm_wrapper, [], []
