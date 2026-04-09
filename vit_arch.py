import torch
import torch.nn as nn
from einops import rearrange  
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BernoulliGammaLoss(nn.Module):
    def __init__(self):
        super(BernoulliGammaLoss, self).__init__()

    def forward(self, true, pred):
        eps = 1e-5

        ocurrence = torch.sigmoid(pred[:, 0, :, :]).clamp(eps, 1-eps)
        shape_parameter = torch.exp(pred[:, 1, :, :].clamp(-5, 5)).clamp(eps, 1e3)
        scale_parameter = torch.exp(pred[:, 2, :, :].clamp(-5, 5)).clamp(eps, 1e3)
        #bool_rain = (true > 0).float()
        #ocurrence = pred[:, 0, :, :]
        #shape_parameter = torch.exp(pred[:, 1, :, :])
        #scale_parameter = torch.exp(pred[:, 2, :, :])
        bool_rain = torch.where(true > 0, torch.tensor(1.0), torch.tensor(0.0))
        epsilon = 0.000001
        #print(f"Valeur de epsilon : {epsilon}")

        # Calcul de la perte en combinant les différentes parties
        loss = (-torch.mean((1 - bool_rain) * torch.log(1 - ocurrence + epsilon) +  # Partie pour les valeurs où il ne pleut pas
                             bool_rain * (torch.log(ocurrence + epsilon) +           # Partie pour les valeurs où il pleut
                                          (shape_parameter - 1) * torch.log(true + epsilon) -
                                          shape_parameter * torch.log(scale_parameter + epsilon) -
                                          torch.lgamma(shape_parameter + epsilon) -         # Calcul du log gamma pour shape_parameter
                                          true / (scale_parameter + epsilon))))  # Terme de normalisation avec scale_parameter
        #print(f"Perte calculée : {loss.item()}")

        return loss

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        patches = self.projection(x)
        h_patches, w_patches = patches.shape[2], patches.shape[3]
        patches = rearrange(patches, 'b e h w -> b (h w) e')
        return patches, h_patches, w_patches

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Linear(emb_size * 4, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_size, num_layers, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(emb_size, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class UpsamplingDecoder(nn.Module):
    def __init__(self, emb_size, patch_size, output_channels):
        """
        Upsamples the encoded representation back into an image.

        emb_size (int): Dimension of each patch
        patch_size (int): Size of each square patch
        output_channels (int): Number of output channels
        """
        super(UpsamplingDecoder, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.ConvTranspose2d(
            emb_size, output_channels,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, h_patches, w_patches, target_h=None, target_w=None):
        x = rearrange(x, 'b (h w) e -> b e h w', h=h_patches, w=w_patches)
        x = self.projection(x)

        if target_h is not None and target_w is not None:
            pad_h = target_h - x.shape[2]
            pad_w = target_w - x.shape[3]
            if pad_h > 0 or pad_w > 0:
               x = F.pad(x, (0, pad_w, 0, pad_h))
        return x


# Modèle complet combinant embedding, encodeur Transformer et décodeur
class DownscalingViT(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads,
                 dropout, output_channels, n_lat_out, n_lon_out):
        """
        Vision Transformer pour des tâches de descente d’échelle 2D.
        """
        super(DownscalingViT, self).__init__()

        # Dimensions de sortie ciblées
        self.n_lat_out = n_lat_out
        self.n_lon_out = n_lon_out

        # Module d'encodage des patchs
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)

        # Embedding de position, initialisé dynamiquement
        self.positional_embedding = None

        # Encodeur Transformer
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)

        # Décodeur final
        self.decoder = UpsamplingDecoder(emb_size, patch_size, output_channels)

    def forward(self, x):
        # x : (batch_size, in_channels, height, width)

        # Redimensionne l’entrée à la taille cible par interpolation nearest neighbor
        x = nn.functional.interpolate(x, size=(self.n_lat_out, self.n_lon_out), mode='nearest')

        # Découpe l’image en patchs encodés
        patches, h_patches, w_patches = self.patch_embedding(x)

        num_patches = patches.shape[1]  # nb_patchs = h_patches × w_patches
        device = patches.device  # GPU s'assure que l'embedding est sur le bon device

        if (self.positional_embedding is None or
                self.positional_embedding.shape[1] != num_patches):
            self.positional_embedding = nn.Parameter(
                torch.randn(1, num_patches, patches.shape[2], device=device)  # GPU 
            )

        # Ajoute l’embedding de position aux patchs
        patches = patches + self.positional_embedding

        # Passage dans l’encodeur Transformer
        encoded_patches = self.transformer(patches)

        # Reconstruction finale de l’image
        x = self.decoder(encoded_patches, h_patches, w_patches,
                 target_h=self.n_lat_out, target_w=self.n_lon_out)
        return x
