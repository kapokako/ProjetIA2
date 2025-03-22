import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

###############################################################################
# 1. CHARGEMENT DES DONNÉES DE PLUSIEURS ACTIFS VIA YFINANCE
###############################################################################

# Exemple : 3 tickers pour illustrer la corrélation (AAPL, MSFT, GOOGL)
tickers = ["AAPL", "MSFT", "GOOGL"]
start_date = "2018-01-01"
end_date = "2023-01-01"

data_frames = []
for t in tickers:
    df = yf.download(t, start=start_date, end=end_date, progress=False)
    df = df[["Close"]].rename(columns={"Close": t})
    data_frames.append(df)

# Fusionner sur la base des dates communes (inner join)
data_all = data_frames[0].join(data_frames[1:], how="inner")
data_all.dropna(inplace=True)  # Supprime les dates avec valeurs manquantes

# data_all est maintenant un DataFrame avec colonnes [AAPL, MSFT, GOOGL]

###############################################################################
# 2. PRÉTRAITEMENT : NORMALISATION & CRÉATION DE SÉQUENCES TEMPORELLES
###############################################################################

# On va normaliser chaque colonne (chaque actif) sur [-1, 1].
# Puis on va créer des séquences temporelles de longueur seq_length.
seq_length = 50

# MinMaxScaler sur chaque colonne indépendamment (on veut [-1,1] par actif)
scalers = {}
for col in data_all.columns:
    scalers[col] = MinMaxScaler(feature_range=(-1, 1))
    data_all[col] = scalers[col].fit_transform(data_all[[col]])

# data_all est un DataFrame (N, 3). On va créer un tenseur 3D : (N - seq_length, n_assets, seq_length).
arr_all = data_all.values  # shape (N, 3)
n_assets = arr_all.shape[1]

sequences = []
for i in range(len(arr_all) - seq_length):
    # segment de taille seq_length
    window = arr_all[i: i + seq_length, :]  # shape (seq_length, n_assets)
    # On veut la forme (n_assets, seq_length) pour le réseau
    window = window.T  # shape (n_assets, seq_length)
    sequences.append(window)

sequences = np.array(sequences)  # (N - seq_length, n_assets, seq_length)
print("Shape des séquences :", sequences.shape)

# Conversion en tenseur PyTorch
tensor_data = torch.tensor(sequences, dtype=torch.float32)  # (N, n_assets, seq_length)

# DataLoader
batch_size = 128
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

###############################################################################
# 3. MATRICE DE CORRÉLATION CIBLE (EMPIRIQUE) & FONCTION DE PÉNALITÉ
###############################################################################

# On calcule la corrélation empirique globale sur tout l'historique complet
# (pour l'instant, on calcule la corrélation sur la matrice (N, n_assets),
#  c'est-à-dire sur la totalité de la période).
corr_empirique = np.corrcoef(arr_all.T)  # shape (n_assets, n_assets)
corr_empirique_torch = torch.tensor(corr_empirique, dtype=torch.float32)


def correlation_matrix_torch(X):
    """
    Calcule la matrice de corrélation d'un tenseur X de forme (batch_size, n_assets, seq_length)
    en aplatisant sur le batch et le temps.
    Sortie : corr (n_assets, n_assets)
    """
    # On veut vectoriser par actif. On va concaténer batch et seq_length => dimension B*T
    # shape: (batch_size, n_assets, seq_length) => (n_assets, batch_size*seq_length)
    n_assets = X.shape[1]
    flatten = X.permute(1, 0, 2).reshape(n_assets, -1)  # (n_assets, B*T)
    # Calculer la corrélation
    # corrcoef = (cov / sqrt(var*var)) => on peut s'appuyer sur la fonction builtin
    # ou faire un calcul manuel. Pytorch n'a pas de corrcoef direct, on le fait manuellement.

    # Centrage
    mean = torch.mean(flatten, dim=1, keepdim=True)  # (n_assets, 1)
    flatten_centered = flatten - mean
    # Covariance
    cov = (flatten_centered @ flatten_centered.T) / (flatten_centered.shape[1] - 1)  # shape (n_assets, n_assets)
    # Variance diag
    var = torch.diag(cov)
    # On évite la division par zéro
    std = torch.sqrt(var + 1e-8)
    outer_std = std.unsqueeze(0) * std.unsqueeze(1)
    corr = cov / outer_std
    return corr


def correlation_penalty(generated, alpha=10.0):
    """
    Compare la matrice de corrélation calculée sur 'generated' à la corrélation empirique.
    Renvoie un scalaire de pénalité, plus il est grand, plus la corrélation est éloignée.
    alpha : coefficient de pondération.
    """
    corr_gen = correlation_matrix_torch(generated)
    # MSE entre corr_gen et corr_empirique
    mse_corr = torch.mean((corr_gen - corr_empirique_torch.to(generated.device)) ** 2)
    return alpha * mse_corr


###############################################################################
# 4. DÉFINITION DU GAN MULTI-ACTIFS (CORR-GAN SIMPLIFIÉ)
###############################################################################

nz = 20  # dimension du bruit
hidden_dim = 64  # dimension cachée
lr = 0.0002
beta1 = 0.5
num_epochs = 2000  # plus d'époques pour améliorer la fidélité


# Générateur : doit produire (batch_size, n_assets, seq_length) à partir d'un bruit (batch_size, nz)
class GeneratorCorr(nn.Module):
    def __init__(self, nz, hidden_dim, n_assets, seq_length):
        super().__init__()
        self.n_assets = n_assets
        self.seq_length = seq_length

        # On part sur un MLP simple, qui produit n_assets*seq_length en sortie
        self.net = nn.Sequential(
            nn.Linear(nz, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, n_assets * seq_length),
            nn.Tanh()  # pour rester dans [-1,1]
        )

    def forward(self, z):
        out = self.net(z)  # shape (batch_size, n_assets*seq_length)
        out = out.view(z.size(0), self.n_assets, self.seq_length)  # (batch_size, n_assets, seq_length)
        return out


# Discriminateur : prend (batch_size, n_assets, seq_length) => sort probabilité
class DiscriminatorCorr(nn.Module):
    def __init__(self, n_assets, seq_length, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # => (batch_size, n_assets*seq_length)
            nn.Linear(n_assets * seq_length, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = GeneratorCorr(nz, hidden_dim, n_assets, seq_length).to(device)
netD = DiscriminatorCorr(n_assets, seq_length, hidden_dim).to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

###############################################################################
# 5. BOUCLE D'ENTRAÎNEMENT AVEC PÉNALITÉ DE CORRÉLATION
###############################################################################

for epoch in range(num_epochs):
    for i, (real_batch,) in enumerate(dataloader):
        b_size = real_batch.size(0)
        real_batch = real_batch.to(device)  # shape (batch_size, n_assets, seq_length)

        # ---------------------------
        # (1) Mise à jour du Discriminateur
        # ---------------------------
        netD.zero_grad()

        # Vrai
        label_real = torch.full((b_size, 1), 1.0, device=device)
        output_real = netD(real_batch)
        lossD_real = criterion(output_real, label_real)

        # Faux
        noise = torch.randn(b_size, nz, device=device)
        fake_data = netG(noise)
        label_fake = torch.full((b_size, 1), 0.0, device=device)
        output_fake = netD(fake_data.detach())
        lossD_fake = criterion(output_fake, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # ---------------------------
        # (2) Mise à jour du Générateur (avec pénalité corrélation)
        # ---------------------------
        netG.zero_grad()
        label_gen = torch.full((b_size, 1), 1.0, device=device)
        output_gen = netD(fake_data)
        lossG_basic = criterion(output_gen, label_gen)

        # On ajoute un terme de pénalité de corrélation
        loss_corr = correlation_penalty(fake_data, alpha=10.0)

        lossG_total = lossG_basic + loss_corr
        lossG_total.backward()
        optimizerG.step()

    # Affichage
    if (epoch + 1) % 200 == 0:
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] LossD: {lossD.item():.4f} | LossG: {lossG_basic.item():.4f} | CorrPenalty: {loss_corr.item():.4f}")

###############################################################################
# 6. GÉNÉRATION DE NOUVELLES SÉRIES ET VISUALISATION
###############################################################################

netG.eval()
with torch.no_grad():
    sample_noise = torch.randn(5, nz, device=device)
    fake_samples = netG(sample_noise).cpu().numpy()  # shape (5, n_assets, seq_length)


# Inverse transform : on repasse dans l'espace des prix
# Rappel : on a un scaler par actif
# => On va faire la même transformation inverse pour chaque actif
# fake_samples[i, asset, t]
def inverse_transform_multi(fake_samples):
    # fake_samples : (batch_size, n_assets, seq_length)
    # On doit faire la transfo inverse asset par asset, sur chaque batch
    batch_size, n_assets, seq_length = fake_samples.shape
    real_space = np.zeros_like(fake_samples)
    for asset_idx, col in enumerate(data_all.columns):
        # On récupère le scaler de cet actif
        sc = scalers[col]
        # On récupère toutes les valeurs associées à cet asset
        # shape (batch_size, seq_length)
        vals = fake_samples[:, asset_idx, :]
        # On reshape en (batch_size*seq_length, 1)
        vals_reshape = vals.reshape(-1, 1)
        inv_vals = sc.inverse_transform(vals_reshape)
        # On remet en forme
        inv_vals = inv_vals.reshape(batch_size, seq_length)
        real_space[:, asset_idx, :] = inv_vals
    return real_space


fake_samples_real_scale = inverse_transform_multi(fake_samples)

# Visualisation
# On va tracer 2 ou 3 courbes (batch_size=5 => on en prend 3)
fig, axes = plt.subplots(3, n_assets, figsize=(4 * n_assets, 4 * 3))
fig.suptitle("Séries temporelles multi-actifs générées (CorrGAN simplifié)")

for i in range(3):  # 3 exemples
    for j in range(n_assets):
        axes[i, j].plot(fake_samples_real_scale[i, j, :], label=f"{data_all.columns[j]}")
        axes[i, j].set_title(f"Exemple {i}, Actif {data_all.columns[j]}")
        axes[i, j].grid(True)

plt.tight_layout()
plt.show()
