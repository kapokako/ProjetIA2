import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis

###############################################################################
# PARAMÈTRES GLOBAUX
###############################################################################
TICKERS = ["AAPL", "MSFT", "GOOGL"]
START_DATE = "2015-01-01"
END_DATE = "2022-12-31"

SEQ_LENGTH = 50  # Longueur des séquences temporelles
BATCH_SIZE = 64
NZ = 32  # Dimension du bruit latent (augmenté)
HIDDEN_DIM = 128  # Taille des couches LSTM (augmenté)
NUM_LAYERS = 2  # On utilise 2 couches LSTM pour plus de profondeur
LR = 0.0002
BETA1 = 0.5
NUM_EPOCHS = 5000  # Entraînement long
ALPHA_CORR = 5.0  # Coefficient pénalité de corrélation
ALPHA_VOL = 2.0  # Coefficient pénalité de volatilité
ALPHA_KURT = 2.0  # Coefficient pénalité de kurtosis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)


###############################################################################
# 1. RÉCUPÉRATION & PRÉTRAITEMENT DES DONNÉES
###############################################################################
def download_data(tickers, start, end):
    dfs = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)
        df = df[['Close']].rename(columns={'Close': t})
        dfs.append(df)
    data_merged = dfs[0].join(dfs[1:], how='inner')
    data_merged.dropna(inplace=True)
    return data_merged


data_all = download_data(TICKERS, START_DATE, END_DATE)
print("Shape data_all :", data_all.shape)
print(data_all.head())

# Normalisation par actif
scalers = {}
for col in data_all.columns:
    sc = MinMaxScaler(feature_range=(-1, 1))
    data_all[col] = sc.fit_transform(data_all[[col]])
    scalers[col] = sc

arr_all = data_all.values  # (N, n_assets)
n_assets = arr_all.shape[1]

# Séquences temporelles
sequences = []
for i in range(len(arr_all) - SEQ_LENGTH):
    window = arr_all[i: i + SEQ_LENGTH, :]  # (SEQ_LENGTH, n_assets)
    sequences.append(window)
sequences = np.array(sequences)  # (N-seq_length, SEQ_LENGTH, n_assets)
print("Shape des séquences :", sequences.shape)

tensor_data = torch.tensor(sequences, dtype=torch.float32)  # (num_samples, SEQ_LENGTH, n_assets)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

###############################################################################
# 2. STATISTIQUES RÉELLES (CORRÉLATION, VOLATILITÉ, KURTOSIS)
###############################################################################
# On calcule la corrélation empirique sur arr_all (N, n_assets)
corr_empirique = np.corrcoef(arr_all.T)
corr_empirique_torch = torch.tensor(corr_empirique, dtype=torch.float32, device=device)


# On calcule la volatilité et la kurtosis des rendements pour chaque actif
# sur la totalité de la période
def compute_returns(prices_1d):
    # prices_1d : (N,) => rendements dimension (N-1,)
    return (prices_1d[1:] - prices_1d[:-1]) / (prices_1d[:-1] + 1e-8)


real_vols = []
real_kurts = []
for idx, col in enumerate(data_all.columns):
    # On repasse en prix réels
    normalized_prices = data_all[col].values
    sc = scalers[col]
    real_prices = sc.inverse_transform(normalized_prices.reshape(-1, 1)).flatten()
    rets = compute_returns(real_prices)
    vol = np.std(rets)
    k = kurtosis(rets, fisher=False)  # fisher=False => kurtosis non centrée
    real_vols.append(vol)
    real_kurts.append(k)

real_vols = np.array(real_vols, dtype=np.float32)
real_kurts = np.array(real_kurts, dtype=np.float32)
real_vols_torch = torch.tensor(real_vols, device=device)
real_kurts_torch = torch.tensor(real_kurts, device=device)


###############################################################################
# 3. FONCTIONS DE PÉNALITÉ
###############################################################################
def correlation_matrix_torch(X):
    """
    X: (batch_size, seq_length, n_assets).
    Concatène batch+temps => (B*T, n_assets), calcule la corrélation.
    """
    b, t, d = X.shape
    flatten = X.reshape(b * t, d)
    mean = torch.mean(flatten, dim=0, keepdim=True)
    fc = flatten - mean
    cov = (fc.T @ fc) / (fc.shape[0] - 1)
    var = torch.diag(cov)
    std = torch.sqrt(var + 1e-8)
    outer_std = std.unsqueeze(0) * std.unsqueeze(1)
    corr = cov / outer_std
    return corr


def correlation_penalty(generated):
    corr_gen = correlation_matrix_torch(generated)
    mse_corr = torch.mean((corr_gen - corr_empirique_torch) ** 2)
    return ALPHA_CORR * mse_corr


def compute_batch_returns(generated):
    """
    generated: (batch_size, seq_length, n_assets)
    On calcule les rendements journaliers par actif pour tout le batch.
    => shape (batch_size, seq_length-1, n_assets)
    """
    # r[t] = (p[t] - p[t-1]) / p[t-1]
    # On shift le tensuer sur l'axe seq_length
    p1 = generated[:, 1:, :]
    p0 = generated[:, :-1, :]
    rets = (p1 - p0) / (p0 + 1e-8)
    return rets


def stats_penalty(generated):
    """
    On compare la volatilité et la kurtosis moyennes par actif entre le fake et le réel.
    On calcule la volatilité et la kurtosis sur l'ensemble du batch.
    """
    # On reshape pour tout concaténer
    # generated shape (b, seq_length, n_assets)
    # On l'inverse transform pour repasser en prix réels => approximatif (on fait un asset par asset).
    # => pour la pénalité, on fait un "mini-batch" approximation. On peut le faire plus précisément
    #   en générant un grand échantillon hors training loop, mais ici on le fait "on the fly".

    b, t, d = generated.shape
    # On va "approx" inverse transform
    # On boucle sur chaque actif => on pourrait vectoriser, mais pour la lisibilité, on boucle
    # => on repasse en CPU pour l'inverse transform
    gen_cpu = generated.detach().cpu().numpy()
    real_space = np.zeros_like(gen_cpu)
    for asset_idx, col in enumerate(data_all.columns):
        sc = scalers[col]
        # Valeurs normalisées => shape (b*t, )
        vals = gen_cpu[:, :, asset_idx].reshape(-1, 1)
        inv_vals = sc.inverse_transform(vals)
        inv_vals = inv_vals.reshape(b, t)
        real_space[:, :, asset_idx] = inv_vals

    # On calcule la volatilité + kurtosis pour chaque actif en concaténant batch & temps
    # => On calcule rendements par échantillon
    rets_list = []
    for i in range(b):
        for j in range(d):
            prices_ij = real_space[i, :, j]
            r_ij = compute_returns(prices_ij)
            rets_list.append((j, r_ij))  # on stocke l'asset j

    # On regroupe par actif
    from collections import defaultdict
    asset_rets = defaultdict(list)
    for (asset_j, r_ij) in rets_list:
        asset_rets[asset_j].extend(r_ij)

    # On calcule la volatilité + kurtosis
    vol_pen = 0.0
    kurt_pen = 0.0
    for j in range(d):
        arr_r = np.array(asset_rets[j])
        if len(arr_r) < 2:
            continue
        vol_fake = np.std(arr_r)
        kurt_fake = kurtosis(arr_r, fisher=False)

        # Diff vs réel
        diff_vol = (vol_fake - real_vols[j]) ** 2
        diff_kurt = (kurt_fake - real_kurts[j]) ** 2

        vol_pen += diff_vol
        kurt_pen += diff_kurt

    # Moyenne sur d actifs
    vol_pen /= d
    kurt_pen /= d

    # On fait la somme pondérée
    penalty = ALPHA_VOL * vol_pen + ALPHA_KURT * kurt_pen
    return torch.tensor(penalty, dtype=torch.float32, device=device)


###############################################################################
# 4. DÉFINITION DU GAN (LSTM à 2 couches)
###############################################################################
class LSTMGenerator(nn.Module):
    def __init__(self, nz, hidden_dim, n_assets, num_layers=1):
        super().__init__()
        self.nz = nz
        self.hidden_dim = hidden_dim
        self.n_assets = n_assets
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=nz, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_assets)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # z: (batch_size, seq_length, nz)
        lstm_out, _ = self.lstm(z)  # (b, seq_length, hidden_dim)
        out = self.fc(lstm_out)  # (b, seq_length, n_assets)
        out = self.tanh(out)
        return out


class LSTMDiscriminator(nn.Module):
    def __init__(self, hidden_dim, n_assets, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_assets, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (b, seq_length, n_assets)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # (b, hidden_dim)
        logit = self.fc(last_out)
        prob = self.sigmoid(logit)
        return prob


netG = LSTMGenerator(NZ, HIDDEN_DIM, n_assets, num_layers=NUM_LAYERS).to(device)
netD = LSTMDiscriminator(HIDDEN_DIM, n_assets, num_layers=NUM_LAYERS).to(device)

optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
criterion = nn.BCELoss()

###############################################################################
# 5. BOUCLE D'ENTRAÎNEMENT (LONGUE)
###############################################################################
print("Démarrage de l'entraînement...")

for epoch in range(NUM_EPOCHS):
    for i, (real_batch,) in enumerate(dataloader):
        real_batch = real_batch.to(device)
        b_size = real_batch.size(0)

        # 1) Mise à jour Discriminateur
        netD.zero_grad()
        label_real = torch.full((b_size, 1), 1.0, device=device)
        output_real = netD(real_batch)
        lossD_real = criterion(output_real, label_real)

        noise = torch.randn(b_size, SEQ_LENGTH, NZ, device=device)
        fake_data = netG(noise)

        label_fake = torch.full((b_size, 1), 0.0, device=device)
        output_fake = netD(fake_data.detach())
        lossD_fake = criterion(output_fake, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # 2) Mise à jour Générateur
        netG.zero_grad()
        label_gen = torch.full((b_size, 1), 1.0, device=device)
        output_gen = netD(fake_data)
        lossG_adv = criterion(output_gen, label_gen)

        # Pénalités
        loss_corr = correlation_penalty(fake_data)
        loss_stats = stats_penalty(fake_data)

        lossG = lossG_adv + loss_corr + loss_stats
        lossG.backward()
        optimizerG.step()

    # Affichage toutes les 200 époques
    if (epoch + 1) % 200 == 0:
        print(
            f"[Epoch {epoch + 1}/{NUM_EPOCHS}] LossD: {lossD.item():.4f} | LossG_adv: {lossG_adv.item():.4f} | Corr: {loss_corr.item():.4f} | Stats: {loss_stats.item():.4f}")

print("Entraînement terminé.")

###############################################################################
# 6. GÉNÉRATION MULTIPLE & COMPARAISONS
###############################################################################
netG.eval()


def inverse_transform_multi(fake_samples_np):
    """
    fake_samples_np: (batch_size, seq_length, n_assets)
    Retour: identique en prix réels
    """
    b, t, d = fake_samples_np.shape
    real_space = np.zeros_like(fake_samples_np)
    for asset_idx, col in enumerate(data_all.columns):
        sc = scalers[col]
        vals = fake_samples_np[:, :, asset_idx].reshape(-1, 1)
        inv_vals = sc.inverse_transform(vals)
        inv_vals = inv_vals.reshape(b, t)
        real_space[:, :, asset_idx] = inv_vals
    return real_space


# 6.1 Générer un échantillon conséquent pour la comparaison
NUM_GEN_SAMPLES = 100  # Générer 100 séquences
with torch.no_grad():
    noise_test = torch.randn(NUM_GEN_SAMPLES, SEQ_LENGTH, NZ, device=device)
    fake_samples = netG(noise_test).cpu().numpy()  # (NUM_GEN_SAMPLES, SEQ_LENGTH, n_assets)

fake_samples_real = inverse_transform_multi(fake_samples)

# 6.2 Comparaison de corrélation
concat_fake = fake_samples_real.reshape(-1, n_assets)  # (NUM_GEN_SAMPLES*SEQ_LENGTH, n_assets)
corr_fake = np.corrcoef(concat_fake.T)

print("\n--- COMPARAISON CORRÉLATION ---")
print("Corrélation empirique :\n", corr_empirique)
print("\nCorrélation fake :\n", corr_fake)


# 6.3 Distribution des rendements sur un actif (ex: 1er ticker)
def get_all_returns(prices_2d):
    # prices_2d : (NUM_GEN_SAMPLES, SEQ_LENGTH)
    # On concatène tous les rendements
    rets_list = []
    for i in range(NUM_GEN_SAMPLES):
        r = compute_returns(prices_2d[i, :])
        rets_list.extend(r)
    return np.array(rets_list)


asset_idx = 0  # ex. AAPL
all_fake_returns = get_all_returns(fake_samples_real[:, :, asset_idx])

# 6.4 Distribution des rendements réels
normalized_prices = data_all[data_all.columns[asset_idx]].values
sc = scalers[data_all.columns[asset_idx]]
real_prices = sc.inverse_transform(normalized_prices.reshape(-1, 1)).flatten()
all_real_returns = compute_returns(real_prices)

plt.figure(figsize=(10, 5))
plt.hist(all_real_returns, bins=50, alpha=0.5, label="Rendements réels")
plt.hist(all_fake_returns, bins=50, alpha=0.5, label="Rendements générés (100 séquences)")
plt.title(f"Distribution des rendements - {TICKERS[asset_idx]}")
plt.legend()
plt.show()

print("\nAnalyse terminée. Vous pouvez comparer la corrélation, la distribution des rendements, etc.")