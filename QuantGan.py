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
TICKERS = ["AAPL", "MSFT", "GOOGL"]  # multi-actifs
START_DATE = "2015-01-01"
END_DATE = "2022-12-31"

SEQ_LENGTH = 50  # Longueur des séquences temporelles
BATCH_SIZE = 64
NZ = 32  # Dimension du bruit latent
HIDDEN_DIM = 128  # Taille des couches LSTM
NUM_LAYERS = 1  # Nombre de couches LSTM
LR = 0.0002
BETA1 = 0.5
NUM_EPOCHS = 1000  # Entraînement (à adapter selon tes ressources)
ALPHA_CHARFUNC = 1.0  # Pondération de la pénalité QuantGAN (carac. function)

# Paramètres de la fonction caractéristique
# On évalue la CF à plusieurs fréquences (positives & négatives)
FREQS = np.linspace(-10, 10, 21)  # ex. 21 fréquences entre -10 et 10
# Pour un usage plus fin, on peut mettre plus de fréquences.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)


###############################################################################
# 1. CHARGEMENT & PRÉTRAITEMENT DES DONNÉES
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

# Normalisation par actif ([-1,1])
scalers = {}
for col in data_all.columns:
    sc = MinMaxScaler(feature_range=(-1, 1))
    data_all[col] = sc.fit_transform(data_all[[col]])
    scalers[col] = sc

arr_all = data_all.values  # shape (N, n_assets)
n_assets = arr_all.shape[1]

# Séquences temporelles
sequences = []
for i in range(len(arr_all) - SEQ_LENGTH):
    window = arr_all[i: i + SEQ_LENGTH, :]
    sequences.append(window)
sequences = np.array(sequences)  # (N-seq_length, SEQ_LENGTH, n_assets)
print("Shape des séquences :", sequences.shape)

tensor_data = torch.tensor(sequences, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


###############################################################################
# 2. CALCUL DES RENDEMENTS & FONCTION CARACTÉRISTIQUE (POUR LES DONNÉES RÉELLES)
###############################################################################
def compute_returns(prices):
    # prices: (T,) => rendements shape (T-1,)
    return (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)


def char_func_1d(returns, freqs):
    """
    Calcule la fonction caractéristique univariée:
      CF(freq) = E[e^{i * freq * returns}]
    returns: shape (M,)
    freqs: np.array de fréquences
    Retour: complex array, shape (len(freqs),)
    """
    # e^{i * freq * r} => real + i * imag
    cf_vals = []
    for w in freqs:
        # On calcule e^{i*w*r} sur tous les rendements, puis la moyenne
        exps = np.exp(1j * w * returns)
        cf_vals.append(np.mean(exps))
    return np.array(cf_vals)


# On calcule la fonction caractéristique globale des rendements réels
#   -> Soit on concatène tous les rendements, tous actifs confondus
#   -> Soit on fait la moyenne des CF de chaque actif. Ici, on choisit la 2e option
#      pour tenir compte de chaque actif séparément.

# 1) Inverse transform pour repasser en prix réels
prices_real = {}
for idx, col in enumerate(data_all.columns):
    norm_vals = data_all[col].values
    inv_vals = scalers[col].inverse_transform(norm_vals.reshape(-1, 1)).flatten()
    prices_real[col] = inv_vals

# 2) On calcule la CF moyenne sur tous les rendements journaliers
#    On concatène tous les rendements journaliers de l'historique complet
all_cf = np.zeros(len(FREQS), dtype=np.complex128)
count_assets = 0

for col in data_all.columns:
    inv_vals = prices_real[col]
    rets = compute_returns(inv_vals)  # shape (N-1,)
    # char func
    cf_col = char_func_1d(rets, FREQS)
    all_cf += cf_col
    count_assets += 1

all_cf /= count_assets  # on fait la moyenne
real_cf_torch = torch.tensor(np.stack([all_cf.real, all_cf.imag], axis=1), dtype=torch.float32, device=device)


# shape (len(FREQS), 2) => on stocke (real, imag)

###############################################################################
# 3. FONCTION CARACTÉRISTIQUE SUR LE FAKE (PÉNALITÉ QUANTGAN)
###############################################################################
def inverse_transform_batch(fake_samples):
    """
    fake_samples: (batch_size, seq_length, n_assets) in [-1,1].
    On repasse en prix réels, CPU => on renvoie un np.array
    """
    b, t, d = fake_samples.shape
    fake_cpu = fake_samples.detach().cpu().numpy()
    real_space = np.zeros_like(fake_cpu)
    for asset_idx, col in enumerate(data_all.columns):
        sc = scalers[col]
        vals = fake_cpu[:, :, asset_idx].reshape(-1, 1)
        inv_vals = sc.inverse_transform(vals)
        inv_vals = inv_vals.reshape(b, t)
        real_space[:, :, asset_idx] = inv_vals
    return real_space


def compute_char_func_penalty(fake_samples, alpha=ALPHA_CHARFUNC):
    """
    Calcule la pénalité QuantGAN = distance entre CF moyenne (fake) et CF moyenne (réel).
    - On concatène tous les rendements (batch, seq_length).
    - On calcule la CF (réelle+imag) pour ces rendements, puis on compare à real_cf_torch.
    """
    # 1) On repasse en prix réels
    real_space = inverse_transform_batch(fake_samples)  # shape (b, t, d)
    b, t, d = real_space.shape

    # 2) On concatène tous les rendements
    #    On calcule la CF univariée globale en concaténant rendements sur b et d
    #    (ici, pour simplifier, on concatène tout en un seul vecteur, comme si
    #     c'était un unique "actif", ce qui n'est pas parfaitement rigoureux
    #     pour un multi-actif. On pourrait aussi faire la moyenne sur les d actifs.)
    rets_all = []
    for i in range(b):
        for j in range(d):
            prices_ij = real_space[i, :, j]
            r_ij = compute_returns(prices_ij)
            rets_all.extend(r_ij)
    rets_all = np.array(rets_all)

    # 3) On calcule la CF sur rets_all pour les fréquences FREQS
    cf_vals = []
    for w in FREQS:
        exps = np.exp(1j * w * rets_all)
        cf_vals.append(np.mean(exps))
    cf_vals = np.array(cf_vals)  # shape (len(FREQS),) complex

    # On sépare real & imag
    cf_real = np.real(cf_vals)
    cf_imag = np.imag(cf_vals)
    cf_stack = np.stack([cf_real, cf_imag], axis=1)  # shape (len(FREQS), 2)

    # 4) On compare à real_cf_torch (len(FREQS), 2)
    cf_fake_torch = torch.tensor(cf_stack, dtype=torch.float32, device=device)
    diff = cf_fake_torch - real_cf_torch
    mse_cf = torch.mean(diff ** 2)

    return alpha * mse_cf


###############################################################################
# 4. ARCHITECTURE DU GAN (LSTM) + TRAINING
###############################################################################
class LSTMGenerator(nn.Module):
    def __init__(self, nz, hidden_dim, n_assets, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=nz, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_assets)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # z: (b, seq_length, nz)
        lstm_out, _ = self.lstm(z)
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
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        logit = self.fc(last_out)
        return self.sigmoid(logit)


netG = LSTMGenerator(NZ, HIDDEN_DIM, n_assets, num_layers=NUM_LAYERS).to(device)
netD = LSTMDiscriminator(HIDDEN_DIM, n_assets, num_layers=NUM_LAYERS).to(device)

optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
criterion = nn.BCELoss()

print("Début de l'entraînement QuantGAN ...")

for epoch in range(NUM_EPOCHS):
    for i, (real_batch,) in enumerate(dataloader):
        real_batch = real_batch.to(device)  # (b, seq_length, n_assets)
        b_size = real_batch.size(0)

        # 1) Discriminateur
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

        # 2) Générateur
        netG.zero_grad()
        label_gen = torch.full((b_size, 1), 1.0, device=device)
        output_gen = netD(fake_data)
        lossG_adv = criterion(output_gen, label_gen)

        # Pénalité QuantGAN : distance de la fonction caractéristique
        loss_char = compute_char_func_penalty(fake_data)

        lossG = lossG_adv + loss_char
        lossG.backward()
        optimizerG.step()

    if (epoch + 1) % 200 == 0:
        print(
            f"[Epoch {epoch + 1}/{NUM_EPOCHS}] LossD: {lossD.item():.4f} | LossG_adv: {lossG_adv.item():.4f} | CF: {loss_char.item():.4f}")

print("Entraînement terminé.")

###############################################################################
# 5. GÉNÉRATION ET COMPARAISONS
###############################################################################
netG.eval()
NUM_GEN_SAMPLES = 100
with torch.no_grad():
    noise_test = torch.randn(NUM_GEN_SAMPLES, SEQ_LENGTH, NZ, device=device)
    fake_samples = netG(noise_test).cpu().numpy()  # (NUM_GEN_SAMPLES, SEQ_LENGTH, n_assets)


# Convertir en prix réels
def inverse_transform_multi(fake_np):
    b, t, d = fake_np.shape
    real_space = np.zeros_like(fake_np)
    for asset_idx, col in enumerate(data_all.columns):
        sc = scalers[col]
        vals = fake_np[:, :, asset_idx].reshape(-1, 1)
        inv_vals = sc.inverse_transform(vals)
        inv_vals = inv_vals.reshape(b, t)
        real_space[:, :, asset_idx] = inv_vals
    return real_space


fake_prices = inverse_transform_multi(fake_samples)

# Exemple de visualisation : histogramme de rendements sur le 1er actif
asset_idx = 0
rets_fake_all = []
for i in range(NUM_GEN_SAMPLES):
    r = compute_returns(fake_prices[i, :, asset_idx])
    rets_fake_all.extend(r)
rets_fake_all = np.array(rets_fake_all)

# Rendements réels pour comparaison
real_prices_1 = scalers[data_all.columns[asset_idx]].inverse_transform(
    data_all.iloc[:, asset_idx].values.reshape(-1, 1)
).flatten()
rets_real_1 = compute_returns(real_prices_1)

plt.figure(figsize=(10, 5))
plt.hist(rets_real_1, bins=50, alpha=0.5, label="Rendements réels")
plt.hist(rets_fake_all, bins=50, alpha=0.5, label="Rendements générés (QuantGAN)")
plt.legend()
plt.title(f"Distribution rendements - {TICKERS[asset_idx]}")
plt.show()

print("Analyse QuantGAN terminée.")
