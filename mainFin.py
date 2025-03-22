# GAN_Finance_API.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Récupération des données via l'API yfinance
# Télécharger les données historiques de l'actif (ici AAPL) sur les 5 dernières années
ticker = "AAPL"
data = yf.download(ticker, period="5y")
data = data.sort_index()  # s'assurer que les données sont triées par date

# 2. Prétraitement des données
# On utilisera la colonne 'Close'
prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
prices_norm = scaler.fit_transform(prices)

# Création de séquences temporelles (exemple : séquences de 50 jours)
sequence_length = 50
sequences = []
for i in range(len(prices_norm) - sequence_length):
    sequences.append(prices_norm[i:i + sequence_length])
sequences = np.array(sequences)  # forme: (N, sequence_length, 1)

# Conversion en tenseur PyTorch et ajustement des dimensions (N, 1, sequence_length)
sequences_tensor = torch.tensor(sequences, dtype=torch.float32).permute(0, 2, 1)
batch_size = 128
dataset = TensorDataset(sequences_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 3. Définition du GAN pour séries temporelles
nz = 20  # dimension du bruit latent
hidden_dim = 64  # dimension cachée


class Generator(nn.Module):
    def __init__(self, nz, hidden_dim, sequence_length):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(nz, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, sequence_length),
            nn.Tanh()  # sortie entre -1 et 1
        )

    def forward(self, z):
        out = self.model(z)
        return out.unsqueeze(1)  # forme: (batch_size, 1, sequence_length)


class Discriminator(nn.Module):
    def __init__(self, sequence_length, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(nz, hidden_dim, sequence_length).to(device)
netD = Discriminator(sequence_length, hidden_dim).to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 4. Boucle d'entraînement du GAN
num_epochs = 500  # ajustez le nombre d'époques selon vos besoins
for epoch in range(num_epochs):
    for i, (data_batch,) in enumerate(dataloader):
        b_size = data_batch.size(0)
        data_batch = data_batch.to(device)

        # Mise à jour du discriminateur avec les vraies données
        netD.zero_grad()
        label_real = torch.full((b_size, 1), 1.0, device=device)
        output_real = netD(data_batch)
        lossD_real = criterion(output_real, label_real)

        # Génération de fausses données
        noise = torch.randn(b_size, nz, device=device)
        fake_data = netG(noise)
        label_fake = torch.full((b_size, 1), 0.0, device=device)
        output_fake = netD(fake_data.detach())
        lossD_fake = criterion(output_fake, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # Mise à jour du générateur
        netG.zero_grad()
        label_gen = torch.full((b_size, 1), 1.0, device=device)
        output_gen = netD(fake_data)
        lossG = criterion(output_gen, label_gen)
        lossG.backward()
        optimizerG.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

# 5. Visualisation des séries générées
netG.eval()
with torch.no_grad():
    sample_noise = torch.randn(10, nz, device=device)
    fake_series = netG(sample_noise).cpu().numpy()

plt.figure(figsize=(10, 6))
for series in fake_series:
    plt.plot(series.flatten(), alpha=0.8)
plt.title("Exemples de séries temporelles générées pour " + ticker)
plt.xlabel("Indice temporel")
plt.ylabel("Valeur normalisée")
plt.show()

# Optionnel : sauvegarder les données générées en CSV
fake_series_rescaled = scaler.inverse_transform(fake_series.reshape(-1, sequence_length)).reshape(fake_series.shape)
np.savetxt("generated_series.csv", fake_series_rescaled.reshape(-1, sequence_length), delimiter=",")
