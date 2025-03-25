# GAN_MNIST.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

# Paramètres
batch_size = 128
image_size = 28
nz = 100  # dimension du bruit (latent vector)
ngf = 64  # taille de feature maps dans le générateur
ndf = 64  # taille de feature maps dans le discriminateur
num_epochs = 5
lr = 0.0002
beta1 = 0.5  # paramètre pour Adam

# Création du répertoire de sauvegarde
os.makedirs("output_images", exist_ok=True)

# Transformation et chargement du jeu de données MNIST
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Définition du dispositif
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Définition du générateur: Transforme un bruit aléatoire en image (28x28)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # couche d'entrée: (nz) -> (ngf*4 x 7 x 7)
            nn.Linear(nz, ngf * 4 * 7 * 7),
            nn.BatchNorm1d(ngf * 4 * 7 * 7),
            nn.ReLU(True),
            # Reshape en (ngf*4, 7, 7)
            nn.Unflatten(1, (ngf * 4, 7, 7)),
            # Convolution transposée pour passer de 7x7 à 14x14
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Convolution transposée pour passer de 14x14 à 28x28
            nn.ConvTranspose2d(ngf * 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # sortie normalisée entre -1 et 1
        )

    def forward(self, input):
        return self.main(input)


# Définition du discriminateur : Distingue vraies et fausses images
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Entrée: (1, 28, 28)
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf, 14, 14)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2, 7, 7)
            nn.Flatten(),
            nn.Linear(ndf * 2 * 7 * 7, 1),
            nn.Sigmoid()  # probabilité de réel
        )

    def forward(self, input):
        return self.main(input)


# Initialisation des modèles
netG = Generator().to(device)
netD = Discriminator().to(device)

# Fonction de perte et optimisateurs
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Boucle d'entraînement
print("Début de l'entraînement...")
for epoch in range(num_epochs):
    for i, (data, _) in enumerate(dataloader):
        # Mise à jour du discriminateur avec les vrais exemples
        netD.zero_grad()
        real_images = data.to(device)
        b_size = real_images.size(0)
        label_real = torch.full((b_size, 1), 1.0, device=device)
        output_real = netD(real_images)
        lossD_real = criterion(output_real, label_real)
        lossD_real.backward()

        # Mise à jour du discriminateur avec les faux exemples générés
        noise = torch.randn(b_size, nz, device=device)
        fake_images = netG(noise)
        label_fake = torch.full((b_size, 1), 0.0, device=device)
        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, label_fake)
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake
        optimizerD.step()

        # Mise à jour du générateur : on veut que le discriminateur se trompe
        netG.zero_grad()
        label_gen = torch.full((b_size, 1), 1.0, device=device)  # on veut "tromper" le discriminateur
        output_gen = netD(fake_images)
        lossG = criterion(output_gen, label_gen)
        lossG.backward()
        optimizerG.step()

        # Affichage toutes les 100 itérations de la progression
        if i % 100 == 0:
            print(
                f"[{epoch + 1}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

    # Sauvegarde d'images générées après chaque époque
    with torch.no_grad():
        fixed_noise = torch.randn(64, nz, device=device)
        fake = netG(fixed_noise).detach().cpu()
    vutils.save_image(fake, f"output_images/epoch_{epoch + 1}.png", normalize=True)

print("Entraînement terminé.")
