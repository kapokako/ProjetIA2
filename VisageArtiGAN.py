import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# Chemin vers le dossier contenant les images CelebA
data_dir = r"D:\ProjetIA2\data\celeba\celeba\img_align_celeba\img_align_celeba"


# Dataset personnalisé pour CelebA
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Liste de tous les fichiers image dans le dossier
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == '__main__':
    # Paramètres
    image_size = 64  # Redimensionnement à 64x64
    nc = 3  # Nombre de canaux (RGB)
    nz = 100  # Dimension du vecteur latent
    ngf = 64  # Taille des feature maps dans le générateur
    ndf = 64  # Taille des feature maps dans le discriminateur
    num_epochs = 1  # Nombre d'époques (à augmenter pour de meilleurs résultats)
    lr = 0.0002
    beta1 = 0.5
    batch_size = 128

    # Transformation sur les images CelebA
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # Recadrage central pour CelebA
        transforms.Resize(image_size),  # Redimensionnement à 64x64
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation [-1,1]
    ])

    # Chargement du dataset
    dataset = CelebADataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Définition du device (GPU si disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé :", device)


    # Définition du Générateur (DCGAN)
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            return self.main(input)


    # Définition du Discriminateur (DCGAN)
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input).view(-1, 1)


    # Instanciation des modèles et initialisation des poids
    netG = Generator().to(device)
    netD = Discriminator().to(device)


    def weights_init(m):
        classname = m.__class__.__name__
        if "Conv" in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif "BatchNorm" in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    netG.apply(weights_init)
    netD.apply(weights_init)

    # Définition de la fonction de perte et des optimisateurs
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Création d'un tenseur fixe pour la visualisation
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Boucle d'entraînement
    print("Début de l'entraînement DCGAN sur CelebA...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Mise à jour du Discriminateur
            ############################
            netD.zero_grad()
            real_images = data.to(device)
            b_size = real_images.size(0)
            label = torch.full((b_size, 1), 1.0, device=device)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(0.0)
            output = netD(fake_images.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()
            errD = errD_real + errD_fake

            ############################
            # (2) Mise à jour du Générateur
            ############################
            netG.zero_grad()
            label.fill_(1.0)
            output = netD(fake_images)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(
                    f"[{epoch + 1}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")

        # Sauvegarde d'un échantillon d'images générées à la fin de chaque époque
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"output_images/epoch_{epoch + 1}.png", normalize=True)

    print("Entraînement terminé.")

    # Visualisation des images générées (optionnel)
    import matplotlib.pyplot as plt

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Exemples de visages générés")
    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True).numpy(), (1, 2, 0)))
    plt.show()
