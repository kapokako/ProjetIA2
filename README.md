# Génération de Séries Temporelles Financières et de Visages Artificiels avec GANs

Ce projet explore l'utilisation de Réseaux Adversariaux Génératifs (GANs) dans différents domaines :
- **Finance** : génération de séries temporelles financières réalistes avec plusieurs approches (GAN_Finance, CorrGAN, QuantGAN).
- **Visages Artificiels** : génération de visages artificiels à partir du dataset CelebA (inspiré de thispersondoesnotexist.com).

Le projet regroupe plusieurs implémentations, chacune adaptée à un cas d'usage spécifique.

---

## Table des Matières

- [Description du Projet](#description-du-projet)
- [Structure du Répertoire](#structure-du-répertoire)
- [Installation et Dépendances](#installation-et-dépendances)
- [Références](#références)

---

## Description du Projet

L'objectif de ce projet est d'explorer plusieurs applications des GANs :

- **Finance** : Générer des séries temporelles financières réalistes en reproduisant la dynamique temporelle, les corrélations entre actifs et les caractéristiques statistiques (rendements, volatilité, queues de distribution).
  - *GAN_Finance* génère des séries pour un seul actif à l'aide d'un MLP.
  - *CorrGAN* génère des séries multi-actifs et intègre une pénalité de corrélation pour respecter la structure de dépendance entre actifs.
  - *QuantGAN* utilise la fonction caractéristique des rendements pour forcer le générateur à reproduire la distribution réelle (volatilité, queues).

- **Visages Artificiels** : Générer des images de visages réalistes (inspirés de sites comme thispersondoesnotexist.com) en entraînant un DCGAN sur le dataset CelebA.

---

## Structure du Répertoire

- **mainFin.py** : GAN_Finance – génération de séries temporelles pour un actif unique (ex. AAPL).
- **CorrGAN.py** : CorrGAN – génération de séries multi-actifs avec pénalité de corrélation.
- **QuantGan.py** : QuantGAN – génération de séries multi-actifs avec pénalité basée sur la fonction caractéristique.
- **GANavance.py** : GAN - génération de séries multi-actifs avec corrélation, volatilité, kurtosis.
- **VisageArtiGAN.py** : DCGAN pour la génération de visages artificiels à partir du dataset CelebA.
- **main.py** : Exemple classique de GAN appliqué au dataset MNIST (pour illustration de la structure GAN).
- **generated_series.csv** : Exemple de série temporelle générée (si applicable).

---

## Installation et Dépendances

### Prérequis

- Python 3.7 ou supérieur
- GPU avec CUDA (optionnel, fortement recommandé pour l'entraînement des GANs)

### Installation

Installez les dépendances nécessaires via pip :

```bash
pip install torch torchvision yfinance pandas numpy scikit-learn matplotlib scipy
```
## Références
- Goodfellow et al. (2014) – Introduction aux GANs.

- Documentation de PyTorch.

- yfinance – Pour la récupération des données financières.

- Articles et ressources sur CorrGAN et QuantGAN.

- Dataset CelebA – pour la génération de visages artificiels.
