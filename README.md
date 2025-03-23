# Génération de Séries Temporelles Financières avec GANs

Ce projet explore plusieurs approches de Réseaux Adversariaux Génératifs (GANs) appliquées à la finance. L’objectif est de générer des séries temporelles réalistes qui reproduisent la dynamique, les corrélations entre actifs et la distribution des rendements (moyenne, volatilité, queues, etc.) à partir de données historiques réelles.

Le projet inclut trois implémentations principales :

- **GAN_Finance** :  
  Une implémentation de base pour générer des séries temporelles univariées (par exemple, pour un actif unique comme AAPL) en utilisant un MLP.

- **CorrGAN** :  
  Un modèle GAN multi-actifs qui intègre une **pénalité de corrélation** afin de reproduire la structure de dépendance entre plusieurs actifs (par exemple, AAPL, MSFT et GOOGL).

- **QuantGAN** :  
  Une approche qui utilise la **fonction caractéristique** des rendements pour forcer le générateur à reproduire la distribution empirique des rendements (ce qui permet de mieux capturer la volatilité, les queues épaisses, etc.).

---

## Table des Matières

- [Description du Projet](#description-du-projet)
- [Structure du Répertoire](#structure-du-répertoire)
- [Installation et Dépendances](#installation-et-dépendances)
- [Utilisation](#utilisation)
- [Algorithmes et Approches](#algorithmes-et-approches)
- [Résultats et Analyse](#résultats-et-analyse)
- [Améliorations et Perspectives](#améliorations-et-perspectives)
- [Références](#références)

---

## Description du Projet

Ce projet a pour objectif de générer des séries temporelles financières réalistes à l’aide de modèles GAN. La génération est basée sur des données historiques réelles récupérées via l’API **yfinance** et traitées pour obtenir des séquences temporelles normalisées.

Les trois approches proposées permettent d’aborder différents aspects de la modélisation financière :

- **GAN_Finance** : Concentre sur la génération de séries temporelles pour un seul actif.
- **CorrGAN** : Vise à générer des séries pour plusieurs actifs en respectant la structure de corrélation observée dans les données réelles.
- **QuantGAN** : Utilise la fonction caractéristique pour aligner la distribution des rendements générés avec celle des rendements réels, en mettant l'accent sur la capture de la volatilité et des événements extrêmes.

---

## Structure du Répertoire

- **CorrGAN.py** : Code de l’implémentation de CorrGAN (modèle multi-actifs avec pénalité de corrélation).
- **GAN_Finance_API.py** (ou **mainFin.py**) : Implémentation de base pour générer des séries temporelles pour un actif unique (AAPL) en utilisant un MLP.
- **QuantGan.py** : Implémentation de QuantGAN, intégrant la pénalité basée sur la fonction caractéristique pour aligner la distribution des rendements.
- **main.py** : Exemple classique de GAN appliqué au dataset MNIST (pour illustration de la structure GAN).

---

## Installation et Dépendances

### Prérequis

- **Python 3.7+**
- **CUDA** (optionnel, pour accélérer l'entraînement avec GPU)

### Dépendances

Installez les bibliothèques nécessaires avec pip :

```bash
pip install torch torchvision yfinance pandas numpy scikit-learn matplotlib scipy

````
## Utilisation
### Exécution des Scripts
GAN_Finance (mainFin.py) :
Pour générer des séries temporelles univariées pour AAPL, exécutez :

```bash
python mainFin.py
```
CorrGAN (CorrGAN.py) :
Pour générer des séries multi-actifs en préservant les corrélations, exécutez :

```bash
python CorrGAN.py
```
QuantGAN (QuantGan.py) :
Pour entraîner le modèle QuantGAN qui intègre la fonction caractéristique, exécutez :

```bash
python QuantGan.py
```
### Résultats
Visualisation :
Chaque script génère des graphiques montrant les séries temporelles générées ou les distributions de rendements comparées aux données réelles.

Sorties :
Certains scripts sauvent des fichiers (par exemple, generated_series.csv pour les séries générées).

## Algorithmes et Approches

### CorrGAN
1. **Prétraitement**

- Télécharger et fusionner les données de plusieurs actifs.

- Normaliser et découper les séries en séquences de longueur fixe.

2. **Calcul de la Corrélation**
- Calculer la matrice de corrélation empirique des données.

3. **Modèles GAN**

- Générateur : transforme un vecteur de bruit en une séquence multi-actifs.

- Discriminateur : évalue la vraisemblance de chaque séquence.

4. **Pénalité de Corrélation**

- Ajouter une pénalité (MSE) entre la matrice de corrélation générée et la matrice réelle.

5. **Entraînement**

- Mise à jour alternée du discriminateur et du générateur avec la pénalité intégrée.

### GAN_Finance
1. **Prétraitement**

- Télécharger les données d’un actif (ex. AAPL) et normaliser.

- Découper en séquences de longueur fixe.

2. **Modèles GAN**

- Générateur : MLP transformant un vecteur de bruit en une séquence.

- Discriminateur : MLP évaluant la séquence générée.

3. **Entraînement Classique**

- Mise à jour du discriminateur et du générateur via une perte BCELoss.

4. **Génération et Visualisation**

- Générer des séries, appliquer l'inverse de normalisation et visualiser les résultats.

### QuantGAN
1. **Prétraitement**

- Télécharger et normaliser les données multi-actifs.

- Découper en séquences de longueur fixe.

2. **Fonction Caractéristique**

- Calculer les rendements journaliers et la fonction caractéristique (transformée de Fourier) pour un ensemble de fréquences définies.

- Faire la moyenne sur les actifs pour obtenir la référence réelle.

3. **Modèles GAN**

- Générateur et Discriminateur basés sur LSTM.

4. **Pénalité QuantGAN**

- Calculer la distance (MSE) entre la fonction caractéristique des rendements générés et celle des rendements réels.

- Intégrer cette pénalité dans la perte du générateur.

5. **Entraînement**

- Mise à jour alternée avec la perte adversariale et la pénalité de la fonction caractéristique.
## Résultats et Analyse
- **CorrGAN** vise à générer des séries temporelles multi-actifs dont la matrice de corrélation se rapproche de celle des données historiques.

- **GAN_Finance** génère des séries univariées pour un actif, permettant d'observer la dynamique temporelle et la distribution des rendements.

- **QuantGAN** s'efforce d'aligner la distribution des rendements générés (notamment la volatilité et les queues) avec celle des données réelles via l'utilisation de la fonction caractéristique.

Les résultats obtenus (graphes et statistiques affichés lors de l'exécution) permettent de comparer la fidélité des séries générées par rapport aux données réelles (corrélations, distributions, etc.).
## Améliorations et Perspectives
Augmenter la durée d’entraînement et ajuster les hyperparamètres pour une meilleure convergence.

Enrichir l’architecture (ex. utiliser des Transformers ou des modèles conditionnels).

Intégrer des pénalités supplémentaires (ex. pour la volatilité, la kurtosis, ou d'autres métriques de risque financier).

Validation statistique approfondie avec des indicateurs comme la VaR, l'autocorrélation, etc.
## Références
[Goodfellow et al., 2014] – Introduction aux GANs.

Documentation de PyTorch.

yfinance – Pour la récupération des données financières.

Articles sur QuantGAN, CorrGAN et applications financières des GANs.

