# SYSMER — Tracking 3D de Robot par Vision Stéréo + YOLO

Système de localisation 3D temps réel d'un robot sous-marin (BlueROV) à partir de deux caméras stéréo calibrées et d'un modèle de détection **YOLOv8** entraîné sur mesure.

---

## Table des matières

1. [Vue d'ensemble du projet](#vue-densemble-du-projet)
2. [Prérequis matériels](#prérequis-matériels)
3. [Installation de Python](#étape-1--installation-de-python)
4. [Installation de l'IDE](#étape-2--installation-de-lide)
5. [Téléchargement du projet](#étape-3--téléchargement-du-projet)
6. [Installation des bibliothèques](#étape-4--installation-des-bibliothèques)
7. [Structure du projet](#structure-du-projet)
8. [Mode Direct (Temps réel)](#étape-5--utilisation--mode-direct-temps-réel)
9. [Mode Post-Traitement (Vidéo)](#étape-6--utilisation--mode-post-traitement-vidéo)
10. [Comparaison TSV](#étape-7--comparaison-tsv)
11. [Problèmes fréquents](#problèmes-fréquents)

---

## Vue d'ensemble du projet

Ce projet est composé de **trois modules** distincts :

| Module | Description |
|--------|-------------|
| `Direct/` | Tracking en **temps réel** depuis deux caméras USB |
| `Post_Traitement/` | Tracking en **différé** depuis des fichiers vidéo `.avi` |
| `Comparaison_tsv/` | Comparaison des trajectoires YOLO avec un système de référence (Qualisys) |

Le pipeline complet se déroule en 4 grandes phases :

```
Calibration Intrinsèque → Calibration Extrinsèque → Redressement du sol → Tracking YOLO
```

---

## Prérequis matériels

Avant de commencer, assurez-vous de disposer de :

- Un **PC Windows 10/11** (les chemins dans les scripts sont écrits pour Windows)
- **2 caméras USB** (webcams ou caméras industrielles) pour le mode Direct
- Une **mire ArUco** imprimée : grille 6×6, dictionnaire `DICT_APRILTAG_36h11`, marqueurs de 88 mm avec espacement de 28 mm
- Un modèle YOLO entraîné (`best.pt`) — fichier à obtenir séparément

---

## Étape 1 : Installation de Python

### 1.1 Télécharger Python

Rendez-vous sur le site officiel : **https://www.python.org/downloads/**

Téléchargez la version **Python 3.10** ou **3.11** (recommandé pour la compatibilité avec toutes les bibliothèques).

> **Attention** : N'installez pas Python 3.13 ou plus récent, car certaines bibliothèques comme `ultralytics` peuvent ne pas encore être compatibles.

### 1.2 Installer Python

Lancez l'installateur téléchargé et, **avant de cliquer sur "Install Now"**, cochez impérativement la case :

```
☑ Add Python to PATH
```

Puis cliquez sur **"Install Now"**.

### 1.3 Vérifier l'installation

Ouvrez un **Invite de commandes** (touche `Win` → tapez `cmd` → Entrée) et tapez :

```bash
python --version
```

Vous devriez voir quelque chose comme `Python 3.11.x`. Si c'est le cas, Python est bien installé.

---

## Étape 2 : Installation de l'IDE

L'IDE utilisé pour ce projet est **Spyder**, inclus dans la distribution **Anaconda**.

### 2.1 Télécharger Anaconda

Rendez-vous sur : **https://www.anaconda.com/download**

Cliquez sur **"Download"** pour Windows et lancez l'installateur. Anaconda installe automatiquement Python, Spyder, et un gestionnaire de paquets (`conda`).

> Si Anaconda est déjà installé sur votre machine, passez directement à l'étape 2.3.

### 2.2 Installer Anaconda

Lors de l'installation, laissez toutes les options par défaut. À l'écran des options avancées, cochez :

```
☑ Add Anaconda to my PATH environment variable
```

### 2.3 Lancer Spyder

Une fois Anaconda installé, ouvrez le **Anaconda Navigator** (depuis le menu Démarrer) et cliquez sur **"Launch"** sous Spyder.

Ou directement depuis l'Invite de commandes Anaconda :

```bash
spyder
```

### 2.4 Configurer le répertoire de travail dans Spyder

Pour que Spyder trouve bien vos fichiers, définissez le dossier de travail :

1. En haut à droite de Spyder, cliquez sur l'icône  **"Global working directory"**
2. Naviguez jusqu'au dossier du projet, par exemple :
   ```
   C:\Users\VotreNom\Desktop\Projet_SYSMER_2A\
   ```

---

## Étape 3 : Téléchargement du projet

### Option A — Depuis GitHub

1. Rendez-vous sur la page du dépôt GitHub
2. Cliquez sur le bouton vert **"Code"** → **"Download ZIP"**
3. Extrayez le ZIP dans un dossier de votre choix, par exemple :
   ```
   C:\Users\VotreNom\Desktop\Projet_SYSMER_2A\
   ```

### Option B — Via Git (si Git est installé)

```bash
git clone https://github.com/<votre-repo>/Methodes-Finales-YOLO.git
```

### 3.1 Adapter les chemins dans les scripts

> **Important** : Les scripts contiennent des chemins codés en dur qu'il faut adapter à votre machine.

Ouvrez chaque script Python et remplacez toutes les occurrences de :

```python
r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\Calibrage Direct"
```

par le chemin du dossier `Matrice/` correspondant sur votre machine, par exemple :

```python
r"C:\Users\VotreNom\Desktop\Projet_SYSMER_2A\Direct\Matrice"
```

Les fichiers concernés sont : `Intrinsec.py`, `Extrinsec.py`, `Redressement.py`, `Tracking.py` (dans `Direct/Code/` et `Post_Traitement/Codes/`).

---

## Étape 4 : Installation des bibliothèques

### 4.1 Ouvrir le terminal Anaconda

Ouvrez l'**Anaconda Prompt** depuis le menu Démarrer (cherchez "Anaconda Prompt"), puis naviguez vers le dossier du projet :

```bash
cd C:\Users\VotreNom\Desktop\Projet_SYSMER_2A
```

### 4.2 (Recommandé) Créer un environnement conda dédié

```bash
conda create -n sysmer python=3.11
conda activate sysmer
```

Vous devriez voir `(sysmer)` apparaître au début de votre ligne de commande.

>  Pour que Spyder utilise cet environnement, installez-y aussi le noyau Spyder :
> ```bash
> conda install spyder-kernels
> ```
> Puis dans Spyder : **Outils → Préférences → Python interpreter → Use the following Python interpreter**, et pointez vers l'environnement `sysmer`.

### 4.3 Installer toutes les bibliothèques

Exécutez la commande suivante pour tout installer d'un coup :

```bash
pip install opencv-python opencv-contrib-python numpy ultralytics scipy pandas matplotlib
```

Détail des bibliothèques installées :

| Bibliothèque | Rôle dans le projet |
|---|---|
| `opencv-python` | Lecture vidéo, affichage, calculs de vision |
| `opencv-contrib-python` | Détection ArUco, calibration stéréo |
| `numpy` | Calculs matriciels (calibration, triangulation) |
| `ultralytics` | Modèle YOLO pour la détection du robot |
| `scipy` | Optimisation de la rotation, interpolation des trajectoires |
| `pandas` | Lecture et manipulation des fichiers `.tsv` |
| `matplotlib` | Tracé des trajectoires comparées |

### 4.4 Vérifier l'installation

```bash
python -c "import cv2; import ultralytics; import numpy; print(' Tout est installé !')"
```

---

## Structure du projet

```
Methodes-Finales-YOLO-main/
│
├── Direct/                         ← Mode temps réel (caméras USB)
│   ├── Code/
│   │   ├── Intrinsec.py            # Calibration intrinsèque (distorsion objectif)
│   │   ├── Extrinsec.py            # Calibration extrinsèque (position relative des caméras)
│   │   ├── Redressement.py         # Calibration du sol (axe vertical = gravité)
│   │   ├── Tracking.py             # Tracking YOLO en temps réel ← SCRIPT PRINCIPAL
│   │   ├── Passage.py              # Conversion de repères
│   │   └── Test_Index.py           # Test des indices caméras
│   └── Matrice/                    # Matrices de calibration (fichiers .npy)
│       ├── K1.npy, K2.npy          # Matrices intrinsèques
│       ├── D1.npy, D2.npy          # Coefficients de distorsion
│       ├── P1.npy, P2.npy          # Matrices de projection
│       ├── R_c2_c1.npy             # Rotation cam2 → cam1
│       ├── t_c2_c1.npy             # Translation cam2 → cam1
│       ├── R_redressement.npy      # Rotation de redressement du sol
│       └── hauteur_cam1.npy        # Hauteur de la caméra 1
│
├── Post_Traitement/                ← Mode différé (fichiers vidéo)
│   ├── Codes/                      # Mêmes scripts, adaptés pour les vidéos
│   ├── Matrice/                    # Matrices de calibration
│   ├── Tsv/                        # Données de trajectoires exportées
│   └── Videos/
│       ├── Etalonnage/             # Vidéos de calibration
│       └── Sequences/              # Vidéos de tracking
│
└── Comparaison_tsv/
│    └── TSV_Rota.py                 # Comparaison YOLO vs Qualisys
│
└── Modèles YOLO
   └── best_air.pt                   # Modèle pour des test en air
   └── best_eau.pt                   # Modèle pour des test en eau
```

---

## Étape 5 : Utilisation — Mode Direct (Temps réel)

Le mode Direct effectue le tracking en capturant les images directement depuis vos caméras USB.

### Phase 1 — Calibration Intrinsèque (une seule fois par caméra)

Ce script calcule les paramètres optiques de chaque caméra (focale, distorsion).

**Répétez cette opération pour chaque caméra (cam1 et cam2).**

1. Ouvrez `Direct/Code/Intrinsec.py`
2. Adaptez les paramètres en haut du fichier :
   ```python
   FICHIER_K_OUT = "K1.npy"   # K1.npy pour cam1, K2.npy pour cam2
   FICHIER_D_OUT = "D1.npy"   # D1.npy pour cam1, D2.npy pour cam2
   ```
3. Filmez la mire ArUco en effectuant des mouvements variés (inclinaisons, rotations, bords du cadre)
4. Exécutez le script :
   ```bash
   python Direct/Code/Intrinsec.py
   ```
5. Le script extrait automatiquement des images toutes les 45 frames. Il s'arrête quand 20 images valides sont capturées.
6.  Résultat attendu : `RMS < 2.0 pixels` et génération de `K1.npy` et `D1.npy`

>  Si le RMS est trop élevé, filmez à nouveau en variant davantage les angles et la distance.

### Phase 2 — Calibration Extrinsèque (une seule fois par setup)

Ce script calcule la position et l'orientation relative entre les deux caméras.

1. Placez la mire ArUco dans le champ de vision **commun** aux deux caméras
2. Exécutez le script :
   ```bash
   python Direct/Code/Extrinsec.py
   ```
3. Le déclenchement est automatique quand au moins 20 ArUcos communs sont détectés sur 3 frames consécutives
4. Un affichage 3D apparaît pour valider la pose : confirmez en console
5. Résultat : génération de `R_c2_c1.npy`, `t_c2_c1.npy`

### Phase 3 : Matrices de Projection (une seule fois par setup)

Cette étape purement algébrique prépare les matrices finales pour la triangulation.

1. Exécutez le script :
   ```bash
   python Direct/Code/Passage.py
   ```
2. Le code va lire les fichiers générés aux étapes 1 et 2.
3. Résultat : Il génère automatiquement `P1.npy` et `P2.npy` dans votre dossier.

### Phase 4 — Calibration du Sol / Redressement (une seule fois par setup)

Ce script oriente l'axe Y du repère monde selon la gravité (sol horizontal).

1. Posez la mire ArUco à plat sur le sol, dans le champ de la caméra 1
2. Exécutez le script :
   ```bash
   python Direct/Code/Redressement.py
   ```
3. Un affichage 3D valide l'orientation : confirmez en console
4. Résultat : génération de `R_redressement.npy` et `hauteur_cam1.npy`

### Phase 5 — Tracking en Temps Réel

1. Placez votre modèle YOLO entraîné (`best.pt`) à l'emplacement indiqué dans le script
2. Vérifiez que toutes les matrices `.npy` sont présentes dans `Direct/Matrice/`
3. Branchez vos deux caméras USB (indices `0` et `1`)
4. Lancez le tracking :
   ```bash
   python Direct/Code/Tracking.py
   ```

**Interface de l'application :**

| Zone | Contenu |
|------|---------|
| En haut | Flux vidéo des deux caméras avec les détections YOLO encadrées |
| Bas gauche | Vue de dessus (plan X-Z) avec la trajectoire du robot |
| Bas centre | Vue de profil (plan Z-Y, hauteur) |
| Bas droit | Tableau de bord : coordonnées 3D, distance, FPS, état de l'enregistrement |

**Commandes clavier :**

| Touche | Action |
|--------|--------|
| `Q` | Quitter le programme |
| `Espace` | Mettre en pause / reprendre |
| `R` | Activer / désactiver l'enregistrement dans le fichier TSV |

Un fichier `trajectoire_robot_temps_reel.tsv` est généré automatiquement dans le dossier de calibration.

---

## Étape 6 : Utilisation — Mode Post-Traitement (Vidéo)

Ce mode fonctionne de manière identique au mode Direct, mais lit des fichiers vidéo `.avi` préenregistrés au lieu d'utiliser les caméras en direct. Il est utile pour rejouer et analyser des séquences.

Les vidéos de calibration et de séquences sont fournies dans `Post_Traitement/Videos/`.

1. **Calibration** : exécutez les scripts dans `Post_Traitement/Codes/` dans le même ordre (Intrinsec → Extrinsec → Redressement), en veillant à ce que les chemins pointent vers `Post_Traitement/Videos/Etalonnage/`
2. **Tracking** : lancez `Post_Traitement/Codes/Tracking.py` en configurant les chemins vers les vidéos `Sequences/`
3. Un fichier script `enregistre.py` est disponible pour gérer l'export des données

---

## Étape 7 : Comparaison TSV

Le script `Comparaison_tsv/TSV_Rota.py` permet de comparer la trajectoire estimée par le système YOLO avec une trajectoire de référence issue du système **Qualisys** (mocap).

### Prérequis

Vous devez disposer de :
- Un fichier `.tsv` exporté par Qualisys (trajectoire de référence)
- Un fichier `.tsv` généré par le tracking YOLO

### Lancer la comparaison

1. Placez vos deux fichiers TSV dans le dossier `Comparaison_tsv/`
2. Ouvrez `TSV_Rota.py` et adaptez les noms de fichiers :
   ```python
   df_a = pd.read_csv('votre_fichier_qualisys.tsv', sep='\t', skiprows=11)
   df_b = pd.read_csv('trajectoire_robot_clean.tsv', sep='\t')
   ```
3. Exécutez le script :
   ```bash
   python Comparaison_tsv/TSV_Rota.py
   ```

Le script aligne automatiquement les deux trajectoires par optimisation de rotation et affiche les courbes comparatives avec `matplotlib`.

---

## Problèmes fréquents

**Les caméras ne s'ouvrent pas (`Impossible d'ouvrir l'une des caméras`)**
→ Vérifiez que les deux caméras sont bien branchées. Testez les indices `0`, `1`, `2` dans `Test_Index.py` pour trouver les bons numéros.

**Erreur `FileNotFoundError` sur les fichiers `.npy`**
→ Les chemins dans les scripts sont incorrects. Relisez l'Étape 3.1 et adaptez toutes les variables `dossier` et `DOSSIER_CALIB`.

**`ModuleNotFoundError: No module named 'ultralytics'`**
→ L'environnement conda n'est peut-être pas activé. Dans l'Anaconda Prompt, exécutez `conda activate sysmer` puis relancez le script.

**Le modèle YOLO ne détecte rien**
→ Vérifiez que le chemin vers `best.pt` est correct dans `Tracking.py`. Assurez-vous que le robot est visible dans les deux caméras simultanément.

**RMS de calibration trop élevé (> 2.0)**
→ Filmez à nouveau la mire en variant fortement les angles, distances et orientations. Évitez le flou de bougé.

**Le redressement du sol ne fonctionne pas**
→ La mire doit être posée strictement à plat, entièrement visible depuis la caméra 1, dans de bonnes conditions d'éclairage.

---

## Format des fichiers TSV exportés

Le tracking génère un fichier TSV avec les colonnes suivantes :

| Colonne | Description |
|---------|-------------|
| `Frame` | Numéro de la frame |
| `Temps(s)` | Timestamp en secondes |
| `X(m)` | Position latérale en mètres |
| `Y(m)` | Hauteur en mètres |
| `Z(m)` | Profondeur (distance caméra) en mètres |
