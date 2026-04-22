# Détection et Tracking 3D du BlueROV2 (YOLO + Stéréovision)

Ce dépôt contient le pipeline complet de notre projet SYSMER : détecter et localiser en 3D un robot sous-marin BlueROV2 à l'aide de deux caméras fisheye, d'un réseau de neurones YOLOv8 et de la triangulation stéréoscopique d'OpenCV.

Le principe : YOLO détecte le robot dans chaque caméra, on triangule les deux points 2D pour reconstruire sa position en 3D, et on l'affiche en temps réel sur une carte vue de dessus + profil.

> 📁 Ce repo est la **suite logique** du repo [Méthodes Classiques (OpenCV)](lien_vers_repo_openCV), qui documente pourquoi nous avons abandonné les approches HSV / ORB au profit de YOLO.

---

## Structure du projet

```
Methodes-Finales-YOLO-main/
│
├── Direct/                      → Pipeline en temps réel (caméras branchées en direct)
│   ├── Code/
│   │   ├── Test_Index.py        ← Étape 0 : Trouver les index de vos caméras
│   │   ├── Intrinsec.py         ← Étape 1 : Calibration intrinsèque (K, D) × 2 caméras
│   │   ├── Extrinsec.py         ← Étape 2 : Calibration extrinsèque (position relative)
│   │   ├── Passage.py           ← Étape 3 : Construction des matrices de projection P1, P2
│   │   ├── Redressement.py      ← Étape 4 : Calibration du plan sol (redressement Y)
│   │   └── Tracking.py          ← Étape 5 : Tracking 3D en direct avec YOLO
│   └── Matrice/                 → Matrices de calibration pré-calculées (.npy)
│       ├── K1.npy, K2.npy       (matrices intrinsèques)
│       ├── D1.npy, D2.npy       (coefficients de distorsion)
│       ├── P1.npy, P2.npy       (matrices de projection stéréo)
│       ├── R_c2_c1.npy          (rotation entre caméras)
│       ├── t_c2_c1.npy          (translation entre caméras)
│       ├── R_redressement.npy   (correction du plan sol)
│       └── hauteur_cam1.npy     (hauteur de la caméra 1 au-dessus du sol)
│
├── Post_Traitement/             → Pipeline sur vidéos pré-enregistrées (même logique)
│   ├── Codes/
│   │   ├── enregistre.py        ← Enregistre les flux des 2 caméras en .avi
│   │   ├── Intrinsec.py
│   │   ├── Extrinsec.py
│   │   ├── Passage.py
│   │   ├── Redressement.py
│   │   └── Tracking.py
│   ├── Matrice/                 → Idem que Direct/Matrice/
│   ├── Tsv/                     → Trajectoires de référence Qualisys (vérité terrain)
│   └── Videos/
│       ├── Etalonnage/          (vidéos des mires de calibration)
│       └── Sequences/           (vidéos des séquences de test avec le robot)
│
└── Comparaison_tsv/
    └── TSV_Rota.py              ← Compare la trajectoire YOLO avec Qualisys (RMSE)
```

**Quelle version utiliser ?**
- **`Direct/`** : vous avez les deux caméras branchées et voulez tracker le robot en temps réel.
- **`Post_Traitement/`** : vous avez déjà enregistré les vidéos (avec `enregistre.py`) et voulez rejouer le pipeline hors ligne.

---

## Étape 0 — Installer Python et les dépendances

### 0.1 Installer Python

1. Rendez-vous sur **[python.org/downloads](https://www.python.org/downloads/)** et téléchargez **Python 3.10** ou **3.11** (recommandé).
2. Lancez l'installateur. **⚠️ Cochez bien "Add Python to PATH"** avant de cliquer sur "Install Now".
3. Vérifiez que l'installation a fonctionné en ouvrant un terminal et en tapant :
   ```bash
   python --version
   ```
   Vous devriez voir quelque chose comme `Python 3.11.x`.

### 0.2 Installer les dépendances

Toutes les bibliothèques nécessaires s'installent en une seule commande :
```bash
pip install opencv-python numpy ultralytics scipy matplotlib pandas
```

Voilà à quoi sert chaque bibliothèque :
- `opencv-python` : traitement d'image, détection ArUco, triangulation stéréo
- `numpy` : calcul matriciel (matrices de calibration, points 3D)
- `ultralytics` : YOLOv8 (détection du robot)
- `scipy` : optimisation Nelder-Mead pour la calibration extrinsèque
- `matplotlib` : affichage 3D de validation des calibrations
- `pandas` : lecture des fichiers TSV Qualisys (utilisé uniquement dans `TSV_Rota.py`)

> 💡 Si vous êtes sous Windows et que `pip` n'est pas reconnu, remplacez-le par `python -m pip install ...`.

---

## Ce qu'il vous faut avant de commencer

- **Un modèle YOLO entraîné** sur le BlueROV2 (fichier `.pt`). Notez bien le chemin vers votre fichier `best.pt`, vous en aurez besoin à l'étape 5.
- **Deux caméras fisheye** branchées en USB.
- **Une mire ArUco** : un plateau 6×6 de tags `DICT_APRILTAG_36h11` (chaque marqueur fait 88 mm, espacement 28 mm).

---

## Étape 1 — Trouver les index de vos caméras

Avant toute chose, il faut savoir quel index OpenCV a attribué à chacune de vos deux caméras.

**Lancez `Test_Index.py` :**
```bash
python Direct/Code/Test_Index.py
```
Le script teste les index 0 à 9, ouvre chaque caméra et affiche l'index à l'écran pendant 5 secondes. Notez lesquels correspondent à vos caméras gauche et droite.

> ⚙️ **À adapter :** Dans tous les scripts qui suivent, `ID_CAM1` et `ID_CAM2` (ou les lignes `cv2.VideoCapture(...)`) devront correspondre aux index que vous venez de trouver.

---

## Étape 2 — Calibration intrinsèque (à faire pour chaque caméra)

Cette étape calcule les paramètres optiques de chaque caméra (focale, centre optique, distorsion) et les sauvegarde dans des fichiers `.npy`.

**Script :** `Intrinsec.py`

### 2.1 Préparer votre vidéo

Filmez votre mire ArUco en la déplaçant lentement devant la caméra : faites-la pivoter, inclinez-la, approchez-la des bords de l'image. Plus vous couvrez de positions différentes, meilleure sera la calibration.

### 2.2 Configurer le script

Ouvrez `Intrinsec.py` et modifiez le bloc de configuration en haut du fichier :
```python
CHEMIN_VIDEO = r"C:\chemin\vers\votre\video_cam1.avi"  # ← Votre vidéo
FICHIER_K_OUT = "K1.npy"   # ← K1.npy pour la caméra gauche, K2.npy pour la droite
FICHIER_D_OUT = "D1.npy"   # ← D1.npy pour la caméra gauche, D2.npy pour la droite
```

### 2.3 Lancer le script

```bash
python Direct/Code/Intrinsec.py
```
Le script extrait automatiquement 20 images de la vidéo (une toutes les 45 frames) et calcule le modèle de lentille. À la fin, il affiche le **RMS de reprojection** :
- ✅ `RMS < 2.0` → calibration acceptée, les fichiers `K1.npy` et `D1.npy` sont sauvegardés.
- ❌ `RMS ≥ 2.0` → calibration rejetée. Re-filmez avec plus de diversité de positions.

**Répétez l'opération pour la deuxième caméra** en changeant `CHEMIN_VIDEO`, `FICHIER_K_OUT` et `FICHIER_D_OUT` pour la caméra 2.

> ⚙️ **À adapter :** Si vous utilisez une mire de taille différente, modifiez `L` (taille du marqueur en mètres) et `S` (espacement en mètres) en haut du script.

---

## Étape 3 — Calibration extrinsèque (position relative des caméras)

Cette étape détermine la position et l'orientation d'une caméra par rapport à l'autre. Elle nécessite que les deux caméras voient la mire en même temps.

**Script :** `Extrinsec.py`

### 3.1 Configurer le script

```python
DOSSIER_CALIB  = r"C:\chemin\vers\votre\dossier_matrice"  # ← Là où sont K1.npy, K2.npy, D1.npy, D2.npy
DOSSIER_SORTIE = DOSSIER_CALIB                             # ← Là où seront sauvegardées les nouvelles matrices
ID_CAM1 = 0   # ← Index de votre caméra gauche
ID_CAM2 = 1   # ← Index de votre caméra droite
```

### 3.2 Lancer le script

```bash
python Direct/Code/Extrinsec.py
```
Le script ouvre les flux des deux caméras en direct. Placez la mire pour qu'elle soit visible par les deux caméras simultanément.

**Procédure :**
1. Quand vous êtes prêt, appuyez sur **ENTRÉE** pour lancer l'analyse de stabilité.
2. Le système attend que 20 ArUcos communs soient visibles et stables pendant 10 frames consécutives.
3. Un graphique 3D s'affiche pour valider visuellement la position relative des deux caméras.
4. Répondez `O` dans le terminal pour sauvegarder, ou `N` pour recommencer.

Les fichiers sauvegardés sont : `R_c2_c1.npy`, `t_c2_c1.npy`, `c2Mc1.npy`, `wMc1.npy`, `wMc2.npy`.

---

## Étape 4 — Construction des matrices de projection

À partir des matrices intrinsèques et extrinsèques, on construit les matrices `P1` et `P2` dont a besoin OpenCV pour la triangulation.

**Script :** `Passage.py`

### 4.1 Configurer le script

```python
DOSSIER = r"C:\chemin\vers\votre\dossier_matrice"  # ← Le même dossier qu'à l'étape précédente
```

### 4.2 Lancer le script

```bash
python Direct/Code/Passage.py
```
Le script se termine en quelques secondes et génère `P1.npy` et `P2.npy` dans votre dossier.

---

## Étape 5 — Calibration du plan sol (redressement)

Les caméras ne regardent pas exactement vers le bas, ce qui introduit une inclinaison dans les coordonnées 3D reconstruites. Cette étape calcule une matrice de rotation pour "redresser" le repère de sorte que l'axe Y corresponde bien à la gravité.

**Script :** `Redressement.py`

### 5.1 Configurer le script

```python
DOSSIER_CALIB = r"C:\chemin\vers\votre\dossier_matrice"
ID_CAM1 = 0
```

### 5.2 Lancer le script

```bash
python Direct/Code/Redressement.py
```

**Procédure :**
1. Posez la mire **à plat sur le sol** du bassin (ou de la surface de test).
2. Appuyez sur **ENTRÉE** pour lancer l'analyse.
3. Un graphique 3D s'affiche avec les axes du nouveau repère monde.
4. Répondez `O` pour sauvegarder.

Les fichiers sauvegardés sont : `R_redressement.npy` et `hauteur_cam1.npy`.

> 💡 **Cette étape est optionnelle.** Si `R_redressement.npy` est absent, `Tracking.py` démarrera quand même avec le repère brut de la caméra 1 (il affichera un avertissement).

---

## Étape 6 — Lancer le Tracking 3D

C'est le script principal. Il ouvre les deux caméras, lance YOLO sur chaque flux, triangule la position du robot et affiche tout en temps réel.

**Script :** `Tracking.py`

### 6.1 Configurer le script

```python
dossier = r"C:\chemin\vers\votre\dossier_matrice"  # ← Dossier contenant tous les .npy
```
Et plus bas, la ligne du modèle YOLO :
```python
model = YOLO(r'C:\chemin\vers\votre\best.pt')  # ← Votre modèle entraîné
```

### 6.2 Lancer le script

```bash
python Direct/Code/Tracking.py
```

### 6.3 Interface et contrôles

La fenêtre affiche :
- En haut : flux vidéo des deux caméras avec la bounding box YOLO.
- En bas à gauche : vue de dessus (plan X-Z) avec la trajectoire et les cônes de vision.
- En bas au milieu : vue de profil (plan Z-Y) avec la hauteur du robot.
- En bas à droite : coordonnées X, Y, Z en temps réel, FPS, état de détection.

| Touche | Action |
|--------|--------|
| `Q` | Quitter proprement (sauvegarde le TSV) |
| `ESPACE` | Mettre en pause |
| `R` | Activer / désactiver l'enregistrement dans le fichier TSV |

La trajectoire est automatiquement sauvegardée dans un fichier `trajectoire_robot_temps_reel.tsv` dans votre dossier de matrices.

---

## (Optionnel) Mode Post-Traitement

Si vous avez enregistré vos vidéos à l'avance (utile quand les caméras ne sont pas sur le même PC que le code), utilisez les scripts du dossier `Post_Traitement/Codes/`.

### Enregistrer les vidéos

```bash
python Post_Traitement/Codes/enregistre.py
```

> ⚙️ **À adapter :** Modifiez les noms des fichiers de sortie (`Balade_Air1_3.avi`, `Balade_Air2_3.avi`) et les index de caméras dans `enregistre.py`.

Ensuite, répétez les étapes 1 à 5 en utilisant les scripts de `Post_Traitement/Codes/` à la place de ceux de `Direct/Code/`, et en pointant `CHEMIN_VIDEO` vers vos fichiers `.avi` au lieu des flux caméra.

Des vidéos de calibration et des séquences de test sont déjà disponibles dans `Post_Traitement/Videos/` pour tester le pipeline sans avoir le matériel.

---

## (Optionnel) Comparaison avec Qualisys

Si vous avez une vérité terrain Qualisys, le script `TSV_Rota.py` compare la trajectoire produite par YOLO avec celle du système de référence et calcule le RMSE par axe.

```bash
python Comparaison_tsv/TSV_Rota.py
```

> ⚙️ **À adapter :** Modifiez les chemins vers vos fichiers TSV en haut du script (`df_a = pd.read_csv(...)` pour Qualisys, `df_b = pd.read_csv(...)` pour la trajectoire YOLO).

Le script produit :
- Une vue 3D superposant les deux trajectoires.
- Une vue de dessus (plan X-Y).
- L'évolution de la hauteur (axe Z).
- Le RMSE global et par axe affiché en bas du graphique.

---

## Récapitulatif de l'ordre d'exécution

```
Test_Index.py          → Trouver les index des caméras
       ↓
Intrinsec.py (×2)      → K1, D1, K2, D2
       ↓
Extrinsec.py           → R_c2_c1, t_c2_c1, c2Mc1, wMc1, wMc2
       ↓
Passage.py             → P1, P2
       ↓
Redressement.py        → R_redressement, hauteur_cam1
       ↓
Tracking.py            → Tracking 3D en temps réel 🚀
```

---

*Projet réalisé dans le cadre du module de vision par ordinateur — SeaTech, 2025-2026.*
