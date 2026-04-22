# README — Installation physique du système

Ce document décrit comment installer physiquement le matériel pour utiliser le système de tracking 3D du BlueROV. Il couvre le montage des caméras, le déploiement de Qualisys et la préparation de la mire. Pour l'installation logicielle et l'utilisation des scripts, voir le [README principal](README.md).

---

## Matériel nécessaire

**Système visuel (notre solution)**
- 2 caméras USB DeepWater exploreHD (400 m)
- 1 PC portable avec 2 ports USB libres
- 1 mire d'étalonnage AprilTag 36h11 — grille 6×6, marqueurs de 8,8 cm, espacement 2,8 cm (85 cm de côté total)
- Support pour les caméras (palette + cordes + serre-joints)
- Cordes pour manipuler la mire dans le bassin

**Système de référence Qualisys** (Si besoin de comparaison)
- 5 caméras Qualisys sous-marines avec supports (barres profilées ITEM)
- Équerre de calibration sous-marine
- Baguette dynamique (T-wand) avec marqueurs
- Marqueurs passifs (boules réfléchissantes) à fixer sur le BlueROV et les caméras USB

---

## 1. Installation des caméras USB

1. Fixez les deux caméras sur un support rigide (palette + cordes) au rebord du bassin avec des serre-joints.
2. Orientez chaque caméra vers le bas, légèrement inclinées en direction de la zone de travail dans le bassin, à la même hauteur approximative.
3. **L'angle entre les axes optiques des deux caméras ne doit pas dépasser 90°.** Au-delà, l'étalonnage extrinsèque échoue.
4. Branchez les deux caméras au PC. Identifiez leurs indices USB avec `Test_Index.py` avant de lancer les scripts.

> **Limite câble USB :** les câbles USB limitent la distance entre le PC et le rebord du bassin, ce qui réduit la zone de travail atteignable. Prévoyez des prolongateurs si nécessaire.

---

## 2. Déploiement du système Qualisys (référence)

Cette étape est facultative — elle sert uniquement à valider la précision du système par comparaison avec une vérité terrain millimétrique.

### 2.1 Installation des caméras Qualisys

1. Fixez les 5 caméras Qualisys sous-marines sur les barres profilées ITEM autour du bassin.
2. Vérifiez que les caméras couvrent bien toute la zone de travail prévue pour le BlueROV.

> L'installation des caméras Qualisys est longue et complexe — prévoir une demi-journée.

### 2.2 Étalonnage Qualisys

1. Placez l'équerre de calibration sous-marine au fond du bassin pour définir l'origine et les axes (X, Y, Z) du repère monde.
2. Agitez le T-wand (baguette dynamique avec marqueurs) dans tout le volume du bassin. Les 5 caméras cartographient ainsi l'espace 3D avec une précision millimétrique.

### 2.3 Pose des marqueurs sur le matériel

- Fixez des marqueurs passifs (boules réfléchissantes) sur le châssis du BlueROV.
- Fixez également des marqueurs sur chacune des deux caméras USB — cela permet de vérifier a posteriori la qualité de l'étalonnage extrinsèque en comparant la distance inter-caméras calculée et mesurée physiquement.

### 2.4 Acquisition et export

1. Lors du test, Qualisys suit individuellement chaque marqueur.
2. En post-traitement dans le logiciel Qualisys, sélectionnez le groupe de marqueurs fixés sur le robot et définissez un **Rigid Body** — le logiciel traite alors ces points comme un seul objet géométrique solide.
3. Exportez la trajectoire du Rigid Body au format `.tsv` pour la comparer ensuite avec la trajectoire produite par notre système (voir `Comparaison_tsv/TSV_Rota.py`).

---

## 3. Préparation de la mire

La mire sert à la fois pour l'étalonnage intrinsèque, l'étalonnage extrinsèque et le redressement du repère.

1. Percez les quatre coins de la mire et fixez-y un bout de corde à chaque coin.
2. Ces cordes permettent de manipuler et d'orienter la mire dans le bassin sans y plonger les bras.
3. Pour l'étalonnage **intrinsèque** : déplacez la mire lentement devant chaque caméra individuellement, en couvrant tous les coins de l'image et en l'inclinant sous différents angles. Évitez tout mouvement rapide — le flou de mouvement empêche OpenCV de détecter les AprilTags.
4. Pour l'étalonnage **extrinsèque** : placez la mire dans la zone commune aux deux caméras et maintenez-la immobile à la pression de la touche `Entrée`.
5. Pour le **redressement du repère** : posez la mire à plat au fond du bassin, bien horizontale, dans le champ de la caméra 1.

> Sous l'eau, la résistance hydrodynamique rend la manipulation difficile. Les mouvements doivent être lents pour éviter les remous et le flou de mouvement.

---

## 4. Préparation du BlueROV

1. Vérifiez l'étanchéité du robot avant immersion.
2. **Aucun autre marqueur ne doit être ajouté** — notre système ne nécessite aucune modification du robot pour la détection YOLO.

---

## 5. Récapitulatif de l'ordre d'installation

```
1. Installer et orienter les caméras USB sur leur support
2. Déployer et étalonner Qualisys (si validation souhaitée)
3. Fixer les marqueurs Qualisys sur le robot et les caméras USB
4. Préparer la mire (cordes aux 4 coins)
5. Lancer les scripts dans l'ordre : Intrinsec → Extrinsec → Redressement → Tracking
```

Pour la suite (scripts Python), voir le [README principal](README.md).
