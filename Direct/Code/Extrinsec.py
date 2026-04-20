# -*- coding: utf-8 -*-
"""
=============================================================================
  CALIBRATION EXTRINSÈQUE STÉRÉO — TEMPS RÉEL
=============================================================================
  - Capture en direct depuis 2 caméras
  - Plateau ArUcos = Origine du Monde (0,0,0)
  - Déclenchement auto si X ArUcos communs sur 3 frames consécutives
  - Affichage 3D bloquant + validation utilisateur en console
=============================================================================
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
import os
import sys

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
DOSSIER_CALIB  = r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\Calibrage Direct"
DOSSIER_SORTIE = DOSSIER_CALIB

ID_CAM1 = 0
ID_CAM2 = 1

# Géométrie de la mire
L   = 0.088
S   = 0.028
COL = 6
LIG = 6

DICTIONNAIRE = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

# Seuil initial
TARGET_ARUCOS = 20

# =============================================================================
# 2. CHARGEMENT DES INTRINSÈQUES
# =============================================================================
print("=" * 60)
print("  CHARGEMENT DES PARAMÈTRES INTRINSÈQUES")
print("=" * 60)

try:
    K1 = np.load(os.path.join(DOSSIER_CALIB, "K1.npy"))
    D1 = np.load(os.path.join(DOSSIER_CALIB, "D1.npy"))
    K2 = np.load(os.path.join(DOSSIER_CALIB, "K2.npy"))
    D2 = np.load(os.path.join(DOSSIER_CALIB, "D2.npy"))
    print(f"✅ K et D chargés depuis : {DOSSIER_CALIB}")
except FileNotFoundError as e:
    print(f"❌ Erreur : Fichier intrinsèque introuvable. ({e})")
    sys.exit(1)

# =============================================================================
# 3. DÉTECTEUR ARUCO & FONCTIONS UTILITAIRES
# =============================================================================
params = aruco.DetectorParameters()
params.markerBorderBits          = 2
params.adaptiveThreshWinSizeMin  = 3
params.adaptiveThreshWinSizeMax  = 30
params.adaptiveThreshWinSizeStep = 3
params.adaptiveThreshConstant    = 7
params.minMarkerPerimeterRate    = 0.01
params.maxMarkerPerimeterRate    = 4.0
detector = aruco.ArucoDetector(DICTIONNAIRE, params)
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.polygonalApproxAccuracyRate = 0.05

def detecter_aruco(img):
    largeur_cible = 2000
    ratio = largeur_cible / float(img.shape[1])
    img_res = cv2.resize(img, (largeur_cible, int(img.shape[0] * ratio)), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)
    marqueurs = {}
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            centre = corners[i][0].mean(axis=0) / ratio   
            marqueurs[int(mid)] = centre
    return marqueurs

def coords_3d_aruco(mid):
    col, row = mid % COL, mid // COL
    r, c = (LIG - 1 - row), (COL - 1 - col)
    x = c * (L + S) + L / 2.0
    y = r * (L + S) + L / 2.0
    return np.array([x, y, 0.0])

def undistordre_points(pts_px, K, D):
    pts = np.array(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    pts_ud = cv2.undistortPoints(pts, K, D, P=K)
    return pts_ud.reshape(-1, 2)

def normaliser_points(pts_px, K):
    K_inv = np.linalg.inv(K)
    pts_h = np.hstack([pts_px, np.ones((len(pts_px), 1))])
    return (K_inv @ pts_h.T).T

def solvePnP_vers_matrice(pts3d, pts2d_px, K, D):
    success, rvec, tvec = cv2.solvePnP(
        pts3d.astype(np.float64), pts2d_px.astype(np.float64),
        K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    R_mat, _ = cv2.Rodrigues(rvec)
    cMw = np.eye(4)
    cMw[:3, :3] = R_mat
    cMw[:3,  3] = tvec.flatten()
    wMc = np.linalg.inv(cMw)
    return wMc, cMw, rvec, tvec

def homogeneousMatrix(tx, ty, tz, rx, ry, rz):
    rot  = Rot.from_euler('XYZ', [rx, ry, rz], degrees=True)
    return np.block([[rot.as_matrix(), np.array([[tx, ty, tz]]).T], [np.array([[0,0,0,1]])]])

def changeFrame(oXs, cMo):
    return np.array([cMo @ oX for oX in oXs])

def pinholeProj(cXs):
    xs = []
    for X in cXs:
        if X[2] <= 0: return None
        xs.append(np.array([X[0]/X[2], X[1]/X[2], 1.0]))
    return np.array(xs)

def cost_nelder(params, wXs_h, c2xs_norm):
    wMc2_try  = homogeneousMatrix(*params)
    c2Mw_try  = np.linalg.inv(wMc2_try)
    c2Xs      = changeFrame(wXs_h, c2Mw_try)
    c2xs_proj = pinholeProj(c2Xs)
    if c2xs_proj is None: return 1e10
    return np.sum((c2xs_norm[:, :2] - c2xs_proj[:, :2])**2)

# =============================================================================
# 4. FONCTION DE CALCUL ET D'AFFICHAGE 3D
# =============================================================================
def executer_calibration(marq1, marq2, communs):
    print(f"\n🚀 Lancement du calcul sur {len(communs)} points communs...")
    ids_sel = sorted(list(communs))
    
    pts1_px = np.array([marq1[mid] for mid in ids_sel], dtype=np.float64)
    pts2_px = np.array([marq2[mid] for mid in ids_sel], dtype=np.float64)

    pts1_ud = undistordre_points(pts1_px, K1, D1)
    pts2_ud = undistordre_points(pts2_px, K2, D2)

    # Mire = Origine (0,0,0)
    wXs_3d = np.array([coords_3d_aruco(mid) for mid in ids_sel], dtype=np.float64)
    wXs_h  = np.hstack([wXs_3d, np.ones((len(ids_sel), 1))])

    c2xs_norm = normaliser_points(pts2_ud, K2)

    # Caméra 1
    wMc1, c1Mw, _, _ = solvePnP_vers_matrice(wXs_3d, pts1_px, K1, D1)
    
    # Caméra 2
    wMc2, c2Mw, _, _ = solvePnP_vers_matrice(wXs_3d, pts2_px, K2, D2)

    # Optimisation Nelder-Mead
    euler_init = Rot.from_matrix(wMc2[:3, :3]).as_euler('XYZ', degrees=True)
    x0 = np.hstack([wMc2[:3, 3], euler_init])
    
    res = minimize(cost_nelder, x0, args=(wXs_h, c2xs_norm), method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-13, 'maxiter': 200_000})
    
    wMc2_opt = homogeneousMatrix(*res.x)
    c2Mw_opt = np.linalg.inv(wMc2_opt)

    # Matrice relative c2Mc1
    c2Mc1_opt = c2Mw_opt @ wMc1
    
    # --- CALCUL DE LA NORME ---
    t_rel_opt = c2Mc1_opt[:3, 3]
    norm_t = np.linalg.norm(t_rel_opt)
    
    # --- Affichage 3D ---
    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection='3d')
    
    def tracer_camera(ax, wMc, couleur, label, scale=0.04):
        pts = np.array([[0,0,0,1],[-scale,-scale,scale,1],[scale,-scale,scale,1],
                        [scale,scale,scale,1],[-scale,scale,scale,1]])
        world = np.array([wMc @ p for p in pts])
        ordre = [0,1, 0,2, 0,3, 0,4, 1,2, 2,3, 3,4, 4,1]
        ax.plot([world[i][0] for i in ordre], [world[i][1] for i in ordre], [world[i][2] for i in ordre], color=couleur, lw=2)
        ax.scatter(*wMc[:3, 3], c=couleur, s=50)
        ax.text(*wMc[:3, 3], label, color=couleur)

    ax.scatter(wXs_3d[:,0], wXs_3d[:,1], wXs_3d[:,2], c='k', marker='s', label='Mire')
    tracer_camera(ax, wMc1, 'red', 'Cam 1')
    tracer_camera(ax, wMc2_opt, 'blue', 'Cam 2')
    
    oc1 = wMc1[:3, 3]
    oc2 = wMc2_opt[:3, 3]
    ax.quiver(oc1[0], oc1[1], oc1[2],
              oc2[0]-oc1[0], oc2[1]-oc1[1], oc2[2]-oc1[2],
              color='green', lw=2.5, arrow_length_ratio=0.15,
              label=f't |{norm_t:.4f} m|')
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Validation de la Calibration (Fermez la fenêtre pour continuer)')
    plt.legend()
    plt.show() # Bloque l'exécution jusqu'à ce que la fenêtre soit fermée

    return c2Mc1_opt, wMc1, wMc2_opt

# =============================================================================
# 5. BOUCLE PRINCIPALE (TEMPS RÉEL)
# =============================================================================
cap1 = cv2.VideoCapture(ID_CAM1)
cap2 = cv2.VideoCapture(ID_CAM2)

if not cap1.isOpened() or not cap2.isOpened():
    print("❌ Erreur : Impossible d'ouvrir les deux caméras.")
    sys.exit(1)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

stable_frames = 0
capture_active = False # NOUVEAU : Variable pour bloquer la capture avant l'appui sur Entrée

print(f"\n🟢 Démarrage du flux. Objectif : {TARGET_ARUCOS} ArUcos communs.")
print("👉 Placez votre mire et appuyez sur 'ENTRÉE' pour lancer l'analyse de stabilité.")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    marq1 = detecter_aruco(frame1)
    marq2 = detecter_aruco(frame2)
    communs = set(marq1.keys()) & set(marq2.keys())
    nb_communs = len(communs)

    # Affichage des marqueurs et du texte
    for mid, centre in marq1.items():
        couleur = (0, 255, 0) if mid in communs else (0, 165, 255)
        pt = tuple(centre.astype(int))
        cv2.circle(frame1, pt, 5, couleur, -1)
        cv2.putText(frame1, str(mid), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)

    for mid, centre in marq2.items():
        couleur = (0, 255, 0) if mid in communs else (0, 165, 255)
        pt = tuple(centre.astype(int))
        cv2.circle(frame2, pt, 5, couleur, -1)
        cv2.putText(frame2, str(mid), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)

    couleur_texte = (0, 255, 0) if nb_communs >= TARGET_ARUCOS else (0, 0, 255)
    cv2.putText(frame1, f"Communs: {nb_communs} / {TARGET_ARUCOS}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, couleur_texte, 2)
    
    # MODIFICATION : On ne vérifie la stabilité que si l'utilisateur a appuyé sur Entrée
    if capture_active:
        if nb_communs >= TARGET_ARUCOS:
            stable_frames += 1
            cv2.putText(frame1, f"Stable: {stable_frames}/10", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        else:
            stable_frames = 0
            cv2.putText(frame1, "Perte de stabilite...", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame1, "Appuyez sur ENTREE pour calibrer", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Cam 1', frame1)
    cv2.imshow('Cam 2', frame2)

    # MODIFICATION : Gestion des touches du clavier
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("❌ Quitté par l'utilisateur.")
        break
    elif key == 13: # 13 correspond à la touche ENTRÉE
        capture_active = True
        stable_frames = 0
        print("\n▶️ Analyse de stabilité en cours (10 frames requises)...")

    # MODIFICATION : Seuil passé de 3 à 10 frames
    if stable_frames >= 10:
        print("\n⏸️ 10 frames stables atteintes. Flux en pause. Calcul en cours...")
        
        c2Mc1, wMc1, wMc2 = executer_calibration(marq1, marq2, communs)
        
        choix = input("👉 Cette calibration vous convient-elle ? (O/N) : ").strip().lower()
        
        if choix == 'o':
            R_rel = c2Mc1[:3, :3]
            t_rel = c2Mc1[:3, 3]
            np.save(os.path.join(DOSSIER_SORTIE, "R_c2_c1.npy"), R_rel)
            np.save(os.path.join(DOSSIER_SORTIE, "t_c2_c1.npy"), t_rel)
            np.save(os.path.join(DOSSIER_SORTIE, "c2Mc1.npy"), c2Mc1)
            np.save(os.path.join(DOSSIER_SORTIE, "wMc1.npy"), wMc1)
            np.save(os.path.join(DOSSIER_SORTIE, "wMc2.npy"), wMc2)
            print(f"💾 Matrices sauvegardées dans {DOSSIER_SORTIE}. Fin du programme.")
            break
        else:
            try:
                nouv_seuil = int(input(f"Entrez le nouveau nombre d'ArUcos cible (actuel = {TARGET_ARUCOS}) : "))
                TARGET_ARUCOS = nouv_seuil
            except ValueError:
                print("Entrée invalide. Le seuil reste le même.")
            
            # MODIFICATION : Réinitialisation si on refuse la calibration
            stable_frames = 0
            capture_active = False # On repasse en attente d'un nouvel appui sur Entrée
            print("\n▶️ Reprise du flux vidéo... Replacez la mire et appuyez sur ENTRÉE.")

cap1.release()
cap2.release()
cv2.destroyAllWindows()