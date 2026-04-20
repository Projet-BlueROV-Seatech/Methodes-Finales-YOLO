# -*- coding: utf-8 -*-
"""
=============================================================================
  CALIBRATION DU SOL (REDRESSEMENT) — TEMPS RÉEL
=============================================================================
  - Capture en direct depuis Caméra 1
  - Plateau ArUcos = Sol
  - Déclenchement auto si X ArUcos sur 10 frames consécutives
  - Calcul des axes du monde (Y = Gravité)
  - Affichage 3D bloquant + validation utilisateur en console
=============================================================================
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
DOSSIER_CALIB  = r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\Calibrage Direct"
DOSSIER_SORTIE = DOSSIER_CALIB

ID_CAM1 = 0

# Géométrie de la mire au sol
L_SOL   = 0.088
S_SOL   = 0.028
COL     = 6
LIG     = 6

DICTIONNAIRE = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

# Seuil initial (on recommande au moins 15-20 pour un bon plan)
TARGET_ARUCOS = 15

# =============================================================================
# 2. CHARGEMENT DES INTRINSÈQUES ET EXTRINSÈQUES (Optionnel)
# =============================================================================
print("=" * 60)
print("  CHARGEMENT DES PARAMÈTRES INTRINSÈQUES")
print("=" * 60)

try:
    K1 = np.load(os.path.join(DOSSIER_CALIB, "K1.npy"))
    D1 = np.load(os.path.join(DOSSIER_CALIB, "D1.npy"))
    print(f"✅ K1 et D1 chargés depuis : {DOSSIER_CALIB}")
except FileNotFoundError as e:
    print(f"❌ Erreur : Fichier intrinsèque introuvable. ({e})")
    sys.exit(1)

# Optionnel : Charger Cam2 pour afficher sa hauteur finale
try:
    R_init = np.load(os.path.join(DOSSIER_CALIB, "R_c2_c1.npy"))
    T_init = np.load(os.path.join(DOSSIER_CALIB, "t_c2_c1.npy")).reshape(3, 1)
    has_cam2 = True
except:
    has_cam2 = False

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
    x = c * (L_SOL + S_SOL) + L_SOL / 2.0
    y = r * (L_SOL + S_SOL) + L_SOL / 2.0
    return np.array([x, y, 0.0])

def undistordre_points(pts_px, K, D):
    pts = np.array(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    pts_ud = cv2.undistortPoints(pts, K, D, P=K)
    return pts_ud.reshape(-1, 2)

# =============================================================================
# 4. FONCTION DE CALCUL ET D'AFFICHAGE 3D DU SOL
# =============================================================================
def executer_redressement(marq1):
    print(f"\n🚀 Lancement du calcul du redressement sur {len(marq1)} tags...")
    ids_sel = sorted(list(marq1.keys()))
    
    # On filtre au cas où il y a des tags hors de la mire 6x6 (ID > 35)
    ids_sel = [mid for mid in ids_sel if mid <= 35]

    pts1_px = np.array([marq1[mid] for mid in ids_sel], dtype=np.float64)
    pts1_ud = undistordre_points(pts1_px, K1, D1)

    # Coordonnées 3D de la mire au sol (Z=0)
    objp_sol = np.array([coords_3d_aruco(mid) for mid in ids_sel], dtype=np.float64)

    # SolvePnP avec points undistorded (D=0)
    ok, rvec_sol, tvec_sol = cv2.solvePnP(
        objp_sol, pts1_ud, K1, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        print("❌ solvePnP a échoué.")
        return None, None

    R_sol, _ = cv2.Rodrigues(rvec_sol)

    # --- ÉTAPE 1 : Axe Y monde (Gravité) ---
    n_up = -R_sol[:, 2].copy()
    n_up /= np.linalg.norm(n_up)
    if np.dot(n_up, tvec_sol.flatten()) > 0:
        n_up = -n_up
    world_Y = -n_up

    # --- ÉTAPE 2 : Axe X monde (Latéral) ---
    cam1_X = np.array([1.0, 0.0, 0.0])
    world_X = cam1_X - np.dot(cam1_X, world_Y) * world_Y
    world_X /= np.linalg.norm(world_X)

    # --- ÉTAPE 3 : Axe Z monde (Profondeur) ---
    world_Z = np.cross(world_X, world_Y)
    world_Z /= np.linalg.norm(world_Z)

    R_redressement = np.stack([world_X, world_Y, world_Z], axis=0)

    # --- ÉTAPE 4 : Hauteur ---
    sol_redresse = R_redressement @ tvec_sol.flatten()
    hauteur_cam1 = sol_redresse[1]

    # --- AFFICHAGE CONSOLE ---
    print(f"\n📏 RÉSULTATS :")
    print(f"   CAM1 monde : X=0.000  Y=+{hauteur_cam1:.3f}m  Z=0.000")
    if has_cam2:
        pos_cam2_w = R_redressement @ (-R_init.T @ T_init).flatten()
        print(f"   CAM2 monde : X={pos_cam2_w[0]:+.3f}  Y={hauteur_cam1 - pos_cam2_w[1]:+.3f}m  Z={pos_cam2_w[2]:+.3f}")

    # --- AFFICHAGE 3D ---
    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection='3d')
    
    # Centre de la caméra 1 = (0,0,0) dans le repère caméra
    ax.scatter(0, 0, 0, c='red', s=100, label='Cam 1')
    
    # Tracer les axes redressés (Nouveau monde)
    ax.quiver(0, 0, 0, world_X[0], world_X[1], world_X[2], color='r', length=0.2, label='X (Latéral)')
    ax.quiver(0, 0, 0, world_Y[0], world_Y[1], world_Y[2], color='g', length=0.2, label='Y (Gravité)')
    ax.quiver(0, 0, 0, world_Z[0], world_Z[1], world_Z[2], color='b', length=0.2, label='Z (Profondeur)')

    # Tracer le sol (approx)
    sol_pts = np.array([[x, y, 0] for x in [-0.5, 0.5] for y in [-0.5, 0.5]])
    sol_cam = np.array([R_sol @ pt + tvec_sol.flatten() for pt in sol_pts])
    ax.plot_trisurf(sol_cam[:,0], sol_cam[:,1], sol_cam[:,2], color='yellow', alpha=0.3)

    ax.set_xlabel('X Cam'); ax.set_ylabel('Y Cam'); ax.set_zlabel('Z Cam')
    # On inverse Z et Y pour un affichage matplotlib plus intuitif par rapport à OpenCV
    ax.set_zlim(0, 1.5); ax.invert_yaxis()
    
    ax.set_title(f'Validation du Sol - Hauteur Cam1: {hauteur_cam1:.2f}m\n(Fermez la fenêtre pour continuer)')
    plt.legend()
    plt.show() # Bloque l'exécution jusqu'à ce que la fenêtre soit fermée

    return R_redressement, np.array([hauteur_cam1])

# =============================================================================
# 5. BOUCLE PRINCIPALE (TEMPS RÉEL)
# =============================================================================
cap1 = cv2.VideoCapture(ID_CAM1)

if not cap1.isOpened():
    print(f"❌ Erreur : Impossible d'ouvrir la caméra {ID_CAM1}.")
    sys.exit(1)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

stable_frames = 0
capture_active = False

print(f"\n🟢 Démarrage du flux. Objectif : {TARGET_ARUCOS} ArUcos.")
print("👉 Posez votre mire SUR LE SOL et appuyez sur 'ENTRÉE' pour lancer l'analyse de stabilité.")

while True:
    ret1, frame1 = cap1.read()
    if not ret1:
        break

    marq1 = detecter_aruco(frame1)
    nb_aruco = len(marq1)

    # Affichage des marqueurs
    for mid, centre in marq1.items():
        couleur = (0, 255, 0)
        pt = tuple(centre.astype(int))
        cv2.circle(frame1, pt, 5, couleur, -1)
        cv2.putText(frame1, str(mid), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)

    couleur_texte = (0, 255, 0) if nb_aruco >= TARGET_ARUCOS else (0, 0, 255)
    cv2.putText(frame1, f"Detectes: {nb_aruco} / {TARGET_ARUCOS}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, couleur_texte, 2)
    
    # Gestion de la stabilité
    if capture_active:
        if nb_aruco >= TARGET_ARUCOS:
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

    cv2.imshow('Calibrage Sol - Cam 1', frame1)

    # Gestion du clavier
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("❌ Quitté par l'utilisateur.")
        break
    elif key == 13: # 13 correspond à la touche ENTRÉE
        capture_active = True
        stable_frames = 0
        print("\n▶️ Analyse de stabilité en cours (10 frames requises)...")

    # Si on a 10 frames stables
    if stable_frames >= 10:
        print("\n⏸️ 10 frames stables atteintes. Flux en pause. Calcul en cours...")
        
        R_redressement, h_cam1 = executer_redressement(marq1)
        
        if R_redressement is not None:
            choix = input("👉 Ce redressement de sol vous convient-il ? (O/N) : ").strip().lower()
            
            if choix == 'o':
                np.save(os.path.join(DOSSIER_SORTIE, "R_redressement.npy"), R_redressement)
                np.save(os.path.join(DOSSIER_SORTIE, "hauteur_cam1.npy"), h_cam1)
                print(f"💾 Matrices sauvegardées dans {DOSSIER_SORTIE}. Fin du programme.")
                break
            else:
                try:
                    nouv_seuil = int(input(f"Entrez le nouveau nombre d'ArUcos cible (actuel = {TARGET_ARUCOS}) : "))
                    TARGET_ARUCOS = nouv_seuil
                except ValueError:
                    print("Entrée invalide. Le seuil reste le même.")
                
                # Réinitialisation si on refuse
                stable_frames = 0
                capture_active = False
                print("\n▶️ Reprise du flux vidéo... Replacez la mire au sol et appuyez sur ENTRÉE.")
        else:
            # En cas d'échec du solvePnP
            stable_frames = 0
            capture_active = False
            print("\n▶️ Reprise du flux vidéo... L'algorithme n'a pas pu résoudre la position.")

cap1.release()
cv2.destroyAllWindows()