import cv2
import cv2.aruco as aruco
import numpy as np
import os

# =================================================================
# 1. CHARGEMENT DES MATRICES
# =================================================================
dossier = r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\Calibrage Direct"

try:
    K1     = np.load(os.path.join(dossier, "K1.npy"))
    D1     = np.load(os.path.join(dossier, "D1.npy"))
    R_init = np.load(os.path.join(dossier, "R_c2_c1.npy"))
    T_init = np.load(os.path.join(dossier, "t_c2_c1.npy")).reshape(3, 1)
except Exception as e:
    print(f"❌ Erreur de fichiers : {e}")
    exit()

# =================================================================
# 2. CONFIGURATION DES MIRES & ARUCO
# =================================================================
COL, LIG = 6, 6

# 1. Dimensions de la GRANDE mire rigide
L, S = 0.088, 0.028

# 2. Dimensions de la PETITE mire imprimée posée au sol
L_SOL, S_SOL = L, S

DICTIONNAIRE = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
params_det = aruco.DetectorParameters()
params_det.markerBorderBits          = 2
params_det.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX
params_det.adaptiveThreshWinSizeMin  = 3
params_det.adaptiveThreshWinSizeMax  = 30
params_det.adaptiveThreshWinSizeStep = 3
params_det.adaptiveThreshConstant    = 7
params_det.minMarkerPerimeterRate    = 0.01
params_det.maxMarkerPerimeterRate    = 4.0
detector = aruco.ArucoDetector(DICTIONNAIRE, params_det)

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

def annoter_image(img, marqueurs, titre=""):
    vis = img.copy()
    couleurs = [(0,255,0),(0,128,255),(255,0,0),(255,0,255),(0,255,255)]
    for rang, mid in enumerate(marqueurs.keys()):
        cx, cy = marqueurs[mid].astype(int)
        col = couleurs[rang % len(couleurs)]
        cv2.circle(vis, (cx, cy), 12, col, -1)
        cv2.circle(vis, (cx, cy), 16, (255,255,255), 2)
        cv2.putText(vis, f"ID{mid}", (cx+18, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    if titre:
        col_titre = (0, 255, 0) if len(marqueurs) >= 6 else (0, 0, 255)
        cv2.putText(vis, titre, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col_titre, 3)
    return vis

# =================================================================
# 3. REDRESSEMENT DU REPÈRE (solvePnP)
# =================================================================
IMAGE_SOL = os.path.join(dossier, "Image_mire.jpg") 

R_redressement = np.eye(3)
hauteur_cam1   = 0.0
sol_actif      = False

if os.path.exists(IMAGE_SOL):
    print("🔧 Calcul du redressement depuis :", IMAGE_SOL)
    img_sol = cv2.imread(IMAGE_SOL)

    if img_sol is not None:
        marq_sol = detecter_aruco(img_sol)

        visu_sol = annoter_image(img_sol, marq_sol, f"Tags trouves : {len(marq_sol)} / 6 min")
        h, w = visu_sol.shape[:2]
        nom_fenetre = "Verification Image Sol (Appuie sur UNE TOUCHE pour continuer)"
        cv2.imshow(nom_fenetre, cv2.resize(visu_sol, (1000, int(1000 * h / w))))
        print("👀 Fenêtre de vérification ouverte. Appuie sur une touche pour continuer...")
        cv2.waitKey(0)
        cv2.destroyWindow(nom_fenetre)

        if len(marq_sol) >= 6:
            obj_points = []
            img_points_px = []
            
            for mid, centre in marq_sol.items():
                if mid > 35: continue
                row          = mid // COL
                physical_col = (COL - 1) - (mid % COL)
                
                cx = physical_col * (L_SOL + S_SOL) + L_SOL / 2.0
                cy = row          * (L_SOL + S_SOL) + L_SOL / 2.0
                
                obj_points.append([cx, cy, 0.0])
                img_points_px.append(centre)

            objp_sol = np.array(obj_points, dtype=np.float32)
            imgp_sol_px = np.array(img_points_px, dtype=np.float32)

            imgp_sol_ud = cv2.undistortPoints(imgp_sol_px.reshape(-1, 1, 2), K1, D1, P=K1).reshape(-1, 2)

            ok, rvec_sol, tvec_sol = cv2.solvePnP(
                objp_sol, imgp_sol_ud, K1, np.zeros(5),
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if ok:
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
                sol_actif      = True

                # --- ÉTAPE 4 : Hauteur ---
                sol_redresse = R_redressement @ tvec_sol.flatten()
                hauteur_cam1 = sol_redresse[1]

                print(f"   CAM1 monde : X=0.000  Y=+{hauteur_cam1:.3f}  Z=0.000")
                pos_cam2_w = R_redressement @ (-R_init.T @ T_init).flatten()
                print(f"   CAM2 monde : X={pos_cam2_w[0]:+.3f}  Y={hauteur_cam1 - pos_cam2_w[1]:+.3f}  Z={pos_cam2_w[2]:+.3f}")

                np.save(os.path.join(dossier, "R_redressement.npy"), R_redressement)
                np.save(os.path.join(dossier, "hauteur_cam1.npy"), np.array([hauteur_cam1]))
                print("   💾 Matrices de redressement et hauteur sauvegardées.")
            else:
                print("   ❌ solvePnP échoué.")
        else:
            print(f"   ❌ Trop peu de tags détectés (Trouvé: {len(marq_sol)}).")
    else:
        print(f"   ❌ Impossible de lire {IMAGE_SOL}.")