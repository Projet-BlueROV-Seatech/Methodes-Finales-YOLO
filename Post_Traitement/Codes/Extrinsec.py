# -*- coding: utf-8 -*-
"""
=============================================================================
  CALIBRATION EXTRINSÈQUE STÉRÉO — À partir de 2 images
=============================================================================
  Pipeline :
    1. Détection automatique des ArUcos communs dans les 2 images
    2. Validation visuelle des points (clic souris)
    3. Calcul des positions 3D sur la mire
    4. solvePnP sur chaque caméra
    5. Minimisation Nelder-Mead
    6. Pose relative  c2Mc1  =  rotation + translation de cam2 / cam1
    7. Sauvegarde + visualisation 3D

CORRECTIONS APPLIQUÉES :
  [1] coords_3d_aruco : Y n'est plus inversé
      L'ancienne version appliquait r = LIG-1-row, retournant le sens des lignes.
      La position physique correcte est y = row * (L+S) + L/2.

  [2] wMmire : utilise maintenant c1Mmire (mire→cam1=monde) et non mireMc1
      solvePnP_vers_matrice retourne (wMc=cam→monde, cMw=monde→cam).
      Pour exprimer des points mire en coords monde=cam1, il faut cMw=c1Mmire.
      L'ancienne version utilisait wMc=mireMc1, qui transforme cam1→mire,
      plaçant les points 3D dans une direction complètement fausse.

  [3] reprojection_error : n'applique plus la distorsion sur des pts déjà undistortés.
      Les pts undistortés (pts1_ud) doivent être comparés à des projections
      sans distorsion (np.zeros), pas avec D1/D2.
=============================================================================
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
import os

# =============================================================================
# 1. CONFIGURATION — À ADAPTER
# =============================================================================

IMAGE_CAM1    = r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\Calibrage Direct\output.jpg"
IMAGE_CAM2    = r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\Calibrage Direct\output2.jpg"
DOSSIER_CALIB = r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\Calibrage Direct"

L    = 0.088
S    = 0.028
COL  = 6
LIG  = 6

IDS_PREFERES = None
NB_POINTS    = 10

DICTIONNAIRE   = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
DOSSIER_SORTIE = DOSSIER_CALIB

# =============================================================================
# 2. CHARGEMENT DES PARAMÈTRES INTRINSÈQUES
# =============================================================================

print("=" * 60)
print("  CHARGEMENT DES PARAMÈTRES INTRINSÈQUES")
print("=" * 60)

K1 = np.load(os.path.join(DOSSIER_CALIB, "K1.npy"))
D1 = np.load(os.path.join(DOSSIER_CALIB, "D1.npy"))
K2 = np.load(os.path.join(DOSSIER_CALIB, "K2.npy"))
D2 = np.load(os.path.join(DOSSIER_CALIB, "D2.npy"))
print(f"✅ K et D chargés depuis : {DOSSIER_CALIB}")
print(f"K1 =\n{np.round(K1, 2)}\n")

# =============================================================================
# 3. DÉTECTEUR ARUCO
# =============================================================================

params_det = aruco.DetectorParameters()
params_det.markerBorderBits          = 2
params_det.adaptiveThreshWinSizeMin  = 3
params_det.adaptiveThreshWinSizeMax  = 30
params_det.adaptiveThreshWinSizeStep = 3
params_det.adaptiveThreshConstant    = 7
params_det.minMarkerPerimeterRate    = 0.01
params_det.maxMarkerPerimeterRate    = 4.0
detector = aruco.ArucoDetector(DICTIONNAIRE, params_det)

# =============================================================================
# 4. FONCTIONS UTILITAIRES
# =============================================================================

def charger_image(chemin):
    img_array = np.fromfile(chemin, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger : {chemin}")
    return img


def detecter_aruco(img):
    """Détecte les ArUcos. Retourne dict {id: centre_pixel (float)}."""
    largeur_cible = 2000
    ratio = largeur_cible / float(img.shape[1])
    img_res = cv2.resize(img, (largeur_cible, int(img.shape[0] * ratio)),
                         interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    marqueurs = {}
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            centre = corners[i][0].mean(axis=0) / ratio
            marqueurs[int(mid)] = centre
    return marqueurs


def coords_3d_aruco(mid):
    """
    Position 3D du centre d'un marqueur sur la mire (repère mire, Z=0).

    CORRECTION [1] : Y n'est plus inversé.
    -------------------------------------------------------
    Disposition physique de la mire (vue de face) :
      Ligne 0 (haut) : IDs 5, 4, 3, 2, 1, 0  (col 0 → col 5)
      Ligne 5 (bas)  : IDs 35,34,33,32,31,30

    Mapping : row = mid // COL,  physical_col = (COL-1) - (mid % COL)
    X (latéral) croît vers la droite, Y (vertical) croît vers le bas.
    L'ancienne version calculait r = LIG-1-row et utilisait r pour y,
    ce qui retournait verticalement le modèle 3D et corrompait solvePnP.
    """
    row          = mid // COL                   # ligne physique  (0=haut … 5=bas)
    physical_col = (COL - 1) - (mid % COL)     # colonne physique (0=gauche … 5=droite)
    x = physical_col * (L + S) + L / 2.0
    y = row          * (L + S) + L / 2.0       # ✅ FIX : y = row (plus de (LIG-1-row))
    return np.array([x, y, 0.0])


def undistordre_points(pts_px, K, D):
    """Corrige la distorsion. Retourne les points en pixels undistordus."""
    pts    = np.array(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    pts_ud = cv2.undistortPoints(pts, K, D, P=K)
    return pts_ud.reshape(-1, 2)


def normaliser_points(pts_px, K):
    """Pixel → coordonnées normalisées (plan Z=1, K=I)."""
    K_inv = np.linalg.inv(K)
    pts_h = np.hstack([pts_px, np.ones((len(pts_px), 1))])
    return (K_inv @ pts_h.T).T


def annoter_image(img, marqueurs, ids_sel, titre=""):
    vis     = img.copy()
    couleurs = [(0,255,0),(0,128,255),(255,0,0),(255,0,255),(0,255,255)]
    for rang, mid in enumerate(ids_sel):
        if mid in marqueurs:
            cx, cy = marqueurs[mid].astype(int)
            col = couleurs[rang % len(couleurs)]
            cv2.circle(vis, (cx, cy), 12, col, -1)
            cv2.circle(vis, (cx, cy), 16, (255,255,255), 2)
            cv2.putText(vis, f"ID{mid}(#{rang+1})", (cx+18, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    if titre:
        cv2.putText(vis, titre, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,255), 3)
    return vis


def homogeneousMatrix(tx, ty, tz, rx, ry, rz):
    """Matrice homogène 4×4 depuis translation + angles Euler XYZ (degrés)."""
    Rmat = Rot.from_euler('XYZ', [rx, ry, rz], degrees=True).as_matrix()
    Tvec = np.array([[tx, ty, tz]]).T
    return np.block([[Rmat, Tvec], [np.array([[0,0,0,1]])]])


# =============================================================================
# 5. DÉTECTION DANS LES 2 IMAGES
# =============================================================================

print("\n" + "=" * 60)
print("  DÉTECTION ARUCO")
print("=" * 60)

img1  = charger_image(IMAGE_CAM1)
img2  = charger_image(IMAGE_CAM2)
marq1 = detecter_aruco(img1)
marq2 = detecter_aruco(img2)

print(f"Caméra 1 : {len(marq1)} marqueurs — IDs : {sorted(marq1.keys())}")
print(f"Caméra 2 : {len(marq2)} marqueurs — IDs : {sorted(marq2.keys())}")

communs = set(marq1.keys()) & set(marq2.keys())
print(f"Communs  : {len(communs)} — IDs : {sorted(communs)}")

if len(communs) < NB_POINTS:
    raise RuntimeError(
        f"❌ Seulement {len(communs)} marqueurs communs (minimum requis : {NB_POINTS}).")

if IDS_PREFERES and all(i in communs for i in IDS_PREFERES[:NB_POINTS]):
    ids_selectionnes = IDS_PREFERES[:NB_POINTS]
else:
    ids_selectionnes = sorted(communs)[:NB_POINTS]

print(f"\n✅ IDs retenus : {ids_selectionnes}")

# =============================================================================
# 6. VALIDATION VISUELLE
# =============================================================================

print("\n" + "=" * 60)
print("  VALIDATION VISUELLE")
print("=" * 60)
print("  → ENTRÉE = valider | Q = annuler")

vis1 = annoter_image(img1, marq1, ids_selectionnes, "CAM 1")
vis2 = annoter_image(img2, marq2, ids_selectionnes, "CAM 2")

MAX_H = 400
def redim(img, max_h):
    h, w = img.shape[:2]
    r = max_h / h
    return cv2.resize(img, (int(w*r), max_h))

combined = np.hstack([redim(vis1, MAX_H), redim(vis2, MAX_H)])
cv2.imshow("Validation ArUco — ENTREE=OK | Q=Quitter", combined)
key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

if key == ord('q') or key == 27:
    print("❌ Annulé.")
    exit(0)

# =============================================================================
# 7. EXTRACTION DES POINTS 2D ET 3D
# =============================================================================

pts1_px = np.array([marq1[mid] for mid in ids_selectionnes], dtype=np.float64)
pts2_px = np.array([marq2[mid] for mid in ids_selectionnes], dtype=np.float64)

pts1_ud = undistordre_points(pts1_px, K1, D1)
pts2_ud = undistordre_points(pts2_px, K2, D2)

mireXs_3d = np.array([coords_3d_aruco(mid) for mid in ids_selectionnes], dtype=np.float64)

print("\nPoints 3D sur la mire (repère mire) :")
print(f"  {'ID':<5} {'X(m)':>8} {'Y(m)':>8} {'Z(m)':>6}")
for mid, pt in zip(ids_selectionnes, mireXs_3d):
    print(f"  {mid:<5} {pt[0]:>8.4f} {pt[1]:>8.4f} {pt[2]:>6.4f}")

c1xs_norm = normaliser_points(pts1_ud, K1)
c2xs_norm = normaliser_points(pts2_ud, K2)

# Vérification rapide : les points 3D doivent couvrir toute la mire
extents = mireXs_3d.max(axis=0) - mireXs_3d.min(axis=0)
mire_physique = np.array([(COL-1)*(L+S)+L, (LIG-1)*(L+S)+L, 0])
print(f"\n📐 Étendue 3D calculée  : X={extents[0]:.3f}m  Y={extents[1]:.3f}m")
print(f"   Étendue physique mire : X={mire_physique[0]:.3f}m  Y={mire_physique[1]:.3f}m")
if abs(extents[0] - mire_physique[0]) > 0.01 or abs(extents[1] - mire_physique[1]) > 0.01:
    print("   ⚠️  Écart détecté — vérifier coords_3d_aruco ou ids_selectionnes")
else:
    print("   ✅ Cohérent avec la mire physique")

# =============================================================================
# 8. solvePnP — REPÈRE MONDE ALIGNÉ SUR CAM1
# =============================================================================

print("\n" + "=" * 60)
print("  solvePnP : POSE DANS LE MONDE (Origine = Cam1)")
print("=" * 60)

def solvePnP_vers_matrice(pts3d, pts2d_px, K, D):
    """
    Retourne (wMc, cMw, rvec, tvec) :
      cMw : transforme FROM monde TO caméra  (x_cam = cMw @ x_monde)
      wMc : transforme FROM caméra TO monde  (x_monde = wMc @ x_cam)
    """
    success, rvec, tvec = cv2.solvePnP(
        pts3d.astype(np.float64), pts2d_px.astype(np.float64),
        K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    R_mat, _ = cv2.Rodrigues(rvec)
    cMw = np.eye(4)
    cMw[:3, :3] = R_mat
    cMw[:3,  3] = tvec.flatten()
    wMc = np.linalg.inv(cMw)
    return wMc, cMw, rvec, tvec

# Cam1 est l'origine absolue du repère monde
wMc1 = np.eye(4)

# solvePnP pose de la mire par rapport à Cam1
# Monde = repère de la mire, Caméra = Cam1
# → cMw = c1Mmire : transforme mire → cam1
# → wMc = mireMc1 : transforme cam1 → mire (position de cam1 dans repère mire)
mireMc1, c1Mmire, rvec1, tvec1 = solvePnP_vers_matrice(mireXs_3d, pts1_ud, K1, D1)

# =============================================================================
# CORRECTION [2] : wMmire utilise maintenant c1Mmire
# =============================================================================
# Objectif : exprimer les points de la mire dans le repère monde = cam1.
# x_monde = x_cam1 = c1Mmire @ x_mire_homogene
#
# c1Mmire = cMw (2ème valeur retournée) : transforme mire → cam1  ✓
# mireMc1 = wMc (1ère valeur retournée) : transforme cam1 → mire  ✗
#
# L'ancienne version utilisait mireMc1 (la MAUVAISE matrice), ce qui plaçait
# les points 3D dans une direction arbitraire et faussait toute la suite.
wMmire = wMc1 @ c1Mmire   # ✅ FIX : c1Mmire au lieu de mireMc1

wXs_3d = np.array([(wMmire @ np.append(pt, 1))[:3] for pt in mireXs_3d])
wXs_h  = np.hstack([wXs_3d, np.ones((len(wXs_3d), 1))])

pos_cam1 = wMc1[:3, 3]
print(f"📍 Position Cam1 (origine) : [0, 0, 0]")

# solvePnP pour Cam2 dans le repère monde
wMc2, c2Mw, rvec2, tvec2 = solvePnP_vers_matrice(wXs_3d, pts2_ud, K2, D2)
pos_cam2 = wMc2[:3, 3]
print(f"📍 Position Cam2 : [{pos_cam2[0]:.4f}, {pos_cam2[1]:.4f}, {pos_cam2[2]:.4f}] m")
print(f"   Distance inter-caméras (solvePnP) : {np.linalg.norm(pos_cam2 - pos_cam1):.4f} m")

# =============================================================================
# CORRECTION [3] : reprojection_error avec distorsion nulle sur pts undistortés
# =============================================================================
# pts1_ud / pts2_ud sont déjà corrigés de la distorsion.
# cv2.projectPoints avec D1 appliquerait la distorsion une seconde fois.
# On passe np.zeros pour que la comparaison soit cohérente.
D_zero = np.zeros((5, 1))

def reprojection_error(pts3d, pts2d_undist, K, rvec, tvec):
    """Erreur de reprojection sur points undistortés (D=0 implicite)."""
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, D_zero)
    return np.mean(np.linalg.norm(pts2d_undist - proj.reshape(-1, 2), axis=1))

err1 = reprojection_error(mireXs_3d, pts1_ud, K1, rvec1, tvec1)
err2 = reprojection_error(wXs_3d,    pts2_ud, K2, rvec2, tvec2)
print(f"\n✅ Cam1 — erreur reprojection : {err1:.3f} px")
print(f"✅ Cam2 — erreur reprojection : {err2:.3f} px")

# =============================================================================
# 9. OPTIMISATION NELDER-MEAD
# =============================================================================

print("\n" + "=" * 60)
print("  OPTIMISATION NELDER-MEAD")
print("=" * 60)

def changeFrame(oXs, cMo):
    return np.array([cMo @ oX for oX in oXs])

def pinholeProj(cXs):
    xs = []
    for X in cXs:
        if X[2] <= 0:
            return None
        xs.append(np.array([X[0]/X[2], X[1]/X[2], 1.0]))
    return np.array(xs)

def cost_nelder(params, wXs_h, c2xs_norm):
    wMc2_try  = homogeneousMatrix(*params)
    c2Mw_try  = np.linalg.inv(wMc2_try)
    c2Xs      = changeFrame(wXs_h, c2Mw_try)
    c2xs_proj = pinholeProj(c2Xs)
    if c2xs_proj is None:
        return 1e10
    return np.sum((c2xs_norm[:, :2] - c2xs_proj[:, :2])**2)

# Initialisation depuis solvePnP (convergence rapide)
euler_init = Rot.from_matrix(wMc2[:3, :3]).as_euler('XYZ', degrees=True)
x0 = np.hstack([wMc2[:3, 3], euler_init])

print(f"Coût initial (solvePnP → Nelder) : {cost_nelder(x0, wXs_h, c2xs_norm):.6e}")

result = minimize(
    cost_nelder, x0,
    args=(wXs_h, c2xs_norm),
    method='Nelder-Mead',
    options={'xatol': 1e-10, 'fatol': 1e-13, 'maxiter': 200_000, 'adaptive': True}
)

print(f"Coût final                        : {result.fun:.6e}")
print(f"Convergence                       : {result.message}")

wMc2_opt = homogeneousMatrix(*result.x)
c2Mw_opt = np.linalg.inv(wMc2_opt)

# =============================================================================
# 10. POSE RELATIVE c2 PAR RAPPORT À c1
# =============================================================================

print("\n" + "=" * 60)
print("  RÉSULTAT : POSE DE CAM2 PAR RAPPORT À CAM1")
print("=" * 60)

# wMc1 = I → c1Mw = I → c2Mc1 = c2Mw @ wMc1 = c2Mw
c2Mc1_pnp  = c2Mw      @ wMc1
R_rel_pnp  = c2Mc1_pnp[:3, :3]
t_rel_pnp  = c2Mc1_pnp[:3,  3]
euler_pnp  = Rot.from_matrix(R_rel_pnp).as_euler('XYZ', degrees=True)

c2Mc1_opt = c2Mw_opt @ wMc1
R_rel_opt  = c2Mc1_opt[:3, :3]
t_rel_opt  = c2Mc1_opt[:3,  3]
euler_opt  = Rot.from_matrix(R_rel_opt).as_euler('XYZ', degrees=True)

print("\n📐 MÉTHODE solvePnP")
print(f"   |t|  : {np.linalg.norm(t_rel_pnp):.4f} m")
print(f"   t    : [{t_rel_pnp[0]:.5f}, {t_rel_pnp[1]:.5f}, {t_rel_pnp[2]:.5f}]")
print(f"   Euler XYZ : rx={euler_pnp[0]:.3f}°  ry={euler_pnp[1]:.3f}°  rz={euler_pnp[2]:.3f}°")

print("\n📐 MÉTHODE Nelder-Mead (retenue)")
print(f"   |t|  : {np.linalg.norm(t_rel_opt):.4f} m")
print(f"   t    : [{t_rel_opt[0]:.5f}, {t_rel_opt[1]:.5f}, {t_rel_opt[2]:.5f}]")
print(f"   Euler XYZ : rx={euler_opt[0]:.3f}°  ry={euler_opt[1]:.3f}°  rz={euler_opt[2]:.3f}°")
print(f"   R :\n{np.round(R_rel_opt, 5)}")

delta_t = np.linalg.norm(t_rel_pnp - t_rel_opt)
delta_R = np.linalg.norm(R_rel_pnp - R_rel_opt, 'fro')
print(f"\n🔎 Cohérence inter-méthodes : Δt = {delta_t:.2e} m | ΔR = {delta_R:.2e}")
if delta_t > 0.05 or delta_R > 0.1:
    print("   ⚠️  Écart important — vérifier que les marqueurs sont bien identifiés.")
else:
    print("   ✅ Bon accord entre les deux méthodes.")

# =============================================================================
# 11. SAUVEGARDE
# =============================================================================

print("\n" + "=" * 60)
print("  SAUVEGARDE")
print("=" * 60)

np.save(os.path.join(DOSSIER_SORTIE, "R_c2_c1.npy"), R_rel_opt)
np.save(os.path.join(DOSSIER_SORTIE, "t_c2_c1.npy"), t_rel_opt)
np.save(os.path.join(DOSSIER_SORTIE, "c2Mc1.npy"),   c2Mc1_opt)
np.save(os.path.join(DOSSIER_SORTIE, "wMc1.npy"),    wMc1)
np.save(os.path.join(DOSSIER_SORTIE, "wMc2.npy"),    wMc2_opt)

print(f"💾 Sauvegardé dans : {DOSSIER_SORTIE}")

# =============================================================================
# 12. VISUALISATION 3D
# =============================================================================

def tracer_camera(ax, wMc, couleur, label, scale=0.04):
    pts   = np.array([[0,0,0,1],[-scale,-scale,scale,1],[scale,-scale,scale,1],
                      [scale,scale,scale,1],[-scale,scale,scale,1]])
    world = np.array([wMc @ p for p in pts])
    ordre = [0,1, 0,2, 0,3, 0,4, 1,2, 2,3, 3,4, 4,1]
    ax.plot([world[i][0] for i in ordre],
            [world[i][1] for i in ordre],
            [world[i][2] for i in ordre], color=couleur, lw=2.5)
    c = wMc[:3, 3]
    ax.scatter(*c, c=couleur, s=100, zorder=6)
    ax.text(c[0], c[1], c[2]+0.025, label, color=couleur, fontsize=11, fontweight='bold')

fig = plt.figure(figsize=(13, 8))
ax  = fig.add_subplot(111, projection='3d')

# Mire
mx = np.array([0, (COL-1)*(L+S)+L, (COL-1)*(L+S)+L, 0, 0])
my = np.array([0, 0, (LIG-1)*(L+S)+L, (LIG-1)*(L+S)+L, 0])
ax.plot(mx, my, np.zeros(5), 'k--', lw=1.5, alpha=0.5, label='Mire (Z=0)')

# Points ArUco dans le repère monde (cam1)
ax.scatter(wXs_3d[:,0], wXs_3d[:,1], wXs_3d[:,2], c='k', s=120, zorder=7, label='ArUco')
for i, mid in enumerate(ids_selectionnes):
    ax.text(wXs_3d[i,0]+0.005, wXs_3d[i,1]+0.005, wXs_3d[i,2]+0.008, f"ID{mid}", fontsize=7)

tracer_camera(ax, wMc1,     'red',  'Cam 1 (origine)')
tracer_camera(ax, wMc2_opt, 'blue', 'Cam 2 (estimée)')

oc1 = wMc1[:3, 3]
oc2 = wMc2_opt[:3, 3]
for pt in wXs_3d:
    ax.plot([oc1[0],pt[0]], [oc1[1],pt[1]], [oc1[2],pt[2]], 'r:', alpha=0.3, lw=0.8)
    ax.plot([oc2[0],pt[0]], [oc2[1],pt[1]], [oc2[2],pt[2]], 'b:', alpha=0.3, lw=0.8)

ax.quiver(oc1[0], oc1[1], oc1[2],
          oc2[0]-oc1[0], oc2[1]-oc1[1], oc2[2]-oc1[2],
          color='green', lw=2.5, arrow_length_ratio=0.15,
          label=f'baseline |{np.linalg.norm(t_rel_opt):.3f} m|')

ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
ax.set_title('Calibration extrinsèque stéréo — repère cam1', fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(DOSSIER_SORTIE, "calibration_extrinseque_3D.png"), dpi=150)
plt.show()

# =============================================================================
# 13. RÉSUMÉ FINAL
# =============================================================================

print("\n" + "=" * 60)
print("  RÉSUMÉ FINAL")
print("=" * 60)
print(f"\n  Distance inter-caméras : {np.linalg.norm(t_rel_opt):.4f} m")
print(f"\n  Vecteur t (m)  : [{t_rel_opt[0]:.6f}, {t_rel_opt[1]:.6f}, {t_rel_opt[2]:.6f}]")
print(f"\n  Euler XYZ (°)  : rx={euler_opt[0]:.4f}°  ry={euler_opt[1]:.4f}°  rz={euler_opt[2]:.4f}°")
print(f"\n  Matrice R (3×3):")
for row in np.round(R_rel_opt, 6):
    print(f"    {row}")
print(f"\n  Matrice c2Mc1 (4×4):")
for row in np.round(c2Mc1_opt, 6):
    print(f"    {row}")
print("\n✅ Calibration extrinsèque terminée.\n")