import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
df_a = pd.read_csv('Balade_ROV_final0012_6D.tsv', sep='\t', skiprows=11)
df_b = pd.read_csv('trajectoire_robot_clean.tsv', sep='\t')

# --- Qualisys ---
ax_x_a = df_a['bluerovSysmer X'].values / 1000.0
ax_y_a = df_a['Y'].values / 1000.0
ax_z_a = df_a['Z'].values / 1000.0
mask = (ax_x_a != 0) | (ax_y_a != 0)
ax_x_a, ax_y_a, ax_z_a = ax_x_a[mask], ax_y_a[mask], ax_z_a[mask]

# --- YOLO : coordonnées brutes dans le repère caméra ---
pts_yolo_raw = np.column_stack([
    df_b['X(m)'].values,
    df_b['Y(m)'].values,
    df_b['Z(m)'].values
])   # shape (N, 3)

# =============================================================================
# 2. CHANGEMENT DE REPÈRE PAR MATRICE DE ROTATION
# =============================================================================
# Le repère YOLO (caméra) et le repère Qualisys (monde) ont des conventions
# d'axes différentes. On exprime ce changement de repère comme une rotation R
# telle que :  p_qualisys = R @ p_yolo
#
# Correspondance identifiée expérimentalement :
#   X_qualisys = -X_yolo     (inversion gauche/droite)
#   Y_qualisys = -Z_yolo     (profondeur caméra → axe latéral monde, inversé)
#   Z_qualisys = -Y_yolo     (axe vertical caméra → hauteur monde, inversé)
#
# Ce qui donne la matrice de rotation ci-dessous (det = +1 → rotation propre) :
#
#         Xq   Yq   Zq
#   Xy  [1    0    0 ]
#   Yy  [ 0    0   1 ]
#   Zy  [ 0   -1    0 ]

R_yolo_to_qualisys = np.array([
    [-1,  0,  0],
    [ 0,  0, -1],
    [ 0, -1,  0]
], dtype=float)

# Vérification : det doit être +1 pour une rotation propre
assert np.isclose(np.linalg.det(R_yolo_to_qualisys), 1.0), \
    "Attention : R n'est pas une rotation propre (det ≠ 1) !"

# Application vectorielle : (R @ pts.T).T transforme chaque ligne (point 3D)
pts_yolo_aligned = (R_yolo_to_qualisys @ pts_yolo_raw.T).T  # shape (N, 3)

ax_x_b = pts_yolo_aligned[:, 0]
ax_y_b = pts_yolo_aligned[:, 1]
ax_z_b = -pts_yolo_aligned[:, 2]

# =============================================================================
# 3. CENTRAGE PAR RAPPORT AU DÉBUT DE LA TRAJECTOIRE
# =============================================================================
# On centre sur les N_DEBUT premiers points pour éviter que la boucle
# ne biaise le calcul de la position de référence.
N_DEBUT = 50
ax_x_a -= np.mean(ax_x_a[:N_DEBUT]); ax_y_a -= np.mean(ax_y_a[:N_DEBUT]); ax_z_a -= np.mean(ax_z_a[:N_DEBUT])
ax_x_b -= np.mean(ax_x_b[:N_DEBUT]); ax_y_b -= np.mean(ax_y_b[:N_DEBUT]); ax_z_b -= np.mean(ax_z_b[:N_DEBUT])

# Axes temporels normalisés (0 → 1)
t_a_full = np.linspace(0, 1, len(ax_x_a))
t_b_full = np.linspace(0, 1, len(ax_x_b))

# =============================================================================
# 4. OPTIMISATION DE L'ALIGNEMENT SUR LA LIGNE DROITE INITIALE
# =============================================================================
# On optimise uniquement une translation résiduelle (dx, dy, dz) + un décalage
# temporel (shift_t) sur les premiers FRACTION_LIGNE_DROITE du parcours.
# La rotation a déjà été traitée proprement par R ci-dessus.

FRACTION_LIGNE_DROITE = 0.20
mask_a_opt = t_a_full <= FRACTION_LIGNE_DROITE

def erreur_superposition_debut(params):
    dx, dy, dz, shift_t = params
    t_b_decale = t_b_full + shift_t

    f_x = interp1d(t_b_decale, ax_x_b + dx, kind='linear', bounds_error=False, fill_value=np.nan)
    f_y = interp1d(t_b_decale, ax_y_b + dy, kind='linear', bounds_error=False, fill_value=np.nan)
    f_z = interp1d(t_b_decale, ax_z_b + dz, kind='linear', bounds_error=False, fill_value=np.nan)

    x_b_sync = f_x(t_a_full[mask_a_opt])
    y_b_sync = f_y(t_a_full[mask_a_opt])
    z_b_sync = f_z(t_a_full[mask_a_opt])

    masque_valide = ~np.isnan(x_b_sync)
    if np.sum(masque_valide) < len(t_a_full[mask_a_opt]) * 0.2:
        return np.inf

    return np.mean(
        (ax_x_a[mask_a_opt][masque_valide] - x_b_sync[masque_valide])**2 +
        (ax_y_a[mask_a_opt][masque_valide] - y_b_sync[masque_valide])**2 +
        (ax_z_a[mask_a_opt][masque_valide] - z_b_sync[masque_valide])**2
    )

print(f"Recherche de la meilleure superposition sur les premiers {FRACTION_LIGNE_DROITE*100:.0f}% du parcours...")
res = minimize(erreur_superposition_debut, x0=[0.0, 0.0, 0.0, 0.0], method='Nelder-Mead')
best_dx, best_dy, best_dz, best_shift = res.x

print(f"Correction trouvée : dX={best_dx:.3f} m  dY={best_dy:.3f} m  dZ={best_dz:.3f} m")
print(f"Décalage temporel  : {best_shift:.4f}")

# =============================================================================
# 5. APPLICATION À TOUTE LA COURBE ET CALCUL DES RMSE
# =============================================================================
t_b_decale_final = t_b_full + best_shift

f_x_final = interp1d(t_b_decale_final, ax_x_b + best_dx, kind='linear', bounds_error=False, fill_value=np.nan)
f_y_final = interp1d(t_b_decale_final, ax_y_b + best_dy, kind='linear', bounds_error=False, fill_value=np.nan)
f_z_final = interp1d(t_b_decale_final, ax_z_b + best_dz, kind='linear', bounds_error=False, fill_value=np.nan)

x_b_sync_final = f_x_final(t_a_full)
y_b_sync_final = f_y_final(t_a_full)
z_b_sync_final = f_z_final(t_a_full)

masque_final = ~np.isnan(x_b_sync_final)

rmse_x  = np.sqrt(np.mean((ax_x_a[masque_final] - x_b_sync_final[masque_final])**2))
rmse_y  = np.sqrt(np.mean((ax_y_a[masque_final] - y_b_sync_final[masque_final])**2))
rmse_z  = np.sqrt(np.mean((ax_z_a[masque_final] - z_b_sync_final[masque_final])**2))
rmse_3d = np.sqrt(np.mean(
    (ax_x_a[masque_final] - x_b_sync_final[masque_final])**2 +
    (ax_y_a[masque_final] - y_b_sync_final[masque_final])**2 +
    (ax_z_a[masque_final] - z_b_sync_final[masque_final])**2
))

# Mise à jour des tableaux YOLO pour la visualisation
ax_x_b = x_b_sync_final
ax_y_b = y_b_sync_final
ax_z_b = z_b_sync_final

# =============================================================================
# 6. VISUALISATION MULTI-VUES
# =============================================================================
fig = plt.figure(figsize=(20, 8))

# --- Vue 3D ---
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(ax_x_a, ax_y_a, ax_z_a, label='Qualisys', color='green', lw=2)
ax1.plot(ax_x_b, ax_y_b, ax_z_b, label='YOLO', color='red', linestyle='--', lw=1.5)
ax1.set_title("Vue 3D")
ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
ax1.legend()

# --- Vue de dessus (plan X-Y) ---
ax2 = fig.add_subplot(132)
ax2.plot(ax_x_a, ax_y_a, color='green', label='Qualisys', lw=2)
ax2.plot(ax_x_b, ax_y_b, color='red', linestyle='--', label='YOLO')
ax2.scatter(0, 0, color='blue', s=100, label='Départ')
ax2.set_title(f"Vue de dessus (Plan X-Y)\nRMSE XY : {np.sqrt(rmse_x**2 + rmse_y**2):.4f} m")
ax2.set_xlabel("Axe X – Latéral (m)")
ax2.set_ylabel("Axe Y – Profondeur (m)")
ax2.axis('equal')
ax2.grid(True, alpha=0.3)
ax2.legend()

# --- Évolution de la hauteur Z ---
ax3 = fig.add_subplot(133)
ax3.plot(np.linspace(0, 100, len(ax_z_a)), ax_z_a, color='green', label='Qualisys Z', lw=2)
ax3.plot(np.linspace(0, 100, len(ax_z_b)), ax_z_b, color='red', linestyle='--', label='YOLO Z')
ax3.set_title(f"Évolution Hauteur (Z)\nRMSE Z : {rmse_z:.4f} m")
ax3.set_xlabel("Temps (%)")
ax3.set_ylabel("Z (m)")
ax3.grid(True, alpha=0.3)
ax3.legend()

# --- Résumé des scores ---
stats_text = (
    f"BILAN DES ERREURS\n"
    f"RMSE X  : {rmse_x:.4f} m\n"
    f"RMSE Y  : {rmse_y:.4f} m\n"
    f"RMSE Z  : {rmse_z:.4f} m\n"
    f"GLOBAL  : {rmse_3d:.4f} m"
)
plt.figtext(0.5, 0.05, stats_text, ha="center", fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "black"},
            family='monospace')

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()
