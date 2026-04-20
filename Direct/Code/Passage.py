import numpy as np
import os

# Dossier contenant tes fichiers de calibration
DOSSIER = r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\Calibrage Direct"

# 1. Chargement des K et des paramètres stéréo
print("⏳ Chargement des matrices...")
try:
    K1 = np.load(os.path.join(DOSSIER, "K1.npy"))
    K2 = np.load(os.path.join(DOSSIER, "K2.npy"))
    R  = np.load(os.path.join(DOSSIER, "R_c2_c1.npy"))
    T  = np.load(os.path.join(DOSSIER, "t_c2_c1.npy"))
except Exception as e:
    print(f"❌ Erreur de chargement : {e}")
    exit()

# 🛠️ CORRECTION : On force le vecteur T (1D) à devenir une colonne 2D (3x1)
T = T.reshape(3, 1)

# 2. Construction de P1
# P1 = K1 * [I | 0] (La Caméra 1 est l'origine, donc Rotation Identité et Translation Nulle)
I = np.eye(3)
zeros = np.zeros((3, 1))
P1 = K1 @ np.hstack((I, zeros))

# 3. Construction de P2
# P2 = K2 * [R | T] (La position de Caméra 2 par rapport à Caméra 1)
P2 = K2 @ np.hstack((R, T))

# 4. Sauvegarde
np.save(os.path.join(DOSSIER, "P1.npy"), P1)
np.save(os.path.join(DOSSIER, "P2.npy"), P2)

# (Optionnel) Sauvegarder un fichier stereo_params.npz par sécurité pour ton tracker
np.savez(os.path.join(DOSSIER, "stereo_params.npz"), R=R, T=T)

print("✅ Matrices P1 et P2 générées et sauvegardées avec succès !")
print("   -> P1.npy")
print("   -> P2.npy")
print("🚀 Ton tracker 3D est prêt à être lancé.")