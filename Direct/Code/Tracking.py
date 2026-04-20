import cv2
import numpy as np
import os
from ultralytics import YOLO
from collections import deque
import time
import csv

# =================================================================
# 1. CHARGEMENT DES MATRICES DE CALIBRATION
# =================================================================
dossier = r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\Calibrage Direct"

try:
    P1     = np.load(os.path.join(dossier, "P1.npy"))
    P2     = np.load(os.path.join(dossier, "P2.npy"))
    K1     = np.load(os.path.join(dossier, "K1.npy"))
    D1     = np.load(os.path.join(dossier, "D1.npy"))
    K2     = np.load(os.path.join(dossier, "K2.npy"))
    D2     = np.load(os.path.join(dossier, "D2.npy"))
    R_init = np.load(os.path.join(dossier, "R_c2_c1.npy"))
    T_init = np.load(os.path.join(dossier, "t_c2_c1.npy")).reshape(3, 1)
except Exception as e:
    print(f"❌ Erreur de fichiers : {e}")
    exit()

# =================================================================
# 2. CHARGEMENT DU REDRESSEMENT DU REPÈRE
# =================================================================
R_redressement = np.eye(3)
hauteur_cam1   = 0.0
sol_actif      = False

if os.path.exists(os.path.join(dossier, "R_redressement.npy")):
    R_redressement = np.load(os.path.join(dossier, "R_redressement.npy"))
    try:
        hauteur_cam1 = np.load(os.path.join(dossier, "hauteur_cam1.npy"))[0]
    except FileNotFoundError:
        hauteur_cam1 = 0.0
    sol_actif = True
    print(f"✅ R_redressement chargé.")
else:
    print("⚠️ Attention : Fichiers de redressement introuvables. Le repère par défaut sera utilisé.")

pos_cam2_brut = (-R_init.T @ T_init).flatten()
pos_cam2_w    = R_redressement @ pos_cam2_brut
pos_cam2      = np.array([pos_cam2_w[0], hauteur_cam1 - pos_cam2_w[1], pos_cam2_w[2]])
print(f"✅ Prêt. Hauteur caméras calculée : ~{hauteur_cam1:.2f}m")

model = YOLO(r'C:\Users\theoc\Desktop\Projet_SYSMER_2A\runs\detect\entrainement_sysmer_air\weights\best.pt')

# =================================================================
# 3. FONCTIONS TECHNIQUES
# =================================================================
def get_center(box):
    x1, y1, x2, y2 = box.xyxy[0]
    return float((x1 + x2) / 2), float((y1 + y2) / 2)

def undistort_point(point_2d, K, D):
    pt = np.array([[[point_2d[0], point_2d[1]]]], dtype=np.float32)
    return cv2.undistortPoints(pt, K, D, P=K)[0][0]

def triangulate_robot(p1_rect, p2_rect, P1, P2):
    pt1 = np.array([[p1_rect[0]], [p1_rect[1]]], dtype=np.float32)
    pt2 = np.array([[p2_rect[0]], [p2_rect[1]]], dtype=np.float32)
    p4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
    return (p4d[:3] / p4d[3]).flatten()

# =================================================================
# 4. VISUALISEUR 3D CUSTOM 
# =================================================================
C_BG       = (18,  18,  28); C_GRID = (45,  45,  65); C_GRID_ACC = (70,  70, 100)
C_CAM1     = (255, 180,  60); C_CAM2 = (80,  140, 255); C_ROBOT = (60,  220, 100)
C_TRAIL    = (40,  160,  80); C_TEXT = (220, 220, 235); C_TEXT2 = (140, 140, 160)
C_RING     = (55,  55,  80); C_ACCENT = (255, 100,  80)

TRAIL_LEN  = 80
trajectory = deque(maxlen=TRAIL_LEN)

VID_W, VID_H   = 540, 304
MAP_W, MAP_H   = 620, 500
SIDE_W, SIDE_H = 280, 500
INFO_W, INFO_H = 280, 500
TOTAL_W = MAP_W + SIDE_W + INFO_W
TOTAL_H = VID_H + MAP_H

MAP_X_MIN,  MAP_X_MAX  = -4.0, 4.0
MAP_Z_MIN,  MAP_Z_MAX  = -0.5, 7.0
SIDE_Z_MIN, SIDE_Z_MAX = -0.5, 7.0
SIDE_Y_MIN, SIDE_Y_MAX = -0.3, 2.0

stats = {"pos": (0,0,0), "dist": 0.0, "speed": 0.0, "fps": 0.0, "frame": 0, "det1": False, "det2": False, "recording": False}

def world_to_map(X, Z): return int((X - MAP_X_MIN) / (MAP_X_MAX - MAP_X_MIN) * MAP_W), int((1.0 - (Z - MAP_Z_MIN) / (MAP_Z_MAX - MAP_Z_MIN)) * MAP_H)
def world_to_side(Z, Y): return int((Z - SIDE_Z_MIN) / (SIDE_Z_MAX - SIDE_Z_MIN) * SIDE_W), int((1.0 - (Y - SIDE_Y_MIN) / (SIDE_Y_MAX - SIDE_Y_MIN)) * SIDE_H)

def draw_map(canvas, robot_pos):
    cv2.rectangle(canvas, (0, 0), (MAP_W, MAP_H), C_BG, -1)
    for z in np.arange(0, MAP_Z_MAX + 0.5, 0.5):
        py = world_to_map(0, z)[1]
        cv2.line(canvas, (0, py), (MAP_W, py), C_GRID_ACC if z % 1.0 == 0 else C_GRID, 1)
        if z % 1.0 == 0: cv2.putText(canvas, f"{z:.0f}m", (4, py-3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT2, 1)
    for x in np.arange(MAP_X_MIN, MAP_X_MAX + 0.5, 0.5):
        px = world_to_map(x, 0)[0]
        cv2.line(canvas, (px, 0), (px, MAP_H), C_GRID_ACC if x % 1.0 == 0 else C_GRID, 1)
        if x % 1.0 == 0 and x != 0: cv2.putText(canvas, f"{x:+.0f}", (px+2, MAP_H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT2, 1)
    ox, oz = world_to_map(0, 0)
    for r_m in [1, 2, 3, 4, 5]:
        r_px = int(r_m / (MAP_Z_MAX - MAP_Z_MIN) * MAP_H)
        cv2.circle(canvas, (ox, oz), r_px, C_RING, 1, cv2.LINE_AA)
    cv2.line(canvas, (ox, 0), (ox, MAP_H), (60,60,90), 1)

    fov_half, cone_len = 35, int(5.0 / (MAP_Z_MAX - MAP_Z_MIN) * MAP_H)
    for ang in [np.radians(90+fov_half), np.radians(90-fov_half)]:
        cv2.line(canvas, (ox, oz), (int(ox + cone_len * np.cos(ang)), int(oz - cone_len * np.sin(ang))), C_CAM1, 1, cv2.LINE_AA)
    
    c2x, c2z = world_to_map(pos_cam2[0], pos_cam2[2])
    z_ax_cam2 = R_redressement @ (R_init.T @ np.array([0,0,1]))
    ang_base  = np.arctan2(z_ax_cam2[0], z_ax_cam2[2])
    for da in [-np.radians(fov_half), np.radians(fov_half)]:
        cv2.line(canvas, (c2x, c2z), (int(c2x + cone_len * np.sin(ang_base + da)), int(c2z - cone_len * np.cos(ang_base + da))), C_CAM2, 1, cv2.LINE_AA)

    trail_list = list(trajectory)
    for i in range(1, len(trail_list)):
        col = tuple(int(c * (i / len(trail_list))) for c in C_TRAIL)
        cv2.line(canvas, world_to_map(trail_list[i-1][0], trail_list[i-1][2]), world_to_map(trail_list[i][0], trail_list[i][2]), col, 2, cv2.LINE_AA)

    cv2.circle(canvas, (ox, oz), 9, C_CAM1, -1, cv2.LINE_AA); cv2.circle(canvas, (ox, oz), 9, C_TEXT, 1, cv2.LINE_AA); cv2.putText(canvas, "CAM1", (ox+11, oz+5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CAM1, 1)
    cv2.circle(canvas, (c2x, c2z), 9, C_CAM2, -1, cv2.LINE_AA); cv2.circle(canvas, (c2x, c2z), 9, C_TEXT, 1, cv2.LINE_AA); cv2.putText(canvas, "CAM2", (c2x+11, c2z+5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CAM2, 1)
    cv2.line(canvas, (ox, oz), (c2x, c2z), C_GRID_ACC, 1, cv2.LINE_AA)

    if robot_pos is not None:
        rx_m, rz_m = world_to_map(robot_pos[0], robot_pos[2])
        cv2.circle(canvas, (rx_m, rz_m), 7, C_ROBOT, -1, cv2.LINE_AA); cv2.line(canvas, (ox, oz), (rx_m, rz_m), C_ROBOT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"X:{robot_pos[0]:+.2f}  Z:{robot_pos[2]:.2f}m", (rx_m+(12 if rx_m < MAP_W - 130 else -135), rz_m-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ROBOT, 1, cv2.LINE_AA)
    
    cv2.putText(canvas, "VUE SOL (X-Z)" if sol_actif else "VUE CAM1 - pas de correction sol", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT2, 1)
    return canvas

def draw_side(canvas, robot_pos):
    cv2.rectangle(canvas, (0, 0), (SIDE_W, SIDE_H), C_BG, -1)
    for z in np.arange(0, SIDE_Z_MAX+0.5, 0.5):
        px = world_to_side(z, 0)[0]
        cv2.line(canvas, (px, 0), (px, SIDE_H), C_GRID_ACC if z % 1.0 == 0 else C_GRID, 1)
        if z % 1.0 == 0: cv2.putText(canvas, f"{z:.0f}", (px+1, SIDE_H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_TEXT2, 1)
    for y in np.arange(SIDE_Y_MIN, SIDE_Y_MAX+0.25, 0.25):
        py = world_to_side(0, y)[1]
        cv2.line(canvas, (0, py), (SIDE_W, py), C_GRID_ACC if y % 0.5 == 0 else C_GRID, 1)
        if y % 0.5 == 0: cv2.putText(canvas, f"{y:.1f}", (2, py-2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_TEXT2, 1)

    py_sol = world_to_side(0, 0)[1]
    cv2.line(canvas, (0, py_sol), (SIDE_W, py_sol), (80, 200, 80), 1)
    cv2.putText(canvas, "SOL", (2, py_sol-3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80,200,80), 1)

    trail_list = list(trajectory)
    for i in range(1, len(trail_list)):
        col = tuple(int(c * (i / len(trail_list))) for c in C_TRAIL)
        cv2.line(canvas, world_to_side(trail_list[i-1][2], trail_list[i-1][1]), world_to_side(trail_list[i][2], trail_list[i][1]), col, 2, cv2.LINE_AA)

    if robot_pos is not None:
        rx, ry = world_to_side(robot_pos[2], robot_pos[1])
        cv2.line(canvas, (rx, ry), (rx, world_to_side(robot_pos[2], 0)[1]), C_GRID, 1, cv2.LINE_AA)
        cv2.circle(canvas, (rx, ry), 6, C_ROBOT, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"H:{robot_pos[1]:+.2f}m", (rx+8, ry-4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_ROBOT, 1)
    cv2.putText(canvas, "PROFIL (Z-Y hauteur)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT2, 1)
    return canvas

def draw_bar(canvas, y, label, value, vmin, vmax, color, unit=""):
    bw, bh, bx = INFO_W-80, 10, 10
    cv2.rectangle(canvas, (bx,y), (bx+bw, y+bh), C_GRID, -1)
    cv2.rectangle(canvas, (bx,y), (int(bx+np.clip((value-vmin)/(vmax-vmin), 0, 1)*bw), y+bh), color, -1)
    cv2.putText(canvas, label, (bx, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT2, 1)
    cv2.putText(canvas, f"{value:.2f}{unit}", (bx+bw+4, y+bh), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

def draw_info(canvas, robot_pos):
    cv2.rectangle(canvas, (0,0), (INFO_W, INFO_H), C_BG, -1); cv2.line(canvas, (1,0), (1, INFO_H), C_GRID_ACC, 1)
    y = 22; cv2.putText(canvas, "SYSMER TRACKER", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_TEXT, 1)
    
    # Indicateur d'enregistrement
    y += 26
    rec_color = (0, 0, 255) if stats["recording"] else (100, 100, 100)
    rec_text  = "REC EN COURS" if stats["recording"] else "REC PAUSE [R]"
    cv2.circle(canvas, (17, y-4), 5, rec_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, rec_text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, rec_color, 1)

    # Info Sol
    y += 24; cv2.putText(canvas, "Sol : actif" if sol_actif else "Sol : NON calibre", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 255, 180) if sol_actif else (255, 100, 80), 1)
    
    # Infos Caméras
    y += 18
    for cam, det, col in [("CAM1", stats["det1"], C_CAM1), ("CAM2", stats["det2"], C_CAM2)]:
        cv2.circle(canvas, (17, y-4), 5, col if det else C_ACCENT, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"{cam} {'DETECTE' if det else 'ABSENT'}", (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col if det else C_ACCENT, 1)
        y += 20
    
    y += 18
    if robot_pos is not None:
        for axis, desc, col, val in [("X","lateral",(100,180,255),robot_pos[0]),("Y","hauteur",(100,255,180),robot_pos[1]),("Z","profond.",(255,200,100),robot_pos[2])]:
            cv2.putText(canvas, f"{axis} ({desc})", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT2, 1); y += 14
            cv2.putText(canvas, f"  {val:+.3f} m", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1); y += 20
    else:
        cv2.putText(canvas, "En attente...", (10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT2, 1); y += 30
    y += 18; draw_bar(canvas, y, "Distance XZ", stats["dist"], 0, 6, C_ROBOT, "m"); y += 44
    for k, v in [("Frame",f"{stats['frame']}"),("FPS",f"{stats['fps']:.1f}"),("Trail",f"{len(trajectory)} pts")]:
        cv2.putText(canvas, f"{k:<7}{v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT2, 1); y += 16
    y += 18
    for label, col in [("CAM1",C_CAM1),("CAM2",C_CAM2),("Robot",C_ROBOT)]:
        cv2.circle(canvas, (17, y-4), 5, col, -1, cv2.LINE_AA); cv2.putText(canvas, label, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1); y += 18
    
    # Ajout du R dans le bandeau de commandes
    cv2.putText(canvas, "[Q] Quitter   [ESPACE] Pause   [R] Enregistrer", (10, INFO_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT2, 1)
    return canvas

def build_frame(frame1, frame2, robot_pos):
    map_c, side_c, info_c = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8), np.zeros((SIDE_H, SIDE_W, 3), dtype=np.uint8), np.zeros((INFO_H, INFO_W, 3), dtype=np.uint8)
    draw_map(map_c, robot_pos); draw_side(side_c, robot_pos); draw_info(info_c, robot_pos)
    f1, f2 = cv2.resize(frame1, (VID_W, VID_H)), cv2.resize(frame2, (VID_W, VID_H))
    for fr, det, col in [(f1, stats["det1"], C_CAM1), (f2, stats["det2"], C_CAM2)]:
        cv2.putText(fr, "DETECTE" if det else "---", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        for (cx,cy) in [(0,0),(VID_W,0),(0,VID_H),(VID_W,VID_H)]:
            cv2.line(fr,(cx,cy),(cx+(1 if cx==0 else -1)*16,cy),col,2); cv2.line(fr,(cx,cy),(cx,cy+(1 if cy==0 else -1)*16),col,2)
    vid_bar = np.full((VID_H, TOTAL_W, 3), 18, dtype=np.uint8)
    vx = (TOTAL_W - 2*VID_W)//2
    vid_bar[:, vx:vx+VID_W] = f1; cv2.line(vid_bar,(vx+VID_W,0),(vx+VID_W,VID_H),C_GRID_ACC,2)
    vid_bar[:, vx+VID_W+2:vx+2*VID_W+2] = f2
    cv2.putText(vid_bar,"CAM1",(vx+6,VID_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.40,C_CAM1,1); cv2.putText(vid_bar,"CAM2",(vx+VID_W+8,VID_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.40,C_CAM2,1)
    out = np.vstack([vid_bar, np.hstack([map_c, side_c, info_c])])
    for x in [0, MAP_W, MAP_W+SIDE_W]: cv2.line(out,(x,VID_H),(x,TOTAL_H),C_GRID_ACC,1)
    return out

# =================================================================
# 5. BOUCLE DE TRACKING EN DIRECT (Avec Enregistrement TSV)
# =================================================================
# 1. Ouverture des flux en direct
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# 2. Forcer la résolution pour matcher exactement avec la calibration
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Vérification rapide
if not cap1.isOpened() or not cap2.isOpened():
    print("❌ Erreur : Impossible d'ouvrir l'une des caméras (0 ou 2).")
    exit()

# Initialisation Fichier TSV
chemin_tsv = os.path.join(dossier, "trajectoire_robot_temps_reel.tsv")
fichier_tsv = open(chemin_tsv, mode='w', newline='')
writer_tsv = csv.writer(fichier_tsv, delimiter='\t')
writer_tsv.writerow(['Frame', 'Temps(s)', 'X(m)', 'Y(m)', 'Z(m)']) 

paused = False; robot_pos = None
t_fps = time.time(); fps_frames = 0
prev_pos, prev_time = None, time.time()
enregistrement_actif = True

print("\n▶️  Tracking DIRECT démarré — [Q] quitter  [ESPACE] pause  [R] Enregistrement\n")

cv2.namedWindow("SYSMER Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SYSMER Tracker", TOTAL_W, TOTAL_H)

try:
    while True:
        if not paused:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2: break
            
            stats["frame"] += 1; fps_frames += 1
            stats["recording"] = enregistrement_actif
            now = time.time()
            if now - t_fps >= 0.5: stats["fps"] = fps_frames / (now - t_fps); fps_frames = 0; t_fps = now

            res1 = model(frame1, conf=0.5, verbose=False)
            res2 = model(frame2, conf=0.5, verbose=False)
            p1, p2 = None, None
            stats["det1"], stats["det2"] = len(res1[0].boxes) > 0, len(res2[0].boxes) > 0

            if stats["det1"]:
                p1 = get_center(res1[0].boxes[0])
                x1,y1,x2,y2 = res1[0].boxes[0].xyxy[0]; cv2.rectangle(frame1,(int(x1),int(y1)),(int(x2),int(y2)),C_ROBOT,2)
            if stats["det2"]:
                p2 = get_center(res2[0].boxes[0])
                x1,y1,x2,y2 = res2[0].boxes[0].xyxy[0]; cv2.rectangle(frame2,(int(x1),int(y1)),(int(x2),int(y2)),C_ROBOT,2)

            if p1 and p2:
                pt1_c = undistort_point(p1, K1, D1)
                pt2_c = undistort_point(p2, K2, D2)

                X_brut, Y_brut, Z_brut = triangulate_robot(pt1_c, pt2_c, P1, P2)

                P_redresse = R_redressement @ np.array([X_brut, Y_brut, Z_brut])
                
                # ✅ APPLICATION DE LA HAUTEUR
                X = P_redresse[0]
                Y = hauteur_cam1 - P_redresse[1] 
                Z = P_redresse[2]

                robot_pos = (X, Y, Z)
                stats["pos"] = robot_pos
                stats["dist"] = np.sqrt(X**2 + Z**2)

                if prev_pos is not None and now > prev_time:
                    stats["speed"] = np.sqrt((X-prev_pos[0])**2 + (Z-prev_pos[2])**2) / (now - prev_time)
                prev_pos, prev_time = robot_pos, now
                trajectory.append(robot_pos)

                # Écriture instantanée dans le TSV
                if enregistrement_actif:
                    writer_tsv.writerow([stats["frame"], round(now, 3), round(X, 4), round(Y, 4), round(Z, 4)])
                    fichier_tsv.flush()

            if p1: cv2.drawMarker(frame1,(int(p1[0]),int(p1[1])),C_ROBOT,cv2.MARKER_CROSS,18,2)
            if p2: cv2.drawMarker(frame2,(int(p2[0]),int(p2[1])),C_ROBOT,cv2.MARKER_CROSS,18,2)

        out = build_frame(frame1, frame2, robot_pos)
        cv2.imshow("SYSMER Tracker", out)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('r') or key == ord('R'):
            enregistrement_actif = not enregistrement_actif

finally:
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    fichier_tsv.close()
    print("✅ Sauvegarde TSV propre terminée !")