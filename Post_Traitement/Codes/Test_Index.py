import cv2
import time

print("\n" + "="*50)
print("🔍 RECHERCHE DES CAMÉRAS CONNECTÉES...")
print("="*50 + "\n")

ports_trouves = []

# On teste les index de 0 à 9 (largement suffisant pour un PC)
for index in range(10):
    cap = cv2.VideoCapture(index)
    
    # Si la caméra s'ouvre, on tente de lire une image
    if cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            print(f"✅ Caméra détectée à l'index : {index}")
            ports_trouves.append(index)
            
            # Ajout du texte pour identifier visuellement la caméra
            cv2.putText(frame, f"INDEX CAMERA : {index}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Fermeture auto dans 2s...", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            # On affiche l'image
            cv2.imshow("Test Camera", frame)
            
            # On attend 2000 millisecondes (2 secondes)
            cv2.waitKey(5000)
            
        cap.release()

cv2.destroyAllWindows()

print("\n" + "="*50)
if not ports_trouves:
    print("❌ Bilan : AUCUNE caméra trouvée. Vérifie tes branchements USB.")
else:
    print(f"🎯 Bilan : Les index utilisables sont : {ports_trouves}")
print("="*50 + "\n")