from ultralytics import YOLO
import cv2

# Charge le modèle fraîchement entraîné
# Remplacez le chemin si vous n'avez pas déplacé le fichier
model = YOLO('best.pt') 

# Utilise l'index de caméra qui fonctionnait lors de tes tests
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # L'IA analyse avec le modèle "best.pt"
        results = model(frame, conf=0.5) # On n'affiche que si l'IA est sûre à 50%
        annotated_frame = results[0].plot()
        
        cv2.imshow("IA Sécurité Industrielle - PPE Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
