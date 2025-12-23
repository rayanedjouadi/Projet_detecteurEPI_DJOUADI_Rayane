from ultralytics import YOLO
import cv2

# On utilise le modèle "Nano", le plus léger, idéal pour l'embarqué
model = YOLO('yolov8n.pt') 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # L'IA analyse l'image en temps réel
        results = model(frame)
        annotated_frame = results[0].plot()
        
        cv2.imshow("Test IA YOLOv8", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): # Appuie sur 'q' pour quitter
            break
cap.release()
cv2.destroyAllWindows()
