# 👷‍♂️ Détection d'EPI en Temps Réel - Vision par Ordinateur (YOLOv8)

Ce projet consiste en un système de détection automatique des Équipements de Protection Individuelle (EPI) par intelligence artificielle. Il a été conçu pour renforcer la sécurité sur les sites industriels en vérifiant en temps réel le port du casque, du gilet et d'autres équipements de sécurité.

## 🚀 Aperçu des Performances

Le modèle a été entraîné avec **YOLOv8** sur un dataset de plus de **5 000 images**. Après un entraînement optimisé sur **10 époques**, le modèle a atteint des scores de précision très élevés, le rendant viable pour une utilisation en conditions réelles.

### Résultats Clés (mAP50) :

* **Casques (Helmet) :** **96.4%**
* **Gilets de sécurité (Vest) :** **95.8%**
* **Chaussures/Bottes (Boots) :** **92.8%**

### Courbes d'Apprentissage :

Voici les graphiques de performance générés lors de l'entraînement, montrant une convergence stable et une progression constante de la précision :

## 🛠️ Installation et Configuration

### 1. Prérequis

Assurez-vous d'avoir Python installé ainsi que les bibliothèques suivantes :

```bash
pip install ultralytics opencv-python roboflow

```

### 2. Utilisation

Pour lancer la détection en direct via votre webcam :

```bash
python final_demo.py

```

## 📂 Structure du Répertoire

* **`best.pt`** : Le modèle final entraîné contenant les poids optimisés.
* **`final_demo.py`** : Script Python pour la détection en temps réel via webcam.
* **`train_epi.py`** : Script d'entraînement utilisé pour spécialiser le modèle sur les EPI.
* **`results.jpg`** : Graphiques d'analyse des performances (Loss, Precision, mAP).
* **`.gitignore`** : Configuration pour exclure les fichiers temporaires et les dossiers lourds (runs, cache).

## 📈 Détails Techniques

* **Algorithme :** YOLOv8 (You Only Look Once).
* **Vitesse d'inférence :** ~47.5 ms par image sur CPU (permettant un flux vidéo fluide).
* **Dataset :** Source Roboflow (PPE-detection-qlq3d).

