import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

# Charger les visages connus et leurs noms
known_face_encodings = []
known_face_names = []

# Remplacer 'known_faces' par le dossier contenant les images des visages connus
for image_name in os.listdir('known_faces'):
    image = face_recognition.load_image_file(f'known_faces/{image_name}')
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(image_name)[0])

# Initialisation des variables
face_locations = []
face_encodings = []
face_names = []

# Démarrer la webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        # Enregistrer la présence dans un fichier CSV
        with open('attendance.csv', 'a') as f:
            writer = csv.writer(f)
            now = datetime.now()
            writer.writerow([name, now.strftime('%Y-%m-%d %H:%M:%S')])

    # Afficher les résultats
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
