import cv2
import time
import face_recognition
import numpy as np
import os

# Prepare face recognition data
known_face_encodings = []
known_face_names = []

# Folder path containing the known face images
known_faces_folder = 'faces'

# Iterate over the files in the folder
for file_name in os.listdir(known_faces_folder):
    file_path = os.path.join(known_faces_folder, file_name)
    if os.path.isfile(file_path):
        image = face_recognition.load_image_file(file_path)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]  # Assuming one face per image
            known_face_encodings.append(face_encoding)
            name = file_name.split('.')[0]  # Extract the name from the file name
            known_face_names.append(name)

trainedDataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

start_time = time.time()
frame_count = 0
while True:
    success, frame = video.read()
    if success:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = trainedDataset.detectMultiScale(gray_image)

        for (x, y, w, h) in faces:
            face_image = frame[y:y + h, x:x + w]
            face_encodings = face_recognition.face_encodings(face_image)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    closest_match_index = np.argmin(face_distances)
                    if face_distances[closest_match_index] < 0.6:  # Adjust the threshold as per your requirement
                        recognized_name = known_face_names[closest_match_index]
                    else:
                        recognized_name = "Unknown"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Video completed or frame nil")
        break

# Release the video capture and close the OpenCV windows
video.release()
cv2.destroyAllWindows()
