import face_recognition
import os
import cv2
import imutils

known_faces_dir = "Known Faces"
unknown_faces_dir = "Unknown Faces"
tolerance =  0.6 #lower the tolerance, less chance of false positives
frame_thickness = 3 #pixel values
font_thickness = 2
model = "cnn" #hog

# step 1: load all known faces 
print("Loading Known Faces")

known_faces = []
known_names =[]

for name in os.listdir(known_faces_dir):
    for filename in os.listdir(os.path.join(known_faces_dir, name)):
        image = face_recognition.load_image_file(os.path.join(known_faces_dir, name, filename))
        image = imutils.resize(image, height=1024)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print("Processing Unknown Faces")
for filename in os.listdir(unknown_faces_dir):
    print(filename)
    image = face_recognition.load_image_file(os.path.join(unknown_faces_dir, filename))
    image = imutils.resize(image, height=1024)
    locations = face_recognition.face_locations(image, model = model)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [0, 255, 0]

            cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)

            top_left = (face_location[3], face_location[2]) 
            bottom_right = (face_location[1], face_location[2]+22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_thickness)
    
    cv2.imshow(filename, image)
    cv2.waitKey(1000)
    # cv2.destroyWindow(filename)