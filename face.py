import cv2
import numpy as np

# This line loads the haarcascade_frontalface_default.xml file
# which contains the pre-trained model for face detection
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_extractor(img):
  """Extracts the face from an image.

  Args:
    img: A numpy array representing the image.

  Returns:
    A numpy array representing the extracted face, or None if no face is found.
  """
  # Convert the image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # Detect faces in the grayscale image
  faces = face_classifier.detectMultiScale(gray, 1.3, 5)
  
  # If no faces are found, return None
  if faces is ():
    return None

  # Extract the first detected face
  for (x, y, w, h) in faces:
    cropped_face = img[y:y+h, x:x+w]
    return cropped_face

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Initialize a counter to keep track of the number of faces collected
count = 0

while True:
  # Capture a frame from the video stream
  ret, frame = cap.read()

  # Extract the face from the frame
  face = face_extractor(frame)

  # If a face is found
  if face is not None:
    # Increment the counter
    count += 1
    
    # Resize the face to a fixed size of 500x500 pixels
    face = cv2.resize(face, (500, 500))
    
    # Convert the face to grayscale
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Generate a filename for the face image
    file_name_path = 'faces/user' + str(count) + ".jpg"
    
    # Save the face image to disk
    cv2.imwrite(file_name_path, face)
    
    # Add a count label to the face image
    cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    # Display the face image
    cv2.imshow("face_Cropper", face)
  else:
    # Print a message if no face is found
    print("Face not Found")
  
  # Exit the loop if 'q' key is pressed or 100 faces are collected
  if cv2.waitKey(1) == 13 or count == 100:
    break

# Release the video capture object
cap.release()

# Close all open windows
cv2.destroyAllWindows()

# Print a message to indicate that the face collection is completed
print("Collecting Samples Completed!!!")
