import cv2

video = cv2.VideoCapture(0)
cat = cv2.imread('Projects/images.jpeg', -1)
success, frame = video.read()
height = frame.shape[0]
width = frame.shape[1]

face_cascade = cv2.CascadeClassifier('videoProcessing/facesVideo.xml')

while success:
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    for(x, y, w , h ) in faces:
        rscat = cv2.resize(cat, (w, h), interpolation = cv2.INTER_AREA)# resize the cat face to match the detected face
        place = frame[y:y+h, x:x+w]
        cv2.imwrite('rscat.jpg', place)
        blend = cv2.addWeighted(place, 0, rscat, 1,0)
        cv2.imwrite('fcat.jpg', blend)
        frame[y:y+h, x:x+w] = blend
        cv2.rectangle(frame, (x, y), (x+h , y+w), (255,255,255), 4)
    cv2.imshow("Show Your Face", frame)
    key = cv2.waitKey(1)
    if key == ord('q') :
        break
    success, frame = video.read()

video.release()
cv2.destroyAllWindows()
