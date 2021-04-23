import cv2
import sys

#give location of haarcascade_frontalface_default.xml
faceCascade = cv2.CascadeClassifier('D:\python\lib\site-packages\cv2\data/haarcascade_frontalface_default.xml')

#give location of images
image = cv2.imread(r'D:\python\image.jpg')

#detection better in gray
imgGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 


#faces = faceCascade.detectMultiScale(imgGray,1.1,5) #for detections face 
faces = faceCascade.detectMultiScale(
    imgGray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
) 
for (x, y, w, h) in faces:  #ccheck all 4 factors in image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,255, 0), 2) # make rectangle
    cv2.putText(image, "cs balotiya", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)   #Add name on frame  
   
cv2.imshow("Face Detection Sucessful",image)

#-----------------------------------------------------
