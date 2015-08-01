import os as os
import numpy as np
import cv2
import pyttsx

class faceComparator():
    """Reference to
    http://hanzratech.in/2015/02/03/face-recognition-using-opencv.html"""
     
    def __init__(self):        
        self.X = []
        self.y = []
        self.path = "C:/Python27/Faces"
        self.recognizerPath = "C:/Python27/LBPHFaceRecognizers/recognizer.re"
        self.speechEngine = pyttsx.init()
        try:
            self.recognizer = cv2.createLBPHFaceRecognizer()
            self.recognizer.load(self.recognizerPath)
        except:
            self.recognizer = cv2.createLBPHFaceRecognizer()
    def faceDetection(self):
        cap = cv2.VideoCapture(0)

        faceCascade = cv2.CascadeClassifier("c:\Python27\haarcascade_frontalface_alt.xml")
        eyeCascade = cv2.CascadeClassifier("c:\Python27\haarcascade_eye.xml")
	Run = True

        while(Run):
            ret, frame = cap.read()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 4, 0,(20,20))
            if len(faces) != 0:
                faces[:, 2:] += faces[:, :2]
           

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                print "face found!"
                Run = False
                cut = frame[y:h, x:w]
                cut = cv2.resize(cut,(256,256))
                path = "c:\Python27\Faces\CurrentFace.jpg"
                cv2.imwrite(path,cut)
                return True
           
    	
          #  cv2.imshow('test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def buildRecognizer(self):
        imageNames = os.listdir(self.path)
        imageNames.remove("CurrentFace.jpg")
        images = []
        
        for name in imageNames:
            images.append(cv2.imread(self.path + "/" + name, cv2.IMREAD_GRAYSCALE))
          
        labels = list(range(len(images)))
        recognizer = cv2.createLBPHFaceRecognizer()
        recognizer.train(images, np.array(labels))
        recognizer.save(self.recognizerPath)
        self.recognizer = recognizer

    def normalize(self,X, low, high, dtype=None):
         
        """Normalizes a given array in X to a value between low and high."""
        X = np.asarray(X)
        minX, maxX = np.min(X), np.max(X)
        # normalize to [0...1].
        X = X - float(minX)
        X = X / float((maxX - minX))
        # scale to [low...high].
        X = X * (high-low)
        X = X + low
        if dtype is None:
            return np.asarray(X)
        return np.asarray(X, dtype=dtype)
    
    def addFace(self):
        self.faceDetection()
        image = cv2.imread(self.path + "/" + "CurrentFace.jpg")
        name = raw_input("Enter your name: ")
        cv2.imwrite(self.path + "/" + name + ".jpg",image)
            
    def findFace(self):
        imageNames = os.listdir(self.path)
        imageNames.remove("CurrentFace.jpg")

        currentImage = cv2.imread(self.path + "/" + "CurrentFace.jpg", cv2.IMREAD_GRAYSCALE)
        predict_image = np.array(currentImage, 'uint8')
        nbr_predicted, conf = self.recognizer.predict(predict_image)
        print conf
        name = imageNames[nbr_predicted]
        name = name[:-4]
        self.speechEngine.say(name + " identified")
        self.speechEngine.runAndWait()
        print(name + " identified")

        return conf
    
