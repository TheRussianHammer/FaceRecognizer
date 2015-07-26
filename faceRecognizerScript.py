from faceComparator import *
import thread

f = faceComparator()
threshold = .6
faceThread = thread
keyThread = thread
def runFaceComparator():
    while True:
        faceFound = f.faceDetection()
        if(faceFound):
            faceConfidence = f.findFace()
def checkKeyPress():
    while True:
        if cv2.waitKey(1) == ord('q'):
            faceThread.exit()
            cv2.destroyAllWindows() 
            f.addFace()
            f.buildRecognizer()
    
def main():
    try:
        faceThread.start_new_thread(runFaceComparator)
        keyThread.start_new_thread(checkKeyPress)        
    except:
        print "An error occured, please run again"

main()
        
