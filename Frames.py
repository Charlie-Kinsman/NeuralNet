import cv2
class image():
    def __init__(self):
        print("SMILE FOR THE CAMERA")
        vc = cv2.VideoCapture(0)
        if vc.isOpened():
            rval, frame = vc.read()
        self.frame = frame
        vc.release()
