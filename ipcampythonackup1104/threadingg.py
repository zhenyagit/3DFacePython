from threading import Thread
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import math


(major, minor) = cv2.__version__.split(".")[:2]

def takeframe(capture,frame):
    while(1):
        req, frame = vs.read()

class GetTitleThread(Thread):        

    def __init__(self, capture):
        self.sTitle = None
        self.capture = capture
        super(GetTitleThread, self).__init__()

    def run(self):
        req, frame = self.capture.read()
        self.sTitle = frame

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
ap.add_argument("-v", "--vid", type=str, default="0", help="VideoStream input")
args = vars(ap.parse_args())

if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())
    print("ok")

else:
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

vs3 = cv2.VideoCapture("http://10.42.0.219:8080/video")
vs = cv2.VideoCapture(0)
vs3.set(cv2.CAP_PROP_BUFFERSIZE,3)
vs.set(cv2.CAP_PROP_BUFFERSIZE,3)
vs2 = cv2.VideoCapture("http://10.42.0.55:8080/video")
vs2.set(cv2.CAP_PROP_BUFFERSIZE,3)
time.sleep(1.0)

r = 1;
l = math.sqrt(2)

def drawkrestik(img,x,y,razmer = 10):

    cv2.line(img,(x-razmer,y),(x+razmer,y),(0,0,255))
    cv2.line(img,(x,y-razmer),(x,y+razmer),(0,0,255))

def angleged(k,l,r):
    return (180*math.asin(k/math.sqrt(r*r-math.pow((l/2),2)+k*k))/math.pi)

xobj = 0
yobj = 0
angle = 0
initBB = None

twrv = GetTitleThread(vs)
twrv2 = GetTitleThread(vs2)
twrv3 = GetTitleThread(vs3)
twrv.start()
twrv2.start()
twrv3.start()

while (vs.isOpened()):
    twrv.run()
    twrv2.run()
    twrv3.run()
    twrv.join()
    twrv3.join()
    twrv2.join()
    frame3 = twrv3.sTitle
    frame2 = twrv.sTitle
    # cv2.imshow("Frame2", frame )
    frame= twrv2.sTitle
    # cv2.imshow("Frame3", frame )
    (H, W) = frame.shape[:2]
    
    if initBB is not None:
        (success, box) = tracker.update(frame)
 
        if success:
            (x, y, w, h) = [int(v) for v in box]
            xobj = int(x+w/2)
            yobj = int(y+h/2)
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
        angle = angleged((xobj-(W/2))*(math.sqrt(2)/2)/250,l,r)
        info = [
            ("Success", "Yes" if success else "No"),
            ("x = ", xobj),
            ("y = ", yobj),
            ("angle = ", angle),
            ("xcen = ", xobj-(W/2)),
        ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    drawkrestik(frame,W/2,H/2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Frame2", frame2)
    cv2.imshow("Frame3", frame3)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("s"):
        if int(major) == 3 and int(minor) < 3:
            tracker = cv2.Tracker_create(args["tracker"].upper())
            print("ok")
        else:
                OPENCV_OBJECT_TRACKERS = {
                "csrt": cv2.TrackerCSRT_create,
                "kcf": cv2.TrackerKCF_create,
                "boosting": cv2.TrackerBoosting_create,
                "mil": cv2.TrackerMIL_create,
                "tld": cv2.TrackerTLD_create,
                "medianflow": cv2.TrackerMedianFlow_create,
                "mosse": cv2.TrackerMOSSE_create
                }
 
                tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)
 
        tracker.init(frame, initBB)
    elif key == ord("q"):
        break
cv2.destroyAllWindows()