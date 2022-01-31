from datetime import datetime
import socket
from unittest import result
import cv2
import os
import json
import numpy as np
import base64

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

faceProto = os.path.join(CURRENT_DIR, "facebox.pbtxt")
faceModel = os.path.join(CURRENT_DIR, "facebox.pb")

####################################################################################################

class FaceBoxDetector():
    #device is cpu or gpu
    def __init__(self, device=""):
        if(device == None or device == ""):
            numGPU = cv2.cuda.getCudaEnabledDeviceCount()
            if(numGPU > 0):
                self.device = "gpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.LoadModel()

    ####################################################################################################

    def LoadModel(self):
        # Load network
        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
        if self.device == "cpu":
            self.faceNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        elif self.device == "gpu":
            self.faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ####################################################################################################

    def getFaceBoxInPath(self, imgPath, conf_threshold=0.7, drawResult=False):        
        mat = cv2.imread(imgPath)
        return self.getFaceBoxInMat(mat, conf_threshold, drawResult)

    ####################################################################################################

    def getFaceBoxInMat(self, frame, conf_threshold=0.7, drawResult=False):
        frameDraw = frame.copy()
        frameHeight = frameDraw.shape[0]
        frameWidth = frameDraw.shape[1]
        blob = cv2.dnn.blobFromImage(frameDraw, 1.0, (300, 300), [104, 117, 123], True, False)

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)

                if(x1 > frameWidth or x2 > frameWidth or y1 > frameHeight or y2 > frameHeight):
                    continue

                bboxes.append([x1, y1, x2, y2])

                if(drawResult):
                    cv2.rectangle(frameDraw, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)

        print("Detected {0} faces".format(len(bboxes)))
        return frameDraw, bboxes

####################################################################################################

def Base64ToMat(base64Str):
    imageByte = base64.b64decode(base64Str)
    nparr = np.frombuffer(imageByte, np.uint8)
    mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return mat

####################################################################################################

def MatToBase64(mat):
    retval, buffer = cv2.imencode('.jpg', mat)
    strBase64 = base64.b64encode(buffer)
    return strBase64.decode("utf8")

####################################################################################################

faceboxDetector = FaceBoxDetector("cpu")
print("Socket server load model complete")

HOST = '127.0.0.1'  
PORT = 8000        
SERVER = (HOST, PORT)
MAX_CONNECTION = 255
BUFFER_SIZE = 1048576 #1MB

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(SERVER)
s.listen(MAX_CONNECTION)

print("Socket server waiting for connection...")

while True:
    client, addr = s.accept()
    
    try:
        print('Connected by', addr)
        data = client.recv(BUFFER_SIZE)
        str_data = data.decode("utf8")
        if str_data == "quit":
            break

        data = json.loads(str_data)
        frame = Base64ToMat(data["imageBase64"])
        frameDraw, bboxes = faceboxDetector.getFaceBoxInMat(frame, drawResult=True)

        result ={
            "frameDraw" : MatToBase64(frameDraw),
            "boxes" : str(bboxes)
        }


        str_result = json.dumps(result)
        client.send(str_result.encode("utf8"))
        client.close()
    except Exception as e:
        print(str(e))
        client.close()

s.close()