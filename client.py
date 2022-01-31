import socket
from PIL import Image
import json
import base64
import os
from io import BytesIO
import cv2 
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HOST = '127.0.0.1'  
PORT = 8000        
BUFFER_SIZE = 1048576 #1MB

####################################################################################################

def LoadImageToBase64(imagePath):
    img = Image.open(imagePath)
    img = img.convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    imageBase64 = base64.b64encode(buffered.getvalue())
    return imageBase64.decode("utf8")

####################################################################################################

def Base64ToMat(base64Str):
    imageByte = base64.b64decode(base64Str)
    nparr = np.frombuffer(imageByte, np.uint8)
    mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return mat

####################################################################################################

s = None
try:
    imagePath = os.path.join(CURRENT_DIR, "example.jpg")
    data = {
        "imageBase64" : LoadImageToBase64(imagePath)
    }
    
    str_data = json.dumps(data)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (HOST, PORT)
    print('connecting to {0}:{1} '.format(HOST, PORT))
    s.connect(server_address)
    
    
    s.sendall(bytes(str_data, "utf8"))


    data = s.recv(BUFFER_SIZE)
    
    s.close()

    obj = json.loads(data)

    imageBase64 = obj["frameDraw"]
    frameDraw = Base64ToMat(imageBase64)
    boxes = obj["boxes"]

    print(boxes)

    #cv2.imshow("frame draw", frameDraw)
    cv2.imwrite("result.jpg", frameDraw)
    cv2.waitKey()

except Exception as e:
    print(str(e))
    if(s != None):
        s.close()