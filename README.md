# Socket-ML
Example predict image via Socket connection, this example will detect face and return result to client.

This example contain a model to predict face, it small, simple and more accuracy than OpenCV Cascade.

![](socket_flow.JPG)

## Requirement software
- Python 3.7.3
- Visual Studio Code or PyCharm or any IDE can write Python


## How to run
- Install Python 3.7.3
- Install requirement package by command
```
pip install requirements.txt
```
- Run file **server.py** to create socket server
- Run file **client.py** to send image data to socket server
- Socket server predict and response result to client
- Client save image returned from server and show image

## Link
https://thigiacmaytinh.com/su-dung-socket-de-deploy-model-machine-learning
