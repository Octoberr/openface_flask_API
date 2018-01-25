# coding:utf-8
import os
import cv2
import openface
import numpy as np
import math
import base64
from PIL import Image
import io
import json

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
predict = os.path.join(dlibModelDir, 'shape_predictor_68_face_landmarks.dat')
torchmodel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
align = openface.AlignDlib(predict)
net = openface.TorchNeuralNet(torchmodel)
landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE


class FACE:

    def angleBetweenVectorsDegrees(self, A, vertex, C):
        """Return the angle between two vectors in any dimension space,
        in degrees."""
        # Convert the points to numpy latitude/longitude radians space
        a = np.radians(np.array(A))
        vertexR = np.radians(np.array(vertex))
        c = np.radians(np.array(C))
        # Vectors in latitude/longitude space
        sideA = a - vertexR
        sideC = c - vertexR
        # Adjust vectors for changed longitude scale at given latitude into 2D space
        lat = vertexR[0]
        sideA[1] *= math.cos(lat)
        sideC[1] *= math.cos(lat)
        direct = np.degrees(math.acos(np.dot(sideA, sideC) / (np.linalg.norm(sideA) * np.linalg.norm(sideC))))
        return direct

    def getthelandmark(self, dataurl):
        head = "data:image/jpeg;base64,"
        assert (dataurl.startswith(head))
        imgdata = base64.b64decode(dataurl[len(head):])
        arryimage = Image.open(io.BytesIO(imgdata))
        rgbFrame = cv2.cvtColor(np.array(arryimage), cv2.COLOR_BGR2RGB)
        bb = align.getLargestFaceBoundingBox(rgbFrame)
        if bb is None:
            msg = {
                "face": 0
            }
        else:
            height, width, chanl = np.array(arryimage).shape
            landmarks = align.findLandmarks(rgbFrame, bb)
            msg = {
                "face": 1,
                "landmarks": landmarks,
                "height": height,
                "width": width,
                "facemidleheight": (bb.bottom()+bb.top())/2
            }
        return msg

    def start(self, dataurl):
        msg = self.getthelandmark(dataurl)
        if msg['face'] == 0:
            return json.dumps(msg)
        else:
            landmarks = msg['landmarks']
            # 左右偏斜
            lefteye = landmarks[36]
            righteye = landmarks[45]
            if lefteye - righteye == 0:
                msg['lineangle'] = 0
            else:
                lineloc = self.angleBetweenVectorsDegrees(righteye, lefteye, [msg['width'], lefteye[1]])
                msg['lineangle'] = lineloc
            # 旋转偏斜
            nose = landmarks[33]
            leftnose = self.angleBetweenVectorsDegrees(lefteye, nose, [nose[0], 0])
            rightnose = self.angleBetweenVectorsDegrees(righteye, nose, [nose[0], 0])
            msg['leftnose'] = leftnose
            msg[rightnose] = rightnose
            # 上下偏向
            facemidleheight = msg['facemidleheight']
            # 结果为正，提示低头；结果为负，提示抬头
            distance = facemidleheight - nose[1]
            msg['upanddown'] = distance
            return json.dumps(msg)



