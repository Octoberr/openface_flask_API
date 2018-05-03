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
comparenet = openface.TorchNeuralNet(torchmodel, 96)
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

    # 计算两个点之间的直线距离
    def twodist(self, A, B):
        x = A[0] - B[0]
        y = A[1] - B[1]
        lenpoint = math.sqrt((x**2)+(y**2))
        return lenpoint

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
            height, width, chanl = rgbFrame.shape
            landmarks = align.findLandmarks(rgbFrame, bb)
            msg = {
                "face": 1,
                "landmarks": landmarks,
                "height": height,
                "width": width,
                "facemidleheight": (bb.bottom()+bb.top())/2
            }
        return msg

    # 使用本地图片测试程序,这段代码很牛B，解决了大部分问题
    # def processthephoto(self):
    #     imgdirectory = os.path.join(fileDir, 'imagetest')
    #     imgpath = os.path.join(imgdirectory, 'openmouse.jpg')
    #     # savepath = os.path.join(imgdirectory, 'openmousetag.jpg')
    #     bgrimg = cv2.imread(imgpath)
    #     rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)
    #     bb = align.getLargestFaceBoundingBox(rgbimg)
    #     landmarks = align.findLandmarks(rgbimg, bb)
    #     # print landmarks, len(landmarks)
    #     # for center in landmarks:
    #     #     cv2.circle(bgrimg, center=center, radius=3, color=(255, 0, 0), thickness=3)
    #     # cv2.imwrite(savepath, bgrimg)
    #     # msg = {"face": 1, "landmarks": landmarks}
    #     height, width, chanl = bgrimg.shape
    #     msg = {
    #         "face": 1,
    #         "landmarks": landmarks,
    #         "height": height,
    #         "width": width,
    #         "facemidleheight": (bb.bottom() + bb.top()) / 2
    #     }
    #     return msg

    def start(self, dataurl):
        msg = self.getthelandmark(dataurl)
        # 本地测试时使用下面代码
        # msg = self.processthephoto()
        if msg['face'] == 0:
            return json.dumps(msg)
        else:
            landmarks = msg['landmarks']
            # 左右偏斜
            lefteye = landmarks[36]
            righteye = landmarks[45]
            if lefteye[1] - righteye[1] == 0:
                msg['lineangle'] = 0
            else:
                # 水平偏斜，以左眼为角心， 角度的正负用高度的差来解决，用右眼减去左眼，因为坐标系为上面为0，所以结果为负表示右眼在上
                lineloc = self.angleBetweenVectorsDegrees(righteye, lefteye, [msg['width'], lefteye[1]])
                msg['lineangle'] = lineloc
                # 负数为右眼在上，正数为左眼在上
                msg['angleposttion'] = righteye[1] - lefteye[1]
            # 旋转偏斜
            nose = landmarks[30]
            leftnose = self.angleBetweenVectorsDegrees(lefteye, nose, [nose[0], 0])
            rightnose = self.angleBetweenVectorsDegrees(righteye, nose, [nose[0], 0])
            msg['leftnose'] = leftnose
            msg['rightnose'] = rightnose
            # 上下偏向
            facemidleheight = msg['facemidleheight']
            # 结果为正，提示低头；结果为负，提示抬头
            distance = facemidleheight - nose[1]
            msg['upanddown'] = distance
            # 眨眼系数的衡量
            lefteyehightdist = (self.twodist(landmarks[37], landmarks[41])+self.twodist(landmarks[38], landmarks[40]))/2
            msg['leftblink'] = lefteyehightdist/self.twodist(landmarks[36], landmarks[39])
            righteyedist = (self.twodist(landmarks[43], landmarks[47])+self.twodist(landmarks[44], landmarks[46]))/2
            msg['rightblink'] = righteyedist/self.twodist(landmarks[42], landmarks[45])
            # 判断是否张嘴的系数
            mousedist = (self.twodist(landmarks[50], landmarks[58])+self.twodist(landmarks[52], landmarks[56]))/2
            msg['mouse'] = mousedist/self.twodist(landmarks[48], landmarks[54])
            return json.dumps(msg)


class COMPARE:

    # def __init__(self, face, idcard):
    #     self.face = face
    #     self.idcard = idcard

    def getcardrep(self, dataurl):
        # 本地测试使用在线的imgpath，线上测试使用bs64的代码
        # bgrimg = cv2.imread(imgcardpath)
        head = "data:image/jpeg;base64,"
        assert (dataurl.startswith(head))
        imgdata = base64.b64decode(dataurl[len(head):])
        arryimage = Image.open(io.BytesIO(imgdata))
        rgb = cv2.cvtColor(np.array(arryimage), cv2.COLOR_BGR2RGB)
        bb = align.getLargestFaceBoundingBox(rgb)
        alignface = align.align(96, rgb, bb, landmarkIndices=landmarkIndices)
        rep = comparenet.forward(alignface)
        return rep

    # 应该是要支持多张人脸，目前先测试一张人脸吧
    def getfacerep(self, dataurl):
        head = "data:image/jpeg;base64,"
        assert (dataurl.startswith(head))
        imgdata = base64.b64decode(dataurl[len(head):])
        arryimage = Image.open(io.BytesIO(imgdata))
        rgb = cv2.cvtColor(np.array(arryimage), cv2.COLOR_BGR2RGB)
        # 测试为找到一张人脸，实际应用可能存在多张人脸
        bb = align.getLargestFaceBoundingBox(rgb)
        alignface = align.align(96, rgb, bb, landmarkIndices=landmarkIndices)
        rep = comparenet.forward(alignface)
        return rep

    def getcompareresult(self, idcardurl, faceurl):
        # imgdirectory = os.path.join(fileDir, 'imagetest')
        # faceimgpath = os.path.join(imgdirectory, 'openmouse.jpg')
        # idcardpath = os.path.join(imgdirectory, 'swmidcard.jpg')
        d = self.getcardrep(idcardurl) - self.getfacerep(faceurl)
        res = np.dot(d, d)
        return res

