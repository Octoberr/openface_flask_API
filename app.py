# coding:utf-8
"""
开启flask服务
create by swm 2018/1/25
增加人像对比
changed 2018/05/03
"""
from flask import Flask, request, Response
import json
import gevent.monkey
from gevent.pywsgi import WSGIServer
gevent.monkey.patch_all()
# 内部引用
from judgetheloc import FACE, COMPARE
app = Flask(__name__)


@app.route('/face', methods=['post'])
def starttheserver():
    args = json.loads(request.data)
    dataurl = args['dataurl']
    face = FACE()
    msg = face.start(dataurl)
    return Response(msg, mimetype="application/json")


# 本地测试通过,用于测试本地的flask是否开启
@app.route('/test', methods=['get'])
def testapi():
    info = {"get": "yes, i got it"}
    return Response(json.dumps(info), mimetype="application/json")


# 人脸和身份证的对比
@app.route('/compare', methods=['post'])
def comparewithidcard():
    args = json.loads(request.data)
    faceurl = args['faceurl']
    cardurl = args['cardurl']
    compare = COMPARE()
    result = compare.getcompareresult(cardurl, faceurl)
    info = {"result": result}
    return Response(info, mimetype="application/json")


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 8000), app)
    try:
        print("Start at " + http_server.server_host +
              ':' + str(http_server.server_port))
        http_server.serve_forever()
    except(KeyboardInterrupt):
        print('Exit...')