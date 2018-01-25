# coding:utf-8
"""
开启flask服务
create by swm 2018/1/25
"""
from flask import Flask, request, Response
import json
import gevent.monkey
from gevent.pywsgi import WSGIServer
gevent.monkey.patch_all()
# 内部引用
from judgetheloc import FACE
app = Flask(__name__)


@app.route('/face', methods=['post'])
def starttheserver():
    args = json.loads(request.data)
    dataurl = args['dataurl']
    face = FACE()
    msg = face.start(dataurl)
    return Response(msg, mimetype="application/json")


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 8000), app)
    try:
        print("Start at " + http_server.server_host +
              ':' + str(http_server.server_port))
        http_server.serve_forever()
    except(KeyboardInterrupt):
        print('Exit...')