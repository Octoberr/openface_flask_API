FROM bamos/openface
COPY . /root/openface/face
WORKDIR /root/openface/face
RUN pip install --no-cache-dir --disable-pip-version-check -i https://mirrors.ustc.edu.cn/pypi/web/simple -r requirements.txt

CMD ["python", "app.py"]