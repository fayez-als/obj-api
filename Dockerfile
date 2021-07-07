FROM tensorflow/tensorflow
EXPOSE 5000

COPY . . 
RUN pip install -r requirements.txt
RUN apt install wget
RUN wget -O aaa.tar.gz "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1?tf-hub-format=compressed"
RUN mkdir resnet
RUN tar -xvzf aaa.tar.gz -C resnet/
RUN export FLASK_APP=app.py 


CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]

