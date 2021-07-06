FROM tensorflow/tensorflow
EXPOSE 5000

COPY . . 
RUN pip install -r requirements.txt
RUN export FLASK_APP=app.py 


CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]

