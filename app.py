
import tensorflow as tf
import tensorflow_hub as hub
import base64
import os

from flask import Flask,jsonify,request 
from flask_restful import Api, Resource


import requests
import subprocess
import json

import numpy as np
from flask_cors import CORS, cross_origin
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps


def load_img(path):
    
    img = tf.io.read_file(path)
    
    
    img = tf.image.decode_jpeg(img, channels=3)
    
    return img
def run_detector(detector, path):
    
    img = load_img(path)
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)
    result = {key:value.numpy() for key,value in result.items()}
    return result


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    

    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    

    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)


    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):

    colors = list(ImageColor.colormap.values()) 

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        
        # only display detection boxes that have the minimum score or higher
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                         int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")

            # draw one bounding box and overlay the class labels onto the image
            draw_bounding_box_on_image(image_pil,
                                       ymin,
                                       xmin,
                                       ymax,
                                       xmax,
                                       color,
                                       font,
                                       display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
        
    return image


module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
model = hub.load(module_handle)
detector = model.signatures['default']

#img = load_img('guitar.jpg')
#converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
#result = detector(converted_img)
#image_with_boxes = draw_boxes(img.numpy(), result["detection_boxes"].numpy(),result["detection_class_entities"].numpy(), result["detection_scores"].numpy())
#im = Image.fromarray(image_with_boxes)
#im.save('detected.jpg')


app = Flask(__name__)
cors =CORS(app,resources={r"/upload":{"origins":"*"}})
app.config['CORS_HEADERS']= 'Content-Type'


@app.route('/')
def home():
    return 'working!'


@app.route('/upload',methods=['POST'])
@cross_origin(origin="*",headers=['Content-Type','Authorization'])
def upload_files():
    uploaded_file = request.files['image']
    if uploaded_file.filename !="":
        uploaded_file.save('guitar.jpg')
    
        img = load_img('guitar.jpg')
        os.remove("guitar.jpg")
        converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        result = detector(converted_img)
        image_with_boxes = draw_boxes(img.numpy(), result["detection_boxes"].numpy(),result["detection_class_entities"].numpy(), result["detection_scores"].numpy())
        im = Image.fromarray(image_with_boxes)
        im.save('detected.jpg')
        with open("detected.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        os.remove("detected.jpg")

        return encoded_string
    


    return jsonify({"result":"detected currectly"})


if __name__=='__main__':
    app.run(host='0.0.0.0')


