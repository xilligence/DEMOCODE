from flask import Flask
from flask import request, jsonify
import ColorRectification
import TextRekognition
import ModelTraining
from PIL import Image
import os
import base64
import json
from io import BytesIO
from keras import backend as K

app = Flask(__name__)

#------------------------------------------------------
# using for color prediction

@app.route('/getImageColor', methods=['POST'])
def getImageColor():
    imageString=request.args.get('imageString',None)
    data = request.data
    dataDict = json.loads(data)
    for x in dataDict:
        if(x=='data'):
            imageString=dataDict[x]
    im = Image.open(BytesIO(base64.b64decode(imageString)))
    objColor,colorAccuracy=ColorRectification.process_image_getColor(im)
    return jsonify({'ResultData': objColor, 'Accuracy': colorAccuracy})

# Find Object,Bumper, Bumper Text ,Sedan or Suv
# AWS API
@app.route('/objectClassification', methods=['POST'])
def objectClassification():
    data = request.data
    dataDict = json.loads(data)
    for x in dataDict:
        if (x == "aws_access_key_id"):
            aws_access_key_id=dataDict[x]
        if (x == 'aws_secret_access_key'):
            aws_secret_access_key=dataDict[x]
        if (x == 'fileName'):
            fileName=dataDict[x]
        if (x == 'bucket'):
            bucket = dataDict[x]
        if (x == 'imgpath'):
            imgpath = dataDict[x]
    print(bucket)
    print(imgpath)
    responseObject,responseText = TextRekognition.getTextAndObjectRekognition(fileName, aws_access_key_id, aws_secret_access_key, bucket,imgpath)
    return jsonify(responseObject,responseText)

# Model Training and prediction classification
# returns model, brand, views etc
@app.route('/getModelTraining', methods=['POST'])
def getModelTraining():
    data = request.data
    dataDict = json.loads(data)
    for x in dataDict:
        if(x=='data'):
            imageString=dataDict[x]
    im = Image.open(BytesIO(base64.b64decode(imageString)))
    folder = './data/car/test/car/files/'

    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))

    im.save(folder + "img.png")
    mo = ModelTraining.modelTraing()
    carBrand, carAccuracy = ModelTraining.accuracyTest(mo)
    c,d=ModelTraining.CarViewChecking(carBrand)
    carView,carViewAccuracy=ModelTraining.checkTestModel(c,d)
    K.clear_session()
    return jsonify({'ResultData': carBrand, 'Accuracy': carAccuracy,'carView': carView, 'carViewAccuracy': carViewAccuracy})

