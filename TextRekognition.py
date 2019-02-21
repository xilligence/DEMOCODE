import boto3

def getTextAndObjectRekognition(s3FileName,aws_access_key_id,aws_secret_access_key,bucket,imgpath):
    textRekognition=getTextRekognition(s3FileName,aws_access_key_id,aws_secret_access_key,bucket,imgpath)
    objectRekognition=getObjectRekognition(s3FileName,aws_access_key_id,aws_secret_access_key,bucket,imgpath)
    return objectRekognition,textRekognition

def getTextRekognition(s3FileName,aws_access_key_id,aws_secret_access_key,bucket,imgpath):
    bucket=bucket
    photo=imgpath+'/'+s3FileName
    client = boto3.client('rekognition', 'ap-south-1',aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key)
    response = client.detect_text(Image={'S3Object': {'Bucket': bucket, 'Name': photo}})
    return response;


def getObjectRekognition(s3FileName,aws_access_key_id,aws_secret_access_key,bucket,imgpath):
    #bucket = 'xilli-ai'
    #photo = 'ai/' + s3FileName
    bucket = bucket
    photo = imgpath + '/' + s3FileName
    client = boto3.client('rekognition', 'ap-south-1',aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key)
    response = client.detect_labels(Image={'S3Object': {'Bucket': bucket, 'Name': photo}}, MaxLabels=10)
    return response;





