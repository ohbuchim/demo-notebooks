import sys
import boto3
import json
from urllib.parse import urlparse
import os
import pandas as pd

args = sys.argv
if len(args) < 2:
    print('usage: python generate_manifestfile.py [s3path of data] [s3path of manifestfile (option)')

s3path = s3manifest_path = args[1]
if len(args) == 3:
    s3manifest_path = arge[2]

def prepare(s3_image_path, s3_manifest_path):

    image_url = urlparse(s3_image_path)
    output_url = urlparse(s3_manifest_path)

    s3 = boto3.client("s3")

    image_response = s3.list_objects(Bucket=image_url.netloc, Prefix=image_url.path[1:])

    image_list = parse_response(image_response)

    content_list = []

    for item in image_list:
        image_filename = item.split('/')[-1]
        entry = {}
        ext = os.path.splitext(item)[1][1:]
        if ext == 'jpg' or ext == 'jpeg' or ext == 'png':
            entry['source-ref'] = "s3://{}/{}".format(image_url.netloc,item)
            print(entry)
            content_list.append(entry)
            
    json_content = json.dumps(content_list, ensure_ascii=False)
    df = pd.read_json(json_content)
    
    content_list = df.to_json(orient='records', lines=True, force_ascii=False)


    body = bytes(content_list,'utf-8')

    resp = s3.put_object(Bucket=output_url.netloc, Key="{}/manifest.json".format(output_url.path[1:]), Body=body)


def parse_response(response):
    list=[]
    prefix = ''
    for content in response['Contents']:
        if (content['Size'] > 0):
            file_name = content['Key']
            list.append(file_name)

    return list

prepare(s3path, s3manifest_path)
