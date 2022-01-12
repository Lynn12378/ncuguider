#我的程式庫2
import re, sqlite3, time
from flask import Flask, render_template, url_for, request,redirect
app = Flask(__name__)
from flask import make_response

#匯入資料庫
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time

'''
Authenticate
Authenticates your credentials and creates a client.
'''
# subscription_key = "PASTE_YOUR_COMPUTER_VISION_SUBSCRIPTION_KEY_HERE"
# endpoint = "PASTE_YOUR_COMPUTER_VISION_ENDPOINT_HERE"
subscription_key = "5282a9e4d28e4bc9be4c815e1e4c0875"
endpoint = "https://minnie.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
#匯入資料庫結束



#先放所有應放函式
# import flask related
from flask import Flask, request, abort
from urllib.parse import parse_qsl
# import linebot related
from linebot import (
    LineBotApi, WebhookHandler
)
import sqlite3
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    LocationSendMessage, ImageSendMessage, StickerSendMessage,
    VideoSendMessage, TemplateSendMessage, ButtonsTemplate, PostbackAction, MessageAction, URIAction,
    PostbackEvent, ConfirmTemplate, CarouselTemplate, CarouselColumn,
    ImageCarouselTemplate, ImageCarouselColumn, FlexSendMessage
)
import json
  #以下為ai之匯入物
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Load the model
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  #結束啦





# create flask server
app = Flask(__name__)
line_bot_api = LineBotApi('XP20XRH5MoUM+ThB+BDaw92NKxBLY6/hIvWJ7Z6oorc7RI+hT5uIHm0s8K6oa2pmGryEcLm9giuVjIYSdBvICvWSgwXI+YgUdUxoXeQmtQUGmHBOjUz4Jyp8mzsL5dI1qcPZDb6Z5ywjmkmiDS/DnAdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('1ba208cb5a1928d2a9c3790bb7a2c3dc')
#結束我的程式庫

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/carousel')
def carousel():
    return render_template('carousel.html')

@app.route('/picture/1')
def picture1():
    return render_template('PictureP1.html')
@app.route('/picture/2')
def picture2():
    return render_template('PictureP2.html')
@app.route('/picture/3')
def picture3():
    return render_template('PictureP3.html')
@app.route('/picture/4')
def picture4():
    return render_template('PictureP4.html')
    
@app.route('/function')
def function():
    return render_template('function.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def page_not_found(e):
    return render_template('500.html'), 500
#我放的東東

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        print('receive msg')
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'#創建server

@handler.add(MessageEvent) #message=TextMessage)
def handle_message(event):
    # get user info & message
    user_id = event.source.user_id
    user_name = line_bot_api.get_profile(user_id).display_name

    if event.message.type=='text':
        receive_text=event.message.text
        print(receive_text)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text = '請傳送圖片'))
        
        
    
    elif event.message.type=='image':
        message_content = line_bot_api.get_message_content(event.message.id)
        with open('static\photo.png', 'wb' ) as fd:
            for chunk in message_content.iter_content():
                 fd.write(chunk)
    #開始跑是不是花了
        #remote_image_url = "http://bbs.mychat.to/attach/Fid_234/234_528201.jpg"
        '''
        Tag an Image - remote
        This example returns a tag (key word) for each thing in the image.
        '''
        print("===== Tag an image - remote =====")
        # Call API with remote image
        #tags_result_remote = computervision_client.tag_image(remote_image_url )
        local_image = open('static\photo.png', "rb")
        # Select visual feature type(s)
        local_image_features = ["categories"]
        # Call API
        categorize_results_local = computervision_client.analyze_image_in_stream(local_image, local_image_features)


        # Print results with confidence score
        print("Tags in the remote image: ")
        if (len(categorize_results_local.categories) == 0):
            print("No tags detected.")
        else:
            for category in categorize_results_local.categories:
                print("'{}' with confidence {:.2f}%".format(category.name, category.score* 100))
            void=0
            for category in categorize_results_local.categories:
                if category.name=='plant_flower':
                        void=1
            if(void==0):
                print("請輸入花的圖片")
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text = '請傳送花的圖片'))
                #開始跑是不是花結束
                #該開始run ai模型了
            else:
    
                #開始跑是不是花結束
                #該開始run ai模型了
                image = Image.open('static\photo.png')
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)

                #turn the image into a numpy array
                image_array = np.asarray(image)
                # Normalize the image
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                 # Load the image into the array
                data[0] = normalized_image_array

                # run the inference
                prediction = model.predict(data)
                #測試測試
                print(prediction.max())
                print(prediction.argmax())
                print(prediction)
                    #測試測試
                #檢驗資訊中
                if prediction.max()< 0.59:
                    print("此花為未知物種，將會盡快將之放入資料庫以提供客戶更好的服務")
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text = '此花為未知物種，將會盡快將之放入資料庫以提供客戶更好的服務'))
                else:
                #檢驗資訊完成
            #終於跑完ai模型了!!!我快要哭死了!!!!
            #我希望能研究出方法把line bot 連接到資料庫
                    number=prediction.argmax()

                #進入資料庫的環節
                    con = sqlite3.connect('homework.db')
                    cursorObj = con.cursor()
                    cursorObj.execute(f'SELECT name FROM test1 WHERE `編號`={number}')
                    name=cursorObj.fetchone()[0]
                    print(name)
                    cursorObj.execute(f'SELECT 經度 FROM test1 WHERE `編號`={number}')
                    logitude=cursorObj.fetchone()[0]
                    print(logitude)
                    cursorObj.execute(f'SELECT 緯度 FROM test1 WHERE `編號`={number}')
                    altitude=cursorObj.fetchone()[0]
                    print(altitude)
                    cursorObj.execute(f'SELECT 資料內容 FROM test1 WHERE `編號`={number}')
                    text=cursorObj.fetchone()[0]
                    print(text)
                    con.close 
    
    
    results=[]
    results.append(TextSendMessage(text=f'{name}'))
    results.append(TextSendMessage(text=f'{text}'))
#results.append(LocationSendMessage(title='花的位置',latitude=f'{logitude}',altitude=f'{altitude}'))
 #messages.append(LocationSendMessage(title='Store Location', address=food[2], latitude=food[4],longitude=food[5]))
    line_bot_api.reply_message(event.reply_token,results)
    # messages.append(LocationSendMessage())

#我放的東東結束



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001, debug=True)
