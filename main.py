
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS

import requests
import bs4
import os
import cv2
import time
import uuid
import imutils
import argparse
import numpy as np
from math import pow, sqrt

from flask import Flask, render_template, Response
from flask import request, jsonify
from flask import send_from_directory

from mask_detector import *
from social_dist_detector import *



stuff=list()



# FUNCTION TO GET URL
def get_html_data(url):
    data = requests.get(url)
    return data

# WEB SCRAPPING FUNCTION
def get_corona_detail_of_india():
    url= "https://www.mohfw.gov.in/"
    html_data = get_html_data(url)

    bs = bs4.BeautifulSoup(html_data.text, 'html.parser') # MAKING OF OBJECT
    info_div1 = bs.find("li",class_="bg-blue").find_all('strong', class_="mob-hide")
    active_no=info_div1[1].get_text().split()[0]
    info_div2 = bs.find("li",class_="bg-green").find_all('strong', class_="mob-hide")
    dis_no=info_div2[1].get_text().split()[0]
    info_div3 = bs.find("li",class_="bg-red").find_all('strong', class_="mob-hide")
    death_no=info_div3[1].get_text().split()[0]
    mig="1"

    all_details = (active_no,dis_no,death_no)
    return all_details


app = Flask(__name__)

@app.route('/', methods=['GET'])
def send_index():
    return render_template('index.html')

@app.route('/mail', methods=['GET'])
def send_mail():
    return send_from_directory('./templates', "mail.html")



@app.route('/social')
def social():
    """Video streaming home page."""
    return render_template("social_distance.html")


@app.route('/video_feed_sd')
def video_feed_sd():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(sd_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/mask')
def mask():
    return render_template('mask.html')


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of image tag in html code
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/<path:path>', methods=['GET'])
def send_root(path):
    return send_from_directory('./templates', path)


@app.route('/api/scrap', methods=['POST'])
def scrap():
    details=get_corona_detail_of_india()
    response = {"id":str(uuid.uuid4()),"active":details[0],"discharged":details[1],"deaths":details[2]}
    return jsonify(response)


@app.route('/api/mail', methods=['POST'])
def mail():
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    subject = request.form.get("subject")
    sender = request.form.get("email")
    msg = request.form.get("message")
    from1=""
    to=""

    mymsg=MIMEMultipart()
    mymsg['From']=from1
    mymsg['To']=to
    mymsg['Subject']=sender+"**"+subject

    mymsg.attach(MIMEText(msg, 'plain'))

    server=smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(from1, "") #password
    text=mymsg.as_string()
    server.sendmail(from1, to, text)
    server.quit()
    return '<script>alert("Message Sent");window.location.href = "/";</script>'












# TO RUN THE APP
if __name__ == '__main__':
    # RUN FUNCTION TO EXECUTE THE APP
    # DEBUD IS TRUE TO ENABLE SOME DEBUG FUNCTIONALITIES
    # THUS THE SERVER WILL START
    # ALTHOUGH UPTILL HERE WITHOUT ANY |----ROUTER----| 
    # THE WEB BROWSER DOESN'T KNOW WHERE TO GO
    # THUS, THE BROWSER WILL SHOW NO URL FOUND    
    app.run(debug=True)



# FOR VIDEO FEED HTML PAGE

# <html>

#   <head>
#     <title>Video Surveillance</title>
#   </head>

#   <body>

#     <p style = "font-family:lato;font-size:25px;">
#          Social Distancing Alert System
#     </p>
# <!--     
#     <img src="{{ url_for('video_feed') }}"> -->

#   </body>

# </html>
