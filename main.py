from flask import Flask, render_template
import sys
import cv2
import subprocess
from IPython.display import IFrame

#for running detect.py
from detect import get_pose_model,get_pose,prepare_vid_out,prepare_image,fall_detection,falling_alarm
from tqdm import tqdm
import time

sys.path.append('../utils')
app = Flask(__name__,template_folder='template')



@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/reg')
def reg():
    return render_template('register.html')

@app.route('/my')
def mylink():
    #file = open(r'G:\My Drive\Final year project\Human-Fall-Detection using YoloV7 Pose estimation model 2','r').read()
    #file = open('C:\\Users\\Acer\\PycharmProjects\\Flask\\detect.py','r').read()
    #exec(file)
    #subprocess.run("python detect1.py --source 0")

    # return  subprocess.run("python detect2.py --source 0")

    return IFrame(subprocess.run("python detect2.py --source 0"), width='50%', height=350)
#subprocess helps to run command line argument. here subprocess running detect1.py --source 0

if __name__== "__main__":
    app.run(debug=True, port=8000)