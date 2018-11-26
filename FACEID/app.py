from flask import Flask, render_template


import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json
import numpy as np
app = Flask(__name__)
@app.route("/")

def camera_recog():
    print(" Dang Tai camera...")

    # vs = cv2.VideoCapture('./input-recog/x1.mp4') co the input la Video
    vs = cv2.VideoCapture(0) # input la webcam
    while True:
        _,frame = vs.read() # 1 img
        
        rects, landmarks = face_detect.detect_face(frame,80)
        
        aligns = [] # mảng của các thumnails
        positions = []
        for (i, rect) in enumerate(rects): # i = số thứ tự của các mặt (face)
            aligned_face, face_pos = aligner.align(160,frame,landmarks[i]) # thumnail & pos = L || R || center
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160: # len(img) = height of image
                aligns.append(aligned_face)
                positions.append(face_pos) # Left Right Center
                #print("face_pos",face_pos)
            else: 
                print("CANNT align face!") #log        
        if(len(aligns) > 0): # không rỗng
            features_arr = extract_feature.get_features(aligns) # features_arr = [[128]] 
            recog_data = findStudent(features_arr,positions) # return name, %
            for (i,rect) in enumerate(rects):  # rect ~ detect & align face 
                # bouding box (face) + name + %
                cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(0,255,0),thickness=2)
                cv2.putText(frame,recog_data[i][0]+"-"+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
'''
facerec_128D.txt Data Structure:
{
"Person ID": {
    "Center": [[128D vector]],
    "Left": [[128D vector]],
    "Right": [[128D Vector]]
    }
}
This function basically does a simple linear search for 
^the 128D vector with the min distance to the 128D vector of the face on screen
'''



def findStudent(features_arr, positions, thres = 0.6, percent_thres = 70):

    f = open('./facerec_128D.txt','r')
    data_set = json.loads(f.read())
    returnRes = []
    # print('face_feature ',features_arr)
    for (i,features_128D) in enumerate(features_arr): # side feature_arr = = webcam input
        result = "(unknown)"
        
        smallest = sys.maxsize # +00
        for person in data_set.keys(): # name 
            person_data = data_set[person][positions[i]] # person = name, position[i] = L R C 
            #print(positions[i]) 
            for data in person_data: # lấy từng SinhVien trong tập trainning >< features_128D (webcam)
                distance = np.sqrt(np.sum(np.square(data-features_128D))) # data side = trainning
                if(distance < smallest):
                    smallest = distance
                    result = person
        percentage =  min(100, 100 * thres / smallest) # thres = 0.6 small <0.6 thì / >(1.1 *100 > 100)
        if percentage <= percent_thres : # 
            result = "(unknown)"
        returnRes.append((result,percentage)) 
    return returnRes


def create_manual_data():
    # vs = cv2.VideoCapture('./input-train/trucnhan-test1.mp4')  # input train la video
    vs = cv2.VideoCapture(0)
    print("Nhap Ten Sinh Vien:")
    new_name = input() #ez python input() cin >>>

    f = open('./facerec_128D.txt','r') # r == read file 
    data_set = json.loads(f.read())


    # trích xuất điểm dặc trưng của face muốn train 1 ngừi
    person_imgs = {"Left" : [], "Right": [], "Center": []}
    person_features = {"Left" : [], "Right": [], "Center": []}
    print("Xoay mat Trai Phai Giua...")

    while True: # lấy vị trí trước
        _, frame = vs.read()
        rects, landmarks = face_detect.detect_face(frame, 80)  # trả về boudingbox & points
        for (i, rect) in enumerate(rects): # {1: ..., 2:....} key : value
            aligned_frame, pos = aligner.align(160,frame,landmarks[i]) # thumnail & position = L R C
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame) # person_imgs['Center'] = img
                cv2.imshow("ABC", aligned_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    # lấy feature của mỗi pos ( giai đoạn extract to vector - feature)
    for pos in person_imgs: 
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    data_set[new_name] = person_features
    f = open('./facerec_128D.txt', 'w')
    f.write(json.dumps(data_set))



def main(args):
    mode = args.mode
    if(mode == "camera"):
        camera_recog()
    elif mode == "input": # tenfile.py --mode "input" train
        create_manual_data()
    else:
        raise ValueError("Unimplemented mode")

if __name__ == '__main__':
	app.run()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
    args = parser.parse_args(sys.argv[1:])
    
    FRGraph = FaceRecGraph() # khởi tạo FaceReGraph
    aligner = AlignCustom() # Khởi tạo AlignCustom

    extract_feature = FaceFeature(FRGraph)
    
    face_detect = MTCNNDetect(FRGraph, scale_factor=2)
    mode = args.mode
    if(mode == "camera"):
        camera_recog()
    elif mode == "input":
        create_manual_data()
    else:
        raise ValueError("Error Mode !")
    return render_template('index.html')


    