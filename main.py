import PySimpleGUI as sg
import os.path
import tensorflow as tf
import cv2
import cvlib as cv
import numpy as np
import os
import detect
import convert
import layout
from tensorflow.keras.preprocessing.image import img_to_array

recording = False

while True:
    event, values = layout.window.read(timeout
    =20)
    if event in (sg.WIN_CLOSED, 'Exit'):
        break

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    if event == '-FOLDER-':
        folder = values['-FOLDER-']
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []
        fnames = [f for f in file_list if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", "jpeg", ".tiff", ".bmp"))]
        layout.window['-FILE LIST-'].update(fnames)

    if event == '-FILE LIST-':
        try:
            filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            prediction = detect.detect_hijab(filename)
            print(prediction)
            if prediction == 'hijab':
                layout.window['-TOUT-'].update("woman (hijab)")
            else:
                prediction = detect.detect_gender(filename)
                layout.window['-TOUT-'].update(prediction)
            new_size = 200, 200
            layout.window['-IMAGE-'].update(data=convert.convert_to_bytes(filename, resize=new_size))
        except Exception as E:
            print(f'** Error {E} **')
            pass

    if event == 'Record':
        cap = cv2.VideoCapture(0)
        recording = True

    if event == '-VIDEO-':
        recording = True
        video = values['-VIDEO-']
        print(video)
        cap = cv2.VideoCapture(video)

    if event == 'Stop':
        recording = False
        img = np.full((480, 640), 45)
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        layout.window['-CAMERA-'].update(data=imgbytes)
        cap.release()

    if recording:
        ret, frame = cap.read()
        face, confidence = cv.detect_face(frame)
        for idx, f in enumerate(face):        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
            face_crop = np.copy(frame[startY:endY,startX:endX])
            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue
            face_crop = cv2.resize(face_crop, (96,96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)
            conf = detect.gender_model.predict(face_crop)[0]
            idx = np.argmax(conf)
            label = detect.gender_classes[idx]
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        layout.window['-CAMERA-'].update(data=imgbytes)

layout.window.close()