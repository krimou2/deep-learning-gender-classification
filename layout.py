import PySimpleGUI as sg
import cv2

left_col =         [[sg.Text('Folder',background_color='#2d2d30',text_color='#8b8b8c'),
                     sg.In(size=(25,1), enable_events=True ,key='-FOLDER-',background_color='#414141'),
                     sg.FolderBrowse(button_color='#2d2d30')],
                    [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST-',expand_x=True,expand_y=True,background_color='#2d2d30',no_scrollbar=True,text_color='#e9deeb')]]
images_col =       [[sg.Image(filename='', key='-IMAGE-',expand_x=True,expand_y=True,background_color='#2d2d30')],
                    [sg.Text(size=(40,1), key='-TOUT-',justification='center',background_color='#2d2d30')]]
camera_col =       [[sg.Image(filename='', key='-CAMERA-',background_color='#2d2d30',size=(640,480))],
                    [sg.Text(size=(40,1), key='-TOUT-',justification='center',background_color='#2d2d30')],
                    [sg.Button('Record', size=(10, 1), font='Helvetica 14',button_color='#2d2d30'),
                     sg.Button('Stop', size=(10, 1), font='Any 14',button_color='#2d2d30'),
                     sg.In(size=(1,1), enable_events=True ,key='-VIDEO-',background_color='#414141'),
                     sg.FileBrowse('Video', size=(10, 1), font='Helvetica 14',button_color='#2d2d30')]]
tab1_layout =      [[sg.Column(left_col, element_justification='c',expand_x=True,expand_y=True,background_color='#2d2d30'), 
                     sg.VerticalSeparator(color='#545456'), 
                     sg.Column(images_col, element_justification='c',expand_x=True,expand_y=True,background_color='#2d2d30')]]
tab2_layout =      [[sg.Text('Tab 2',background_color='#2d2d30')]]
tab_group_layout = [[sg.Tab('Picture Detection', tab1_layout, font='Courier 15', key='-TAB1-',background_color='#2d2d30',title_color='#2d2d30'),
                     sg.Tab('Live Detection', camera_col,key='-TAB2-',background_color='#2d2d30',title_color='#2d2d30')]]
layout =           [[sg.TabGroup(tab_group_layout, enable_events=True, key='-TABGROUP-', background_color='#2d2d30', tab_background_color='#2d2d30', selected_background_color='#68217a', title_color='#e9deeb')]]

window = sg.Window('Test',layout,titlebar_background_color='#2d2d30',use_custom_titlebar=True,background_color='#68217a',element_padding=(3,2),font='cascadia_code')
recording = False