import cv2 

import numpy as np 
import operator as op 
import itertools as it, functools as ft 

from typing import List, Tuple, Dict 
from loguru import logger 

"""
    x: namedWindow
    x: resizeWindow
    x: moveWindow
    x: destroyWindow
    x: destroyAllWindows

    x: imread 
    x: imwrite
    x: imshow
    
    ?: imencode
    ?: imdecode 

    x: waitKey
    VideoCapture
    release
"""

def create_window(winname:str, win_shape:Tuple[int, int], win_position:Tuple[int, int]=None) -> None:
    width, height = win_shape # unpack a tuple 
    cv2.namedWindow(winname=winname, flags=cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(winname=winname, width=width, height=height)

    if win_position is not None:
        pos_x, pos_y = win_position  # unpack 
        cv2.moveWindow(winname=winname, x=pos_x, y=pos_y)

    logger.success(f'window : {winname} was created')

def read_image(path2image:str) -> np.ndarray:
    # cv2.IMREAD_COLOR | cv2.IMREAD_GRAYSCALE
    colored_image = cv2.imread(filename=path2image, flags=cv2.IMREAD_COLOR)
    return colored_image

def display_image(image:np.ndarray, target_window:str) -> None:
    cv2.imshow(winname=target_window, mat=image)

def save_image(image:np.ndarray, path2location:str) -> None:
    cv2.imwrite(filename=path2location, img=image)
    
def image_processing():
    logger.debug(' ... [image processing] ... ')
    create_window(winname='displat-000', win_shape=(800, 800))
    create_window(winname='display-001', win_shape=(800, 800), win_position=(1200, 100))
    
    colored_image = read_image('dataset/dog.1540.jpg')  # this will not work on windows : check path
    display_image(colored_image, 'display-001')

    # chapter 2 
    resized_image = cv2.resize(src=colored_image, dsize=(512, 512))
    gray_image = cv2.cvtColor(src=resized_image, code=cv2.COLOR_BGR2GRAY)
    save_image(image=gray_image, path2location='transformed_image.jpg')

    # response = MODEL(gra_image)
    # send response to client ...! 

    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()  # apply destroyWindow for each window 
    

def video_processing():
    logger.debug(' ... [video processing] ... ')
    path2video = '/home/ibrahima/Downloads/TownCentreXVID.mp4'



    create_window(winname='display-000', win_shape=(800, 800))
    create_window(winname='display-001', win_shape=(800, 800), win_position=(1200, 100))

    subtractor = cv2.createBackgroundSubtractorKNN()
    capture = cv2.VideoCapture(path2video)  # webcam device | path2video
    keep_capture = True 
    while keep_capture:
        key_code = cv2.waitKey(delay=25) & 0xFF 
        if key_code != 27:  # keep grabbing until user hit the [ESCAPE] button
            capture_status, colored_frame = capture.read()  # grab frame
            if capture_status:
                colored_frame = cv2.resize(colored_frame, (800, 800))
                mask = subtractor.apply(colored_frame)

                contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contour_areas = list(map(cv2.contourArea, contours))
                largest_contour_area = max(contour_areas)

                cursor = 0 
                for cnt, srf in zip(contours, contour_areas):
                    score = srf / largest_contour_area
                    if score > 0.1:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(colored_frame, (x, y), (x + w, y + h), (255, 0, 0), 3, 1)
                        cv2.drawContours(colored_frame, contours, -1, (0, 0, 255), 1)
                        cursor = cursor + 1
                logger.debug(f'{cursor:03d} persons were found')

                display_image(mask, 'display-000')
                display_image(colored_frame, 'display-001')
        else:
            keep_capture = False 
    # end loop 

    capture.release()  # free all video ressources
    logger.success('all video ressources were removed')

if __name__ == '__main__':
    logger.debug(' ... [opencv tutorial] ... ')
    video_processing()