import cv2
from lane_detect import detect_image, detect_video

##################### EXECUTE CODE #############################
image = cv2.imread('./img/lane.jpg')
cap = cv2.VideoCapture('./video/test4.webm')
detect_image(image)
detect_video(cap)