import cv2
import numpy as np

def canny(image):
#Ham nay dung de xac dinh canh tu image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #Bien doi mau thanh hinh xam
    blur = cv2.GaussianBlur(gray, (5, 5), 0)        #Khu nhieu bang cach lam mo Gaussian
    canny = cv2.Canny(blur, 50, 110)                #Ap dung thuat toan xac dinh canh Canny
    return canny

def region_of_interest(image):
#Ham nay dung de chon ra mot vung anh can xu ly
    #Tao hinh thang voi toa do tuong ung
    height = image.shape[0]
    width = image.shape[1]
    bot_left = np.array([int(0.2*width), height])
    bot_right = np.array([int(0.8*width), height])
    top_left = np.array([int(0.4*width), int(0.6*height)])
    top_right = np.array([int(0.6*width), int(0.6*height)])
    polygons = np.array([[(bot_left[0], bot_left[1]), (bot_right[0], bot_right[1]),
                          (top_right[0], top_right[1]), (top_left[0], top_left[1])]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    #mask hinh anh voi hinh thang vua tao
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Phan chua hoan thanh: chi xac dinh 2 line là lane duong
# def lane_detect(data):
#     canny_data = canny(data)
#     cv2.imshow('res', canny_data)
#     cropped_image = region_of_interest(canny_data)
#     res = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
#     resP = np.copy(res)
#     linesP = cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 50, None, 50, 10)
#
#     candidate_lines = []
#     for line in detected_lines:
#         if 0.5 <= np.abs(line.slope) <= 2:
#             candidate_lines.append(line)
#     lane_lines = compute_lane_from_candidates(candidate_lines, img_gray.shape)

def detect_image(image):
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    cdst = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    linesP = cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('image', cdstP)
    cv2.waitKey(0)

def detect_video(video):
    while(video.isOpened()):
        _, frame = video.read()
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        cdst = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
        linesP = cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

        combo_image = cv2.addWeighted(frame, 0.8, cdstP, 1, 1)
        cv2.imshow('video',combo_image)
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()