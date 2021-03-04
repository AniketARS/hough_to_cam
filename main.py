import cv2
import numpy as np
from utils import Utils

cameraCapture = cv2.VideoCapture(0)
success, frame = cameraCapture.read()


def get_canny(img, lower, upper):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray, cv2.Canny(gray, lower, upper)


def apply_hough(image, canny_res):
    line_image = np.copy(image)
    lines = cv2.HoughLinesP(canny_res, rho=1, theta=np.pi / 180, threshold=60, lines=np.array([]), minLineLength=50,
                            maxLineGap=5)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color=(255, 0, 255), thickness=5)
    return line_image


while success:
    success, frame = cameraCapture.read()

    gray, canny_applied = get_canny(frame, 50, 100)
    hough_applied = apply_hough(frame, canny_applied)

    collection = Utils.stack_images(row=2, col=2, img_list=[frame, gray, canny_applied, hough_applied], scale=1)
    cv2.imshow("Result", collection)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cameraCapture.release()
cv2.destroyAllWindows()
