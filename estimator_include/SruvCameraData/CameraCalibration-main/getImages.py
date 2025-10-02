import cv2

url = "rtsp://admin:laborAT@139.6.109.114"
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

num = 0

while cap.isOpened():

    succes, img = cap.read()

    if succes:
        # Get the frame size
        height, width = img.shape[:2]
        print(f"Frame size: {width}x{height}")
    else:
        print("Error: Could not read frame from camera.")
    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()