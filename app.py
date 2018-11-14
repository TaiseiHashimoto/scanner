import cv2
from cvui import cvui
import numpy as np
import utility
import scanner

def read_webcam(camera):
    cap = camera.read()[1].transpose(1, 0, 2)[:, ::-1]    # 90度回転
    return cap

if __name__ == '__main__':
    dst_size = (600, 450)

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    assert camera.isOpened(), "Cannot open camera"

    # 画像の表示サイズを確定する
    cap = utility.shrink_img(read_webcam(camera), dst_size)
    img_size = cap.shape
    window_size = (img_size[0] + 65, img_size[1] * 2)

    WINDOW_NAME = "Easy Scanner"
    cvui.init(WINDOW_NAME, 20)

    STATE_START = 0
    STATE_CAPTURED = 1
    STATE_SCANNED = 2
    STATE_MULTI_START = 3
    STATE_MULTI_CAPTURED = 4
    STATE_BLENDED = 5

    state = STATE_START
    n_subimgs = 0

    window = np.zeros((window_size[0], window_size[1], 3), np.uint8)
    img1 = np.zeros((img_size[0], img_size[1], 3), np.uint8)
    img2 = np.zeros((img_size[0], img_size[1], 3), np.uint8)
    bgcolor = (49, 52, 49)
    window[:] = bgcolor
    img1[:] = bgcolor
    img2[:] = bgcolor

    main_img = []
    sub_imgs = []
    blended = []
    intermediates = {}
    while True:
        if state == STATE_START or state == STATE_MULTI_START:
            img1 = utility.shrink_img(read_webcam(camera), dst_size)

        # 各種ボタン
        cvui.beginRow(window, 0, 0)
        flag_capture = cvui.button(100, 30, '&Capture')
        flag_scan = cvui.button(100, 30, '&Scan')
        flag_add_capture = cvui.button(120, 30, '&Add capture')
        flag_blend = cvui.button(100, 30, '&Blend')
        flag_restart = cvui.button(100, 30, '&Restart')
        flag_intermediate = cvui.button(180, 30, '&Intermediates')
        flag_quit = cvui.button(100, 30, '&Quit')
        cvui.endRow()

        if flag_capture:
            if state == STATE_START:
                main_img = img1
                cv2.imwrite("original.jpeg", main_img)
                state = STATE_CAPTURED
            elif state == STATE_MULTI_START:
                sub_imgs.append(img1)
                n_subimgs += 1
                state = STATE_MULTI_CAPTURED
            else:
                print("Unable to capture. Please push Restart button or Add capture button.")

        if flag_scan:
            if state == STATE_CAPTURED:
                main_img = scanner.scan(main_img, intermediates)
                if main_img is not None:
                    img2 = main_img
                    cv2.imwrite("detected.jpeg", main_img)
                    state = STATE_SCANNED
                else:
                    print("scan failed")
            else:
                print("Please capture first.")
            
        if flag_blend:
            if STATE_MULTI_CAPTURED:
                blended = scanner.blend(main_img, sub_imgs, intermediates)
                if blended is not None:
                    cv2.imwrite("blended.jpeg", blended)
                    img2 = blended.copy()
                    state = STATE_BLENDED

        if flag_restart:
            img2[:] = bgcolor
            sub_imgs = []
            n_subimgs = 0
            state = STATE_START

        if flag_add_capture:
            if state == STATE_SCANNED or state == STATE_BLENDED:
                state = STATE_MULTI_START
            else:
                print("Please scan first.")

        if flag_intermediate:
        	if 'lines' in intermediates:
	        	scan = np.empty((img_size[0], img_size[1]*4, 3), dtype=np.uint8)
	        	scan[:, :img_size[1]] = intermediates['lines']
	        	scan[:, img_size[1]:img_size[1]*2] = intermediates['segments']
	        	scan[:, img_size[1]*2:img_size[1]*3] = intermediates['intersections']
	        	scan[:, img_size[1]*3:] = intermediates['detected']
	        	cv2.imshow("scan", scan)
        	if 'main_state' in intermediates:
        		blend = np.empty((img_size[0], img_size[1]//2*3, 3), dtype=np.uint8)
        		blend[:img_size[0]//2, :img_size[1]//2] = main_img[::2, ::2]
        		blend[img_size[0]//2:, :img_size[1]//2] = intermediates['sub_aligned_0'][::2, ::2]
        		blend[:img_size[0]//2, img_size[1]//2:img_size[1]] = intermediates['main_weight'][::2, ::2]
        		blend[img_size[0]//2:, img_size[1]//2:img_size[1]] = intermediates['sub_weight_0'][::2, ::2]
        		blend[:img_size[0]//2, img_size[1]:] = intermediates['blended_dark'][::2, ::2]
        		blend[img_size[0]//2:, img_size[1]:] = blended[::2, ::2]
        		cv2.imshow("blend", blend)
	        else:
	        	print("Please scan first.")

        if flag_quit:
            break

        cvui.text(window, 150, 40, "orginal image", 0.6)
        cvui.text(window, img_size[1] + 150, 40, "processed image", 0.6)

        cvui.image(window, 0, 65, img1)
        cvui.image(window, img_size[1], 65, img2)

        cvui.imshow(WINDOW_NAME, window)