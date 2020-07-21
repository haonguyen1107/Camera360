#Import các thư viện
import cv2
import dlib
from math import hypot
from PIL import Image
import numpy as np
from scipy import ndimage

print("Nhập 1 - Chọn ảnh không nghiêng\n"
      "Nhập 2 - Chọn ảnh nghiêng\n"
      "Nhập 3 - Thực hiện trên webcam\n")

nhap=int(input("Nhập: "))

if nhap==1:
    # Đọc ảnh
    # Đọc ảnh cần ghép
    img = cv2.imread('CS06.jpg')

    # Đọc ảnh snapchat
    nose_image = cv2.imread('CS015.png')

    # Gọi file landmark
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # File này có trong thư mục

    faces = detector(img)
    for face in faces:
        landmarks = predictor(img, face)

        # Xác định các landmark

        # Như trong ảnh chúng ta xác định các landmark số 29, 31, 35
        # top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(51).x, landmarks.part(51).y)
        left_nose = (landmarks.part(18).x, landmarks.part(18).y)
        right_nose = (landmarks.part(25).x, landmarks.part(25).y)

        # Tính kích thước của mũi
        # hybot() là hàm tính theo công thức sqrt(x*x+y*y)
        nose_width = int(hypot(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) * 2.5)
        # Từ kích thước chiều cao chỉ cần nhân tỉ lệ sẽ ra chiều rộng

        # CS000
        nose_height = int(nose_width * 1.2)

        # CS003
        # nose_height = int(nose_width * 1.15)

        # New nose position
        # CS003 +15 topleft
        top_left = (int(center_nose[0] - nose_width / 2)+10, int(center_nose[1] - nose_height / 2))
        bottom_right = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2))

        # Rotate nose_image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = predictor(gray, face)
        # print(landmarks)

        a = np.array([landmarks.part(27).x, landmarks.part(27).y])
        b = np.array([landmarks.part(8).x, landmarks.part(8).y])
        c = np.array([landmarks.part(27).x, 0])
        # print(a, b, c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

        if (landmarks.part(8).x >= landmarks.part(27).x):
            q = np.arccos(cosine_angle) * 100
        else:
            q = 360 - np.arccos(cosine_angle) * 100

        num_rows, num_cols = nose_image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), q, 1)
        nose_image = cv2.warpAffine(nose_image, rotation_matrix, (num_cols, num_rows))

        # Adding the new nose
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))

        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

        print(nose_mask)

        nose_area = img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_pig)

        img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = final_nose

    cv2.imshow("CV", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif nhap==2:
    # Đọc ảnh
    # Đọc ảnh cần ghép
    img = cv2.imread('CS32.jpg')

    # Đọc ảnh snapchat
    nose_image = cv2.imread('CS015.png')

    # Gọi file landmark
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # File này có trong thư mục

    faces = detector(img)
    for face in faces:
        landmarks = predictor(img, face)

        # Xác định các landmark

        # Như trong ảnh chúng ta xác định các landmark số 29, 31, 35
        # top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(33).x, landmarks.part(33).y)
        left_nose = (landmarks.part(18).x, landmarks.part(18).y)
        right_nose = (landmarks.part(25).x, landmarks.part(25).y)

        # Tính kích thước của mũi
        # hybot() là hàm tính theo công thức sqrt(x*x+y*y)
        nose_width = int(hypot(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) * 2.0)
        # Từ kích thước chiều cao chỉ cần nhân tỉ lệ sẽ ra chiều rộng

        # CS000
        nose_height = int(nose_width * 1.2)

        # CS003
        # nose_height = int(nose_width * 1.15)

        # New nose position
        # CS003 +15 topleft
        sskq=10
        top_left = (int(center_nose[0] - nose_width / 2)+sskq, int(center_nose[1] - nose_height / 2))
        bottom_right = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2))

        # Rotate nose_image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = predictor(gray, face)
        # print(landmarks)

        a = np.array([landmarks.part(27).x, landmarks.part(27).y])
        b = np.array([landmarks.part(8).x, landmarks.part(8).y])
        c = np.array([landmarks.part(27).x, 0])
        # print(a, b, c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

        if (landmarks.part(8).x >= landmarks.part(27).x):
            q = np.arccos(cosine_angle) * 100
        else:
            q = 360 - np.arccos(cosine_angle) * 100
        q=q+15
        num_rows, num_cols = nose_image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), q, 1)
        nose_image = cv2.warpAffine(nose_image, rotation_matrix, (num_cols, num_rows))

        # Adding the new nose
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))

        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

        print(nose_mask)

        nose_area = img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_pig)

        img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = final_nose

    cv2.imshow("CV", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif nhap==3:
    video_capture = cv2.VideoCapture(0)
    glasses = cv2.imread("CS014.png", -1)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    # Resize an image to a certain width
    def resize(img, width):
        r = float(width) / img.shape[1]
        dim = (width, int(img.shape[0] * r))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img


    # Combine an image that has a transparency alpha channel
    def blend_transparent(face_img, sunglasses_img):
        overlay_img = sunglasses_img[:, :, :3]
        overlay_mask = sunglasses_img[:, :, 3:]

        background_mask = 255 - overlay_mask

        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

        face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

        return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


    # Find the angle between two points
    def angle_between(point_1, point_2):
        angle_1 = np.arctan2(*point_1[::-1])
        angle_2 = np.arctan2(*point_2[::-1])
        return np.rad2deg((angle_1 - angle_2) % (2 * np.pi))


    # Start main program
    while True:

        ret, img = video_capture.read()
        img = resize(img, 700)
        img_copy = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            # detect faces
            dets = detector(gray, 1)

            # find face box bounding points
            for d in dets:
                x = d.left()
                y = d.top()
                w = d.right()
                h = d.bottom()

            dlib_rect = dlib.rectangle(x, y, w, h)

            ##############   Find facial landmarks   ##############
            detected_landmarks = predictor(gray, dlib_rect).parts()

            landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                if idx == 0:
                    eye_left = pos
                elif idx == 16:
                    eye_right = pos

                try:
                    # cv2.line(img_copy, eye_left, eye_right, color=(0, 255, 255))
                    degree = np.rad2deg(np.arctan2(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]))

                except:
                    pass

            ##############   Resize and rotate glasses   ##############

            # Translate facial object based on input object.

            eye_center = (eye_left[1] + eye_right[1]) / 2

            # Sunglasses translation
            glass_trans = int(.2 * (eye_center - y))

            # Funny tanslation
            # glass_trans = int(-.3 * (eye_center - y ))

            # Mask translation
            # glass_trans = int(-.8 * (eye_center - y))

            # resize glasses to width of face and blend images
            face_width = w - x

            # resize_glasses
            glasses_resize = resize(glasses, face_width)

            # Rotate glasses based on angle between eyes
            yG, xG, cG = glasses_resize.shape
            glasses_resize_rotated = ndimage.rotate(glasses_resize, (degree + 90))
            glass_rec_rotated = ndimage.rotate(img[y + glass_trans:y + yG + glass_trans, x:w], (degree + 90))

            # blending with rotation
            h5, w5, s5 = glass_rec_rotated.shape
            rec_resize = img_copy[y + glass_trans:y + h5 + glass_trans, x:x + w5]
            blend_glass3 = blend_transparent(rec_resize, glasses_resize_rotated)
            img_copy[y + glass_trans:y + h5 + glass_trans, x:x + w5] = blend_glass3
            cv2.imshow('Output', img_copy)

        except:
            cv2.imshow('Output', img_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
