import cv2
import numpy as np
from PIL import Image
import dlib


def load_img(path):
    img = cv2.imread(path)
    return img


def get_eyes_nose_dlib(shape):
    nose = shape[4][1]
    left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
    left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
    right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
    right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)


def get_eyes_nose(eyes, nose):
    left_eye_x = int(eyes[0][0] + eyes[0][2] / 2)
    left_eye_y = int(eyes[0][1] + eyes[0][3] / 2)
    right_eye_x = int(eyes[1][0] + eyes[1][2] / 2)
    right_eye_y = int(eyes[1][1] + eyes[1][3] / 2)
    nose_x = int(nose[0][0] + nose[0][2] / 2)
    nose_y = int(nose[0][1] + nose[0][3] / 2)

    return (nose_x, nose_y), (right_eye_x, right_eye_y), (left_eye_x, left_eye_y)


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (
        point2[1] - point1[1]
    ) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (
        point3[1] - point2[1]
    ) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (
        point1[1] - point3[1]
    ) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3**2 - length_line2**2 - length_line1**2) / (
        2 * length_line2 * length_line1
    )
    return cos_a


def show_img(img):
    while True:
        cv2.imshow("face_alignment_app", img)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cv2.destroyAllWindows()


def shape_to_normal(shape):
    shape_normal = []
    for i in range(0, 5):
        shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
    return shape_normal


def rotation_detection_dlib(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) > 0:
        for rect in rects:
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()
            shape = predictor(gray, rect)
            shape = shape_to_normal(shape)
            nose, left_eye, right_eye = get_eyes_nose_dlib(shape)
            center_of_forehead = (
                (left_eye[0] + right_eye[0]) // 2,
                (left_eye[1] + right_eye[1]) // 2,
            )
            center_pred = (int((x + w) / 2), int((y + y) / 2))
            length_line1 = distance(center_of_forehead, nose)
            length_line2 = distance(center_pred, nose)
            length_line3 = distance(center_pred, center_of_forehead)
            cos_a = cosine_formula(length_line1, length_line2, length_line3)
            angle = np.arccos(cos_a)
            rotated_point = rotate_point(nose, center_of_forehead, angle)
            rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
            if is_between(nose, center_of_forehead, center_pred, rotated_point):
                angle = np.degrees(-angle)
            else:
                angle = np.degrees(angle)

            img = Image.fromarray(img)
            img = np.array(img.rotate(angle))
        return img
    else:
        return img


def save_img(path, img):
    cv2.imwrite(path, img)


def face_alignment_and_crop(img_path):
    img = cv2.imread(img_path)
    img = rotation_detection_dlib(img)
    img_copy = img.copy()

    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) > 0:
        for rect in rects:
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 1)
        cx, cy = int((x + w) / 2), int((h + y) / 2)
        n_off = int((h * 0.62))
        cv2.rectangle(
            img, (cx - n_off, cy - n_off), (cx + n_off, cy + n_off), (255, 0, 255), 3
        )
        crop_region = img_copy[cy - n_off : cy + n_off, cx - n_off : cx + n_off]
        resize_img = cv2.resize(crop_region, (256, 256))
        cv2.imshow("out", resize_img)
        cv2.waitKey(0)


face_alignment_and_crop("images/images (1).jpg")
