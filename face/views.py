from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from mtcnn.mtcnn import MTCNN
import cv2
import numpy
import imutils
import os


def index(request):
    return render(request, 'index.html')


def technology(request):
    return render(request, 'technology.html')


def upload(request):
    detector = MTCNN()
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        image = cv2.imread('uploads/' + name)
        # Увеличение/уменьшение наименьшей стороны изображения до 1000 пикселей
        if image.shape[0] < image.shape[1]:
            image = imutils.resize(image, height=1000)
        else:
            image = imutils.resize(image, width=1000)
        # Получить размеры изображения
        image_size = numpy.asarray(image.shape)[0:2]
        # Получение списка лиц с координатами и значением уверенности
        faces_boxes = detector.detect_faces(image)
        # Копия изображения для рисования рамок на нём
        image_detected = image.copy()
        # Копия изображения для рисования меток на нём
        image_marked = image.copy()
        # Замена BGR на RGB (так находит в два раза больше лиц)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Цвет меток BGR
        marked_color = (0, 255, 0, 1)

        # Работа с лицами
        if faces_boxes:

            face_n = 0  # Инициализация счётчика лиц
            for face_box in faces_boxes:

                # Увеличение счётчика файлов
                face_n += 1
                # Координаты лица
                x, y, w, h = face_box['box']
                # Отступы для увеличения рамки
                d = h - w  # Разница между высотой и шириной
                w = w + d  # Делаем изображение квадратным
                x = numpy.maximum(x - round(d / 2), 0)
                x1 = numpy.maximum(x - round(w / 4), 0)
                y1 = numpy.maximum(y - round(h / 4), 0)
                x2 = numpy.minimum(x + w + round(w / 4), image_size[1])
                y2 = numpy.minimum(y + h + round(h / 4), image_size[0])

                # Получение картинки с лицом
                cropped = image_detected[y1:y2, x1:x2, :]
                face_image = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)

                # Имя файла (уверенность + номер)
                face_file_name = str(face_box['confidence']) + '.' + str(face_n) + '.jpg'

                # Отборка лиц {selected|rejected}
                if face_box['confidence'] > 0.99:  # 0.99 - уверенность сети в процентах что это лицо
                    # Рисует белый квадрат на картинке по координатам
                    cv2.rectangle(
                        image_detected,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255, 1),
                        1
                    )

                else:
                    # Рисует красный квадрат на картинке по координатам
                    cv2.rectangle(
                        image_detected,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255, 1),
                        1
                    )

                # Рисует левый глаз
                cv2.rectangle(
                    image_marked,
                    (face_box['keypoints']['left_eye'][0], face_box['keypoints']['left_eye'][1]),
                    (face_box['keypoints']['left_eye'][0] + 1, face_box['keypoints']['left_eye'][1] + 1),
                    marked_color,
                    1
                )

                # Рисует правый глаз
                cv2.rectangle(
                    image_marked,
                    (face_box['keypoints']['right_eye'][0], face_box['keypoints']['right_eye'][1]),
                    (face_box['keypoints']['right_eye'][0] + 1, face_box['keypoints']['right_eye'][1] + 1),
                    marked_color,
                    1
                )

                # Рисует нос
                cv2.rectangle(
                    image_marked,
                    (face_box['keypoints']['nose'][0], face_box['keypoints']['nose'][1]),
                    (face_box['keypoints']['nose'][0] + 1, face_box['keypoints']['nose'][1] + 1),
                    marked_color,
                    1
                )

                # Рисует левую часть рта
                cv2.rectangle(
                    image_marked,
                    (face_box['keypoints']['mouth_left'][0], face_box['keypoints']['mouth_left'][1]),
                    (face_box['keypoints']['mouth_left'][0] + 1, face_box['keypoints']['mouth_left'][1] + 1),
                    marked_color,
                    1
                )

                # Рисует правую часть рта
                cv2.rectangle(
                    image_marked,
                    (face_box['keypoints']['mouth_right'][0], face_box['keypoints']['mouth_right'][1]),
                    (face_box['keypoints']['mouth_right'][0] + 1, face_box['keypoints']['mouth_right'][1] + 1),
                    marked_color,
                    1
                )

            # Сохранение исходного изображения с выделенными лицами
            detected_path = os.path.join('uploads/detected', 'detected_' + name)
            cv2.imwrite(detected_path, image_detected)

            # Сохранение исходного изображения с точками
            marked_path = os.path.join('uploads/marked', 'marked_' + name)
            cv2.imwrite(marked_path, image_marked)

        context['marked'] = fs.url('marked/marked_' + name)
        context['detected'] = fs.url('detected/detected_' + name)

    return render(request, 'upload.html', context)
