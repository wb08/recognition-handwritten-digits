import cv2
from tensorflow import keras
import numpy as np
import os
import imutils
import json
from PIL import Image
from typing import List, Dict


def delete_file():
    try:
        os.remove('other_files/digits.txt')
    except FileNotFoundError:
        pass


def sort_file(path_to_digits_directory: str) -> List[str]:
    files = os.listdir(path_to_digits_directory)
    files = sorted(files)
    return files


def final_resized_image(big_image: str, len_countors: int) -> str:
    img = cv2.imread(big_image)
    resized_img = imutils.resize(img, width=80*len_countors)
    cv2.imwrite(big_image, resized_img)
    return big_image


def sort_contours(cnts: tuple, method="left-to-right") -> tuple:
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return (cnts, boundingBoxes)


def number_circuits(medium_image: str) -> str:
    img = cv2.imread(medium_image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray_img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((2, 2)))
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    big_image = final_resized_image(medium_image, len(contours))
    return big_image


def improve_image(big_image: str) -> str:
    black = (0, 0, 0)
    white = (255, 255, 255)
    threshold = (160, 160, 160)
    img = Image.open(big_image).convert("LA")
    pixels = img.getdata()
    newPixels = []
    for pixel in pixels:
        if pixel < threshold:
            newPixels.append(black)
        else:
            newPixels.append(white)
    newImg = Image.new("RGB", img.size)
    newImg.putdata(newPixels)
    newImg.save(big_image)
    return big_image


def cropping(improved_image: str):
    image = cv2.imread(improved_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilate = cv2.dilate(thresh1, None, iterations=2)
    cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    (cnts, boundingBoxes) = sort_contours(cnts, method="left-to-right")
    nh, nw = image.shape[:2]
    for contour in cnts:
        for i in range(0, len(cnts)):
            x, y, w, h = cv2.boundingRect(cnts[i])

            if h < 0.27 * nh:
                continue

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 0, cv2.LINE_AA)
            img_crop = image[y - 10:y + h + 10, x:x + w]
            cv2.imwrite('images_digit/image{}.jpg'.format(i), img_crop)


def number_of_files(path_to_digits_directory: str) -> int:
    len_files = os.listdir(path_to_digits_directory)
    return len(len_files)


def convert_to_black(path_to_number_directory: str):
    all_files = os.listdir(path_to_number_directory)
    for file in all_files:
        im_gray = cv2.imread(path_to_number_directory+'/'+file, cv2.IMREAD_GRAYSCALE)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(path_to_number_directory+'/'+file, im_bw)


def image_resize(path_to_digits_directory: str):
    files = os.listdir(path_to_digits_directory)
    for file in files:
        im = Image.open(path_to_digits_directory+file)
        width = 50
        height = 50
        resized_img = im.resize((width, height), Image.ANTIALIAS)
        resized_img.save(path_to_digits_directory+file)


def predicting(path_to_digits_directory: str):
    image = keras.preprocessing.image
    model = keras.models.load_model('other_files/model_digits.h5')
    papka = os.listdir(path_to_digits_directory)
    papka = sorted(papka, key=lambda x: int(''.join(filter(str.isdigit, x))))
    for file_image in papka:
        path = path_to_digits_directory+file_image
        img = image.load_img(path, target_size=(50, 50))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        img = image.load_img(path, target_size=(50, 50))
        my_file = open("other_files/digits.txt", "a")
        my_file.write(str(np.argmax(classes[0])))


def read_digits(file_name: str) -> Dict[str, str]:
    arr_digits = []
    try:
        f = open('other_files/digits.txt')
        for line in f:
            arr_digits.append(line)
        data = {"name image": file_name,
                "digits": arr_digits[0]}
        return data
    except FileNotFoundError:
        pass


def size_images(path_to_digits_directory: str):
    files = os.listdir(path_to_digits_directory)
    for file in files:
        image_size = os.path.getsize(path_to_digits_directory+file)
        if image_size == 0:
            os.remove(path_to_digits_directory+file)


def form(data: List[dict]):
    file = open("other_files/result.json", "w")
    json.dump(data, file, indent=2)


def main():
    arrays_digits = []
    image_directory = './Числа 3'
    convert_to_black(image_directory)
    files = os.listdir(image_directory)
    for file in files:
        big_image = number_circuits(image_directory+'/'+file)
        improved_image = improve_image(big_image)
        try:
            cropping(improved_image)
        except ValueError:
            print("Контуров не найдено!")
        digits_directory = './images_digit/'
        sort_file(digits_directory)
        if number_of_files(digits_directory) == 0:
            print('Контуров не найдено!')
        else:
            size_images(digits_directory)
            image_resize(digits_directory)
            predicting(digits_directory)
            arr_digits = read_digits(file)
            delete_file()
            os.system('rm -rf %s/*' % './images_digit')
            arrays_digits.append(arr_digits)
        form(arrays_digits)


if __name__ == '__main__':
    main()
