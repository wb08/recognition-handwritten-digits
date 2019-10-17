from PIL import Image, ImageFilter
import tensorflow as tf
import cv2
import numpy as np
import os
from time import sleep



try:
    os.remove('digits.txt')
except FileNotFoundError:
    pass

directory ='./images_digit'
files = os.listdir(directory)
for file in files:
    os.remove('images_digit/'+str(file))


def image_enhancement(path_to_image):
    im = Image.open(path_to_image)
    width, height = im.size
    width = width
    height = height
    big_img = path_to_image
    resized_img = im.resize((width, height), Image.ANTIALIAS)
    save_resized_img = resized_img.save(big_img)
    return big_img

big_img=image_enhancement('Enter the path to the file')

def sort_file(path_to_directory):
    files=os.listdir(path_to_directory)
    files=sorted(files)
    return files

def sort_contours(cnts, method="left-to-right"):
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


def cropping():
    img = cv2.imread(big_img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray_img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((2, 2)))
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    (contours, boundingBoxes) = sort_contours(contours, method="left-to-right")
    nh, nw = img.shape[:2]
    for contour in contours:
        for i in range(1, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])

            if h < 0.2 * nh:
                continue

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1, cv2.LINE_AA)
            img_crop = img[y:y + h, x:x + w]
            gray_img_norm = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('images_digit/digit{}.jpg'.format(i), img_crop)



def predicting(imvalue):

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    def constant_width(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = constant_width([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = constant_width([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = constant_width([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = constant_width([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model2.ckpt")
        prediction = tf.argmax(y_conv, 1)
        return prediction.eval(feed_dict={x: [imvalue], keep_prob: 1.0}, session=sess)



def imageprepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))

    if width > height:
        nheight = int(round((20.0 / width * height), 0))
        if nheight==0:
            nheigth = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1
        # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))


    tv = list(newImage.getdata())

    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva



def main(argv):
    imvalue = imageprepare(argv)
    predint = predicting(imvalue)
    file = open('digits.txt', 'a')
    file.write(str(predint[0]))



if __name__ == "__main__":
    cropping()
    sleep(8)
    num_graph=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    directory = './images_digit'
    files = sort_file(directory)

    for i in range(len(files)):
        num_graph[i] = tf.Graph()
        with num_graph[i].as_default():
            main(directory + '/' + str(files[i]))