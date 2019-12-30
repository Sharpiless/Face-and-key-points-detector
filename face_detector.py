import cv2
import tensorflow as tf
from network import Net
import numpy as np


class FaceDetertor(object):

    def __init__(self):

        self.model_path = './model'

        self.net = Net(is_training=False)

        self.size = 96

    def ad_threshold(self, img):

        th2 = cv2.adaptiveThreshold(img, 255,

                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

                                    cv2.THRESH_BINARY, 19, 4)  # 自适应二值化

        return th2

    def CatchUsbVideo(self, window_name, camera_index):

        # 定义主函数

        cv2.namedWindow(window_name)  # 创建摄像头窗口

        cap = cv2.VideoCapture(camera_index)  # 调用摄像头（一般电脑自带摄像头index为0）

        # 调用分类（人脸识别分类器是cv一个预训练的模型，文件名为haarcascade_frontalface_alt2.xml）

        # 在我的电脑里查找就可以找到，找到后复制到当前文件夹内

        # 我的电脑的储存路径是C:\Users\dell\AppData\Roaming\Python\Python37\site-packages\cv2\data

        classfier = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')

        # 设置边框颜色（用于框出人脸）

        color = (0, 255, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX  # 创建摄像头前置的文字框

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(self.model_path)

            if ckpt and ckpt.model_checkpoint_path:

                # 如果保存过模型，则在保存的模型的基础上继续训练

                self.net.saver.restore(sess, ckpt.model_checkpoint_path)

                print('Model Reload Successfully!')

            while cap.isOpened():

                catch, frame = cap.read()  # 读取每一帧图片

                if not catch:

                    raise Exception('Check if the camera if on.')

                    break

                # 转换为灰度图片

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                gray = self.ad_threshold(gray)  # 自适应二值化处理

                # scaleFactor 为图片缩放比例

                # minNeighbors 是至少检查为3次是人脸才作为标记，适当增大可以有效抗干扰

                # minSize 是检测的最小人脸的大小

                faceRects = classfier.detectMultiScale(

                    gray, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))

                if len(faceRects) > 0:

                    # 历遍每次检测的所有脸

                    for face in faceRects:

                        x, y, w, h = face  # face是一个元祖，返回了分类器的检测结果，包括起始点的坐标和高度宽度

                        image = frame[y-10:y+h+10, x-10:x+w+10]  # 对原图片进行裁剪

                        cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5),
                                      color, 2)  # 绘制人脸检测的线框

                        cv2.putText(frame, 'face', (x + 30, y + 30),
                                    font, 1, (255, 0, 255), 4)

                        image_x = cv2.resize(cv2.cvtColor(
                            image, cv2.COLOR_BGR2GRAY), (self.size, self.size))

                        points = self.net.test_net(image_x, sess)

                        points_x = points[::2] / self.size * w + x
                        points_y = points[1::2] / self.size * h + y

                        points_x = points_x.astype(np.int)
                        points_y = points_y.astype(np.int)

                        for x_, y_ in zip(points_x, points_y):

                            cv2.circle(frame, (x_, y_), 2, (0,0,255), -1)


                cv2.imshow(window_name, frame)  # 显示人脸检测结果

                c = cv2.waitKey(10)

                if c & 0xFF == ord('q'):

                    # 按q退出

                    break

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 1:

                    # 点x退出

                    break

        # 释放摄像头

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    Detertor = FaceDetertor()

    Detertor.CatchUsbVideo("face_detect", camera_index=0)

    # camera_index 是摄像头的编号，其中笔记本前置摄像头编号为0
