import cv2
import numpy as np
from morphology import Morphology


class ImgSegmentationPointDetection:
    def __init__(self):
        self.box_list = []

    def convolution(self, org_img, operator, threshold=120):
        """
        Convolution of given image with operator
        :param org_img: original image matrix
        :param operator: operator
        :return: matrix, result of convolution
        """
        new_img = np.zeros(shape=org_img.shape)  # duplicate_image(org_img)
        x_max, y_max = org_img.shape
        op_s = int(operator.shape[0] / 2)
        cnt = 0
        point = []
        for x in range(op_s, x_max - op_s):
            for y in range(op_s, y_max - op_s-5):
                val = self.mat_sum(org_img[x - op_s:x + op_s + 1, y - op_s:y + op_s + 1], operator)
                if val >= threshold:
                    new_img[x][y] = 255
                    cnt += 1
                    point = [x,y]
        print("point ", point)
        return new_img

    def mat_sum(self, mat_a, mat_b):
        """
        Match respective elements in given matrices and return sum of matches
        :param mat_a: matrix a
        :param mat_b: matrix b
        :return: sum of multiplications
        """
        sum = 0
        for i in range(mat_a.shape[0]):
            for j in range(mat_a.shape[1]):
                sum += mat_a[i][j] * mat_b[i][j]
        return sum

    def get_kernel(self):
        a = 3
        b = 3
        kernel = np.ones((a, b))
        kernel[:, :] = -1
        kernel[1][1] = 8
        kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        return kernel

    def threshold(self,img,threshold):
        new_img = np.zeros(img.shape)
        dict_ = {}
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                key=int(img[i][j])
                if key == 0:
                    continue
                if key in dict_:
                    dict_[key] += 1
                else:
                    dict_[key] = 1

        import matplotlib.pylab as plt
        # plot histogram of intensity vs number of pixel with intensity
        lists = sorted(dict_.items())  # sorted by key, return a list of tuples
        x, y = zip(*lists)  # unpack a list of pairs into two tuples
        plt.plot(x, y)
        plt.show()

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] >= threshold:
                    new_img[i][j] = 255
        return new_img

    def task_a(self):
        image = cv2.imread("input/point.jpg", 0)
        _image1 = self.convolution(image, self.get_kernel(), 290)
        cv2.imwrite('task2_a.jpg', _image1)

    def task_b(self):
        image = cv2.imread("input/segment.jpg", 0)
        _image2 = self.threshold(image, 202)

        t = Morphology()
        I = t.dilation(t.erosion(t.dilation(_image2)))
        cv2.imshow("image",I)
        cv2.waitKey(0)
        [x, y] = I.shape
        I = I / 255
        white_val = 50
        new_img = np.zeros(shape=I.shape)
        opt_img = np.zeros(shape=I.shape)
        x_axis_sum = np.sum(I, axis=0)
        for i in range(len(x_axis_sum)):
            if x_axis_sum[i] < 2:
                new_img[:, i] = white_val
        y_axis_sum = np.sum(I, axis=1)
        for i in range(len(y_axis_sum)):
            if y_axis_sum[i] < 3:
                new_img[i, :] = white_val

        for i in range(x):
            j = 0
            while j < y:
                if new_img[i][j] != white_val:
                    len_ = 0
                    hei = 0
                    while new_img[i][j + len_] == 0:
                        len_ += 1
                    # print((i,j),(i,j+len_))
                    if sum(I[i][j:j + len_]) == 0:
                        opt_img[i, j:j + len_] = white_val
                    j += len_
                else:
                    j += 1

        for i in range(x):
            for j in range(y):
                new_img[i][j] = max(new_img[i][j], opt_img[i][j])

#        cv2.imwrite("task2_b_box.jpg", new_img)
        self.find_and_draw_box(new_img,_image2)

    def add_cord_if_not_exist(self, cord, new_c):
        flag = True
        bias = 10
        for c in cord:
            if c[0][0] - bias <= new_c[0][0] <= c[1][0] + bias and c[0][0] - bias <= new_c[1][0] <= c[1][0] + bias \
                    and c[0][1] - bias <= new_c[0][1] <= c[1][1] + bias:
                flag = False
                break

        if flag:
            cord.append(new_c)

    def find_and_draw_box(self, box_img, I):
        img = box_img
        org = I
        #cv2.imread("input/segment.jpg", 0)
        # org = np.zeros(img.shape)
        cord = []
        y_max = img.shape[1]
        for x in range(img.shape[0]):
            y = 0
            while y < y_max:
                if img[x][y] != 0:
                    y += 1
                else:
                    len_ = 0
                    hei_ = 0
                    if img[x + 10][y + len_] == 0:
                        while img[x + 10][y + len_] == 0:
                            len_ += 1
                    else:
                        while img[x][y + len_] == 0:
                            len_ += 1
                    while img[x + hei_][y] == 0 or img[x + hei_+10][y] == 0:
                        hei_ += 1

                    if len_ > 20 and hei_ > 20:
                        self.add_cord_if_not_exist(cord, [(x, y), (x + hei_, y + len_)])
                    y += len_
        # print(len(cord), cord)
        n_img = cv2.imread("input/segment.jpg")
        self.draw_box(n_img, cord)
        cv2.imwrite("task2_b.jpg", n_img)

    def draw_box(self,new_img, box_list):
        for b in box_list:
            print("draw", b)
            cv2.rectangle(new_img, b[0][::-1], b[1][::-1], (5, 255, 2), 2)


def main():
    t = ImgSegmentationPointDetection()
    t.task_b()


if __name__ == "__main__":
    main()

