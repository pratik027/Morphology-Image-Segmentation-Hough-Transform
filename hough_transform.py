import cv2
import numpy as np
import math
from morphology import Morphology


class HoughTransform:
    def erosion(self, image, kernel):
        """
        Perform Erosion of image and kernel
        :param image: image for erosion
        :param kernel: kernel for erosion
        :return: image after erosion
        """
        new_img = np.zeros(image.shape)
        x_ops, y_ops = int(kernel.shape[0] / 2), int(kernel.shape[1] / 2)
        for i in range(x_ops, len(image) - x_ops):
            for j in range(y_ops, len(image[0]) - y_ops):
                sub_img = image[i - x_ops: i + x_ops + 1, j - y_ops: j + y_ops + 1]
                if self.cross_check_with_kernel(sub_img, kernel) == 9:
                    new_img[i][j] = 255
        return new_img

    def cross_check_with_kernel(self, img, kernel):
        """
        Count pixels from image which has more value than kernel
        :param img: image
        :param kernel: kernel
        :return: count
        """
        cnt = 0
        for x in range(kernel.shape[0]):
            for y in range(kernel.shape[1]):
                if img[x][y] >= kernel[x][y]:
                    cnt += 1
        return cnt

    def hough_space(self, edge_img, roh, theta_):
        new_image = np.zeros((roh, theta_))
        for i in range(edge_img.shape[0]):
            for j in range(edge_img.shape[1]):
                if edge_img[i][j] == 255:
                    for theta in range(theta_):
                        sin = math.sin(math.radians(theta))
                        cos = math.cos(math.radians(theta))
                        roh = int(i * cos + j * sin)
                        if roh < 0:
                            roh = roh * (-1)
                        new_image[roh][theta] += 1

        return new_image

    def find_row_theta_values(self, img, list_):
        dict_ = {}
        list_ = list_.tolist()
        for rho in range(img.shape[0]):
            for theta in range(img.shape[1]):
                if img[rho][theta] in list_:
                    dict_[rho] = theta
                    list_.remove(img[rho][theta])
        return dict_

    def task3(self, lines="vertical"):
        image = cv2.imread(r'input\hough.jpg', 0)
        rows, colums = image.shape
        t = Morphology()
        # form kernel for vertical line
        vertical_kernel = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        diagonal_kernel = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        print("Edge detection")
        kernel = np.array(vertical_kernel)
        if lines == "diagonal":
            kernel = np.array(diagonal_kernel)
            edge_image = self.erosion(t.erosion(t.dilation(self.erosion(self.erosion(self.erosion(t.erosion(t.dilation(self.erosion(self.erosion(edges, kernel), kernel))), kernel), kernel),kernel))),kernel)
        else:
            edge_image = self.erosion(edges, kernel)
        cv2.imwrite("edge" + lines + ".jpg", edge_image)
        roh = int((2 * ((rows ** 2 + colums ** 2) ** 0.5)) + 1)
        theta = 180
        print("calculating hough space")
        hough_space_matrix = self.hough_space(edge_image, roh, theta)
        cv2.imwrite("haughspace.jpg", hough_space_matrix)
        list_ = hough_space_matrix.flatten()
        print(len(list_))
        list_.sort()
        thresholds = list_[-150::10]
        print(thresholds)
        roh_theta_pair = self.find_row_theta_values(hough_space_matrix, thresholds)
        print("store result",len(roh_theta_pair))
        result_image = cv2.imread("input\hough.jpg")
        for x in range(rows):
            for y in range(colums):
                for roh, theta in roh_theta_pair.items():
                    value = int(-x/math.tan(math.radians(theta)) + roh/math.sin(math.radians(theta)))
                    if y == value:
                        result_image[x][y] = np.array([3, 255, 5])
                        break
        cv2.imwrite('task3_'+lines+'.jpg', result_image)


if __name__ == '__main__':

    HoughTransform().task3("diagonal")
