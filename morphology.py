import cv2
import numpy as np


class Morphology:
    def convolution(self, org_img, operator, op="erosion"):
        """
        Perform convolution on image and return convoluted image
        :param org_img: Image to convolute
        :param operator: Operator for convolution
        :param op: operation to perform "erosion" and "dilation"
        :return: convoluted Image
        """
        new_img = np.zeros(shape=org_img.shape)  # duplicate_image(org_img)
        x_max = org_img.shape[0]
        y_max = org_img.shape[1]
        op_s = int(operator.shape[0] / 2)
        kernal_size=operator.shape[0]*operator.shape[1]
        for x in range(op_s, x_max - op_s):
            for y in range(op_s, y_max - op_s):
                mat_sum = self.mat_sum(org_img[x - op_s:x + op_s + 1, y - op_s:y + op_s + 1], operator)
                if mat_sum == kernal_size:
                    new_img[x][y] = 255
                elif op == "dilation" and mat_sum != 0:
                    new_img[x][y] = 255
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
                if mat_a[i][j] == mat_b[i][j]:
                    sum += 1
        return sum

    def get_kernel(self,):
        a = 5
        b = 5
        kernel = np.ones((a, b))
        kernel[:, :] = 255

        return kernel

    def erosion(self, image):
        kernel = self.get_kernel()
        return self.convolution(image,kernel,op="erosion")

    def dilation(self, image):
        kernel = self.get_kernel()
        return self.convolution(image, kernel, op="dilation")

    def task(self):
        image = cv2.imread("input/noise.jpg", 0)
        _image1 = self.dilation(self.erosion(self.erosion(self.dilation(image))))
        cv2.imwrite('res_noise1.jpg', _image1)
        _image2 = self.erosion(self.dilation(self.dilation(self.erosion(image))))
        cv2.imwrite('res_noise2.jpg', _image2)

        cv2.imwrite('res_bound1.jpg', (_image1 - self.erosion(_image1)))
        cv2.imwrite('res_bound2.jpg', (_image2 - self.erosion(_image2)))


def main():
    t = Morphology()
    t.task()


if __name__ == "__main__":
    main()

