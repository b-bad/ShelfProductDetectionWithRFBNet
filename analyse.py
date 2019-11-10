import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
import heapq

#用于读取中文路径图片的函数
def cv_read(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img

def analysis():

    path = os.getcwd()
    abspath = os.path.abspath(os.path.dirname(os.getcwd()))
    #print(path, abspath)

    files = os.listdir(path)
    for file in files:
        if file[-3:] != "jpg":
            continue
        
        print("start to analyse " + file)
        img = cv2.imread(file)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        canny = cv2.Canny(img, 60, 100)
        #cv2.imwrite(abspath + "\\canny\\" + file.split('.')[0] + "_c.jpg", canny)
        #中文路径保存方式
        cv2.imencode('.jpg', canny)[1].tofile(abspath + "\\canny\\" + file.split('.')[0] + "_c.jpg")
        
        #二值化
        ret, thresh = cv2.threshold(canny, 128, 1, cv2.THRESH_BINARY)
        #cv2.imwrite(abspath + "\\二值化\\" + file.split('.')[0] + "_d.jpg", thresh)
        #cv2.imencode('.jpg', thresh)[1].tofile(abspath + "\\二值化\\" + file.split('.')[0] + "_d.jpg")
        #垂直投影
        summ = thresh.sum(axis = 1)

        def smooth(a,WSZ):
        # a:原始数据，NumPy 1-D array containing the data to be smoothed
        # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
        # WSZ: smoothing window size needs, which must be odd number,
        # as in the original MATLAB implementation
            out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
            r = np.arange(1,WSZ-1,2)
            start = np.cumsum(a[:WSZ-1])[::2]/r
            stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
            return np.concatenate((  start , out0, stop  ))

        sm = smooth(summ, 19)
        plt.plot(sm, color="green")

        peaks, _ = signal.find_peaks(-sm, distance=80)
        plt.plot(peaks, sm[peaks], 'x')

        value = peaks.copy()

        for i in range(len(peaks)):
            value[i] = sm[peaks[i]]
        
        peaks_find = heapq.nsmallest(8, range(len(value)), value.take)

        i = 0
        while True:
            j = i + 1
            flag = peaks[peaks_find[i]]
            while j < len(peaks_find):
                if abs(flag - peaks[peaks_find[j]]) < max(200, img.shape[0] / 6):
                    peaks_find.pop(j)
                else:
                    j += 1
            i += 1
            if i >= len(peaks_find):
                break
        
        place = peaks_find.copy()
        for i in range(len(peaks_find)):
            place[i] = peaks[peaks_find[i]]

        fig = plt.gcf()
        fig.set_size_inches(img.shape[0]/100/3,7.0/3) #dpi = 300, output = 700*700 pixels
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        fig.savefig(abspath + "\\边缘直方图\\" + file.split('.')[0] + "_p.jpg", dpi=300)        
        
        img90 = np.rot90(img)
        #cv2.imwrite(abspath + "\\旋转后\\" + file.split('.')[0] + "_t.jpg", img90)
        cv2.imencode('.jpg', img90)[1].tofile(abspath + "\\旋转后\\" + file.split('.')[0] + "_t.jpg")
        
        '''img_1 = cv2.imread(abspath + "\\旋转后\\" + file.split('.')[0] + "_t.jpg")
        img_2 = cv2.imread(abspath + "\\边缘直方图\\" + file.split('.')[0] + "_p.jpg")'''
        img_1 = cv_read(abspath + "\\旋转后\\" + file.split('.')[0] + "_t.jpg")
        img_2 = cv_read(abspath + "\\边缘直方图\\" + file.split('.')[0] + "_p.jpg")

        for i in range(len(place)):
            for j in range(10):
                if(place[i]-j>=0):
                    img_1[:,place[i]-j,0]=255
                    img_1[:,place[i]-j,1]=0
                    img_1[:,place[i]-j,2]=0
                if(place[i]+j<img_1.shape[1]):
                    img_1[:,place[i]+j,0]=255
                    img_1[:,place[i]+j,1]=0
                    img_1[:,place[i]+j,2]=0
        
        img_r = np.concatenate((img_1, img_2), axis = 0)

        #cv2.imwrite(abspath + "\\结果\\" + file.split('.')[0] + "_r.jpg", img_r)
        cv2.imencode('.jpg', img_r)[1].tofile(abspath + "\\结果\\" + file.split('.')[0] + "_r.jpg")

        plt.close()

        print("finish")

if __name__ == "__main__":
    analysis()