import cv2
import matplotlib.pyplot as plt
import numpy as np
import pprint  
import scipy.signal as signal
import os

def brisk_detection(path):
    img = cv2.imread(path)


    #plt.imshow(img)

    brisk = cv2.BRISK_create()

    (kpt, desc) = brisk.detectAndCompute(img, None)

    #brisk检测

    '''
    bk_img = img.copy()
    out_img = img.copy()
    out_img = cv2.drawKeypoints(bk_img, kpt, out_img)
    #plt.figure(2)
    plt.imshow(out_img)
    '''
    result = np.zeros(img.shape[0:2])

    for i in range(len(kpt)):
        place = list(kpt[i].pt)
        place[0] = int(place[0])
        place[1] = int(place[1])
        amp = int(kpt[i].size)
        result[place[1]][place[0]] = amp

    #结果转换为图像/二维数组

    final = result.sum(axis=0)

    #垂直投影
    
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
    #滑动平均滤波


    sm = smooth(final,9)


    print(np.mean(sm))
    ave = np.mean(sm)


    plt.plot(sm)
    plt.show()
    #plt.plot(final,color='red')



    
    '''
    一种找极小值的方式，出现值相等时无法识别，有问题
    print(sm[signal.argrelextrema(sm, np.greater)])
    print(signal.argrelextrema(sm, np.greater))

    plt.plot(signal.argrelextrema(sm,np.greater)[0],sm[signal.argrelextrema(sm, np.greater)],'o')
    plt.plot(signal.argrelextrema(-sm,np.greater)[0],sm[signal.argrelextrema(-sm, np.greater)],'+')
    # plt.plot(peakutils.index(-x),x[peakutils.index(-x)],'*')
    '''



    
    sm0 = sm.copy()
    for i in range(len(sm0)):
        if sm0[i]>5:
            sm0[i]=5

            
    plt.plot(sm,color='green')

    peaks, _ = signal.find_peaks(-sm0, distance=20)
    plt.plot(peaks, sm[peaks], "x")
    #峰值点（极小值）检测

    

 #   print(sm)

 #   plt.show()



    fig = plt.gcf()
    #fig.set_size_inches(float(img.shape[1]/100/5),7.0/5) #dpi = 300, output = 700*700 pixels
    fig.set_size_inches(float(img.shape[1]/100.0+0.01),7.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    #fig.savefig("9_pp.png", format='png', transparent=True, dpi=300, pad_inches = 0)
    fig.savefig(path.split('.')[0]+ "_p.jpg", dpi=100)

    img_1 = cv2.imread(path)
    img_2 = cv2.imread(path.split('.')[0]+ "_p.jpg")
    img_2 = cv2.resize(img_2, (img.shape[1], 700))

    #将峰值点进行作图，并将两张图进行合并

    for i in range(len(peaks)):
        img_1[:,peaks[i],:]=0


    print(img_1.shape)
    print(img_2.shape)

    img_r = np.concatenate((img_1, img_2), axis = 0)

    cv2.imwrite(path.split('.')[0]+ "_r.jpg",img_r)

    plt.close()


'''
    #print(kpt.shape)
    #print(desc.shape)
    print("kpt:",len(kpt))
    print("desc:",len(desc))

    print("kpt1:",kpt[0].pt)
    print("kpt1:",kpt[0].size)
    print("kpt1:",kpt[0].angle)
    print("kpt1:",kpt[0].response)
    print("kpt1:",kpt[0].octave)
    print("kpt1",kpt[0].class_id)


    #pprint.pprint(kpt[0])
'''

'''
#对文件夹下的图像进行brisk分割
path = os.getcwd()

filenames = os.listdir(path)
for filename in filenames:
    if(len(filename)==10):
        brisk_detection(filename)
'''

brisk_detection('000012.jpg')
