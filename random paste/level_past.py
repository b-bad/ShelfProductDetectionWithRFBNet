import numpy as np
import cv2
import random
from xml.dom import minidom
import os
import random

from lxml.etree import ElementTree, Element, SubElement, tostring




def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

    

def findboarder_horizon(img):
    
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    a = np.max(img_gray, axis=0)

    for i in range(len(a)):
        if a[i] > 128:
            left_boarder = i
            break

    for j in range(len(a)-1,-1,-1):
        if a[j]> 128:
            right_boarder = j
            break
        
    print("左边界" + str(left_boarder) + "右边界" + str(right_boarder))

    
    return left_boarder, right_boarder

def findboarder_vertical(img):


    
    
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    a = np.max(img_gray, axis=1)

    for i in range(len(a)):
        if a[i] > 128:
            top_boarder = i
            break

    for j in range(len(a)-1,-1,-1):
        if a[j]> 128:
            bottom_boarder = j
            break
        
    print("上边界" + str(top_boarder) + "下边界" + str(bottom_boarder))

    
    return top_boarder, bottom_boarder
    

def LevelPaste23(img_path, mask_path, origin_path, xml_path):


    print("*******************************************")
    print(xml_path)
    print(origin_path)
    
    tree=ElementTree()
    tree.parse(xml_path)

    root=tree.getroot()
    
    for object in root.findall('object'):
        root.remove(object)

    
    # img为贴图原图 mask为贴图的mask origin为背景图
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    origin = cv2.imread(origin_path)
    img_shape = img.shape[0:2]
    mask_shape = mask.shape[0:2]
    origin_shape = origin.shape[0:2]
    print(img_shape)
    print(mask_shape)
    print(origin_shape)

    # 计算行数、堆叠数
    # stack_num = random.randint(0,1) + 2
    stack_num = 1
    if origin_shape[0]>origin_shape[1]:
        print("采用三排粘贴模式")
        row = 3

    elif origin_shape[0]<=origin_shape[1]:
        print("采用两排粘贴模式")
        row = 2

    # 计算间隔， 并根据间隔计算出img经过resize后的大小
    y_interval = origin_shape[0]/row
    margin = 0.0716075226186985
    # margin = random.uniform(0.05, 0.15)
    print("纵向margin" + str(margin))
    img_resize_y = int ( y_interval * ( 1 - margin*2) / stack_num )
    y_margin = int (margin * y_interval)
    print("y_margin" + str(y_margin))
    print("img_resize_y" + str(img_resize_y))

    # 进行resize, 滤波
    fx = fy = img_resize_y/img_shape[0]
    img_resize_ori = cv2.resize(img, (0, 0), fx=fx, fy=fy)
    print("缩放后图像大小：" + str(img_resize_ori.shape))
    mask_resize = cv2.resize(mask,(int(img_resize_ori.shape[1]), int(img_resize_ori.shape[0])))

    img_resize = cv2.blur(img_resize_ori, (2, 2))
    cv2.imwrite("1_resize_ori.jpg", img_resize_ori)
    cv2.imwrite("1_mask_resize.jpg", mask_resize)
    cv2.imwrite("1_resize.jpg", img_resize)
    # img_resize_x = img_resize.shape[1]
    # img_resize_x = int(img_resize.shape[1] * 0.95)

    # 计算图片边界每行个数
    left_boarder, right_boarder = findboarder_horizon(mask_resize)
    img_resize_x = right_boarder - left_boarder
    column = ( origin_shape[1]//img_resize_x)
    

    top_boarder, bottom_boarder = findboarder_vertical(mask_resize)
    img_resize_y_cut = bottom_boarder - top_boarder


    if (column<=7):
        print("每行物体个数"+str(column))

        left_space = int(origin_shape[1] - img_resize_x * column)/2
        
        # x_margin = (origin_shape[1] - column*img_resize.shape[1])//(column)
        # x_margin = int((origin_shape[1] - column*img_resize.shape[1] - left_space*0.5))//(column)

        # 计算粘贴的起始坐标
        N = row * column
        start_node  = np.zeros((row, column, 2), dtype=np.uint16)
        print(start_node.shape)

        for i in range(row):
            for j in range(column):
                # y_loc = random.randint(int((2*i+1)*y_margin + i*img_resize_y*0.95),int((2*i+1)*y_margin + i*img_resize_y*1.05))
                # 采用正负5个象素
                y_loc = random.randint(int((2*i+1)*y_margin + i*img_resize_y*stack_num-5 - 1*top_boarder),
                                       int((2*i+1)*y_margin + stack_num*i*img_resize_y+5 - 1*top_boarder))
                if(y_loc)<0:
                    y_loc = 0
                if(y_loc)>=origin.shape[0]:
                    y_loc = origin.shape[0] - 1
                # x_loc = random.randint(j*x_margin +img_resize.shape[1]*j, (j+1)*x_margin +img_resize.shape[1]*j)
                x_loc = left_space + j * img_resize_x - 1*left_boarder
                if(x_loc)<0:
                    x_loc = 0
                if(x_loc)>=origin.shape[1]:
                    x_loc = origin.shape[1] - 1
                start_node[i][j] = [y_loc, x_loc]
        print(start_node)

        #进行MASK制作
        sss1 = np.zeros(origin.shape,dtype=np.uint8)
        sss2 = np.zeros(origin.shape,dtype=np.uint8)

        for i in range(row):
            for j in range(column):
                for k in range(stack_num):
                    for ii in range(start_node[i][j][0] + k*img_resize_y_cut, start_node[i][j][0] + img_resize_y_cut + k*img_resize_y_cut):
                        for jj in range(start_node[i][j][1], start_node[i][j][1] + img_resize_x):
                            sss1[ii,jj,:] = img_resize[ii-start_node[i][j][0]-k*img_resize_y_cut+top_boarder,jj-start_node[i][j][1] + left_boarder,:]
        cv2.imwrite("sss1.jpg", sss1)


        for i in range(row):
            for j in range(column):
                for k in range(stack_num):
                    node_object = SubElement(root, 'object')
                    node_name = SubElement(node_object, 'name')
                    node_name.text = "wt"
                    node_pose = SubElement(node_object, 'pose')
                    node_pose.text = 'Unspecified'
                    node_truncated = SubElement(node_object, 'truncated')
                    node_truncated.text = '0'
                    node_difficult = SubElement(node_object, 'difficult')
                    node_difficult.text = '0'
                    node_bndbox = SubElement(node_object, 'bndbox')
                    node_xmin = SubElement(node_bndbox, 'xmin')
                    node_xmin.text = str(start_node[i][j][1])
                    node_ymin = SubElement(node_bndbox, 'ymin')
                    node_ymin.text = str(start_node[i][j][0]+k*img_resize_y_cut)
                    node_xmax = SubElement(node_bndbox, 'xmax')
                    node_xmax.text = str(start_node[i][j][1] + img_resize_x)
                    node_ymax = SubElement(node_bndbox, 'ymax')
                    node_ymax.text = str(start_node[i][j][0] + img_resize_y_cut+k*img_resize_y_cut)
                    for ii in range(start_node[i][j][0]+k*img_resize_y_cut, start_node[i][j][0] + img_resize_y_cut+k*img_resize_y_cut,):
                        for jj in range(start_node[i][j][1], start_node[i][j][1] + img_resize_x):
                            sss2[ii,jj,:] = mask_resize[ii-start_node[i][j][0]-k*img_resize_y_cut+top_boarder ,jj-start_node[i][j][1] + left_boarder,:]
        cv2.imwrite("sss2.jpg", sss2)


    else:
        print("每行物体个数7个， 已达上限")
        column = 7

        # 生成坐标,x方向均匀分布
        N = row * column
        start_node  = np.zeros((row, column, 2), dtype=np.uint16)
        print(start_node.shape)
        margin_x = int(origin_shape[1] - img_resize_x * column)/(column+1)
        print("X方向间隔" + str(margin_x))

        for i in range(row):
            for j in range(column):
                # y_loc = random.randint(int((2*i+1)*y_margin + i*img_resize_y*0.95),int((2*i+1)*y_margin + i*img_resize_y*1.05))
                # 采用正负5个象素
                y_loc = random.randint(int((2*i+1)*y_margin + i*img_resize_y*stack_num-5 - 1*top_boarder),
                                       int((2*i+1)*y_margin + stack_num*i*img_resize_y+5 - 1*top_boarder))
                # x_loc = random.randint(j*x_margin +img_resize.shape[1]*j, (j+1)*x_margin +img_resize.shape[1]*j)
                x_loc = margin_x * (j+1) + j*img_resize_x
                start_node[i][j] = [y_loc, x_loc]
        print(start_node)


        sss1 = np.zeros(origin.shape,dtype=np.uint8)
        sss2 = np.zeros(origin.shape,dtype=np.uint8)



        for i in range(row):
            for j in range(column):
                for k in range(stack_num):
                    for ii in range(start_node[i][j][0] + k*img_resize_y_cut, start_node[i][j][0] + img_resize_y_cut + k*img_resize_y_cut):
                        for jj in range(start_node[i][j][1], start_node[i][j][1] + img_resize_x):
                            sss1[ii,jj,:] = img_resize[ii-start_node[i][j][0]-k*img_resize_y_cut+top_boarder, jj-start_node[i][j][1] + left_boarder,:]
        cv2.imwrite("sss1.jpg", sss1)

        for i in range(row):
            for j in range(column):
                for k in range(stack_num):
                    node_object = SubElement(root, 'object')
                    node_name = SubElement(node_object, 'name')
                    node_name.text = "wt"
                    node_pose = SubElement(node_object, 'pose')
                    node_pose.text = 'Unspecified'
                    node_truncated = SubElement(node_object, 'truncated')
                    node_truncated.text = '0'
                    node_difficult = SubElement(node_object, 'difficult')
                    node_difficult.text = '0'
                    node_bndbox = SubElement(node_object, 'bndbox')
                    node_xmin = SubElement(node_bndbox, 'xmin')
                    node_xmin.text = str(start_node[i][j][1])
                    node_ymin = SubElement(node_bndbox, 'ymin')
                    node_ymin.text = str(start_node[i][j][0]+k*img_resize_y_cut)
                    node_xmax = SubElement(node_bndbox, 'xmax')
                    node_xmax.text = str(start_node[i][j][1] + img_resize_x)
                    node_ymax = SubElement(node_bndbox, 'ymax')
                    node_ymax.text = str(start_node[i][j][0] + img_resize_y_cut+k*img_resize_y_cut)
                    for ii in range(start_node[i][j][0]+k*img_resize_y_cut, start_node[i][j][0] + img_resize_y_cut+k*img_resize_y_cut,):
                        for jj in range(start_node[i][j][1], start_node[i][j][1] + img_resize_x):
                            sss2[ii,jj,:] = mask_resize[ii-start_node[i][j][0]-k*img_resize_y_cut+top_boarder ,jj-start_node[i][j][1] + left_boarder,:]
        cv2.imwrite("sss2.jpg", sss2)
        
    #贴图
    sss2_gray = cv2.cvtColor(sss2,cv2.COLOR_BGR2GRAY)
    for i in range(origin_shape[0]):
        for j in range(origin.shape[1]):
            if sss2_gray[i, j] > 128:
                origin[i, j, :] = sss1[i, j, :]

    cv2.imwrite("merge.jpg", origin)

    indent(root,0)
    tree.write(xml_path,encoding="utf-8", method="xml")
    print("*******************************************")
    
LevelPaste23("1.jpg", "1_mask.jpg", "000001.jpg", "000001.xml")


