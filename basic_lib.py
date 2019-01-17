import os
import numpy as np
import cv2
def Get_List(path):
    files = os.listdir(path);
    dirList = []
    fileList = []
    for f in files:
        if (os.path.isdir(path + '/' + f)):
            if (f[0] == '.'):
                pass
            else:
                dirList.append(f)
        if (os.path.isfile(path + '/' + f)):
            fileList.append(f)
    return [dirList, fileList]

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def get_muliti_bbox(img):
    tmp = (img[:,:,0]>0)*1 + (img[:,:,1]>0)*1 + (img[:,:,2]>0)*1
    tmp = tmp > 0
    sp = tmp.shape  # 行 列
    # 先做二值化处理
    test_scale = 1
    # 思想 选取目标点按以前的方式，暴力外扩 跳变点的方式无法确定是不是同一个物体
    data_map = np.ones(sp)
    glob_bbox = [sp[0], sp[1], 0, 0]
    for hang in range(0, sp[0], int(sp[0] / 100)):
        for lie in range(0, sp[1], int(sp[1] / 100)):
            if data_map[hang][lie] == 1:  # 该点还没被查找过
                if tmp[hang][lie]:  # 该处存在特征点
                    data_map[hang][lie] = 0
                    xmin = lie
                    xmax = lie
                    ymin = hang
                    ymax = hang
                    exit_flag = 0
                    # center=lie   #寻找的中心位置
                    # 先找右边
                    right_hang = hang
                    right_lie = lie
                    left_hang = hang
                    left_lie = lie

                    flag_right_out = 0
                    flag_left_out = 0
                    while exit_flag == 0:
                        flag_right_out = 0
                        flag_left_out = 0
                        # right_lie=center
                        while right_lie + test_scale < sp[1] and tmp[right_hang][right_lie]:  # 找列
                            right_lie = right_lie + test_scale

                        if xmax < right_lie:
                            xmax = right_lie;

                        # 找左边
                        # left_lie = center
                        while left_lie - test_scale > 0 and tmp[left_hang][left_lie]:  # 找列
                            left_lie = left_lie - test_scale

                        if xmin > left_lie:
                            xmin = left_lie
                        # 行标下移
                        if left_hang + test_scale < sp[0]:
                            left_hang = left_hang + test_scale
                        while left_lie + test_scale < right_lie and (tmp[left_hang][left_lie] == False):  # 找行
                            left_lie = left_lie + test_scale
                        if left_lie + test_scale >= right_lie or left_hang + test_scale >= sp[0]:
                            flag_left_out = 1

                        if right_hang + test_scale < sp[0]:
                            right_hang = right_hang + test_scale
                        while right_lie - test_scale > left_lie and (tmp[right_hang][right_lie] == False):  # 找行
                            right_lie = right_lie - test_scale
                        if right_lie - test_scale <= left_lie or right_hang + test_scale >= sp[0]:
                            flag_right_out = 1

                        if (flag_left_out == 1 and flag_right_out == 1):
                            exit_flag = 1

                    if left_hang > right_hang:
                        ymax = left_hang
                    else:
                        ymax = right_hang
                    ymin = hang
                    if xmin < glob_bbox[0]:
                        glob_bbox[0] = xmin
                    if ymin < glob_bbox[1]:
                        glob_bbox[1] = ymin
                    if xmax > glob_bbox[2]:
                        glob_bbox[2] = xmax
                    if ymax > glob_bbox[3]:
                        glob_bbox[3] = ymax
                    data_map[ymin - 1:ymax + 1, xmin - 1:xmax + 1] = 0;
    return {'xmin':glob_bbox[0],'ymin':glob_bbox[1],'xmax':glob_bbox[2],'ymax':glob_bbox[3]}

def get_bbox(img):
    tmp = img[...,0] + img[...,1] + img[...,2]
    tmp = 1*(tmp>5)
    width = tmp.shape[1]
    hight = tmp.shape[0]
    left = 0
    right = width
    top = 0
    down = hight
    index = 0
    final_result = [0,0,0,0]
    while 1:
        for index in range(left,width,5):
            if sum(tmp[:,index]) != 0:
                break
        if index >= (width-10):
            break
        left = index

        for index in range(left,width,5):
            if sum(tmp[:,index]) == 0:
                break
        right = index

        for index in range(0,hight,5):
            if sum(tmp[index,left:right]) != 0:
                break
        top = index

        for index in range(top,hight,5):
            if sum(tmp[index,left:right]) == 0:
                break
        down = index
        if (final_result[1]-final_result[0])*(final_result[3]-final_result[2]) < (right-left)*(down-top):
            final_result = [left,right,top,down]
        left = right
    # 细化部分
    left = final_result[0]
    index = final_result[0]
    for index in range(left, width,-1):
        if sum(tmp[:, index]) == 0:
            break
    left = index

    right = final_result[1]
    index = final_result[1]
    for index in range(right, width, -1):
        if sum(tmp[:, index]) != 0:
            break
    right = index

    top = final_result[2]
    index = final_result[2]
    for index in range(top, hight,-1):
        if sum(tmp[index, left:right]) == 0:
            break
    top = index

    down = final_result[3]
    index = final_result[3]
    for index in range(down, hight, -1):
        if sum(tmp[index, left:right]) != 0:
            break
    down = index
    return {"min":(left,top),"max":(right,down),"xmax":right,"xmin":left,"ymin":top,"ymax":down}

def ImageToIUV(im,IUV):
    U = IUV[:,:,1]
    V = IUV[:,:,2]
    I = IUV[:,:,0]
    TextureIm = np.zeros([24, 200, 200, 3]).astype(np.uint8)
    ###
    for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
        x,y = np.where(I==PartInd)
        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        v_tmp = ((255 - v_current_points) * 199. / 255.).astype(int)
        u_tmp = (u_current_points * 199. / 255.).astype(int)
        TextureIm[PartInd - 1,v_tmp,u_tmp,...] = im[x, y,...]
    generated_image = np.zeros((1200, 800, 3)).astype(np.uint8)
    for i in range(4):
        for j in range(6):
            generated_image[(200 * j):(200 * j + 200), (200 * i):(200 * i + 200),...] = TextureIm[(6 * i + j),...]
    return generated_image

def IUVToImage(Tex_Atlas,IUV):
    TextureIm = np.zeros([24, 200, 200, 3]).astype(np.uint8)
    for i in range(4):
        for j in range(6):
            TextureIm[(6 * i + j), :, :, :] = Tex_Atlas[(200 * j):(200 * j + 200), (200 * i):(200 * i + 200), :]
    U = IUV[:,:,1]
    V = IUV[:,:,2]
    #
    im = np.zeros(IUV.shape).astype(np.uint8)
    ###
    for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
        tex = TextureIm[PartInd-1,:,:,:].squeeze() # get texture for each part.
        ###############
        x,y = np.where(IUV[:,:,0]==PartInd)
        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        ##
        current_points = tex[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int),...]
        im[x,y,...] = current_points.astype(np.uint8)
    return im
