
import configparser
import shutil
import numpy as np
import cv2

def sythetic_pic():
    img = cv2.imread("E:\\BaiduNetdiskDownload\\fk-egp00676_job\\fk-egp00676.b2\\Layer.bmp", cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    result_img = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(256):
        tmp = np.zeros(img.shape)
        tmp[img == i] = 255
        # intdex = np.reshape(intdex, (img.shape[0], img.shape[1], 1))
        # intdex = np.tile(intdex, (1, 1, 3))
        # print(intdex.shape)
        # print(result_img[intdex])
        # print(result_img[intdex].shape)
        # count = np.count_nonzero(tmp)
        # if count >0 :
        #     print("there are mask", i)
        bshow = False
        putcolor = np.array([0, 0, 0])
        if i == 0:
            intdex = (img == i)
            print("result_img[:, :, 0].shape:", result_img[:, :, 0].shape)
            print("result_img[:, :, 0][intdex].shape:", result_img[:, :, 0][intdex].shape)
            # print(putcolor[0])
            putcolor = np.array([0, 0, 0])
            result_img[:, :, 0][intdex] = putcolor[0]
            result_img[:, :, 1][intdex] = putcolor[1]
            result_img[:, :, 2][intdex] = putcolor[2]
            # result_img[intdex] = putcolor
            bshow = True
        elif i == 19:
            intdex = (img == i)
            putcolor = np.array([39, 47, 40])
            print(putcolor[0])
            # result_img[intdex] = putcolor
            result_img[:, :, 0][intdex] = putcolor[0]
            result_img[:, :, 1][intdex] = putcolor[1]
            result_img[:, :, 2][intdex] = putcolor[2]
            bshow = True
        elif i == 53:
            intdex = (img == i)
            putcolor = np.array([200, 200, 200])
            # result_img[intdex] = putcolor
            result_img[:, :, 0][intdex] = putcolor[0]
            result_img[:, :, 1][intdex] = putcolor[1]
            result_img[:, :, 2][intdex] = putcolor[2]
            bshow = True
        elif i == 174:
            intdex = (img == i)
            putcolor = np.array([63, 82, 61])
            # result_img[intdex] = putcolor
            result_img[:, :, 0][intdex] = putcolor[0]
            result_img[:, :, 1][intdex] = putcolor[1]
            result_img[:, :, 2][intdex] = putcolor[2]
            bshow = True
        if bshow:
            print(i)
            cv2.namedWindow("IMREAD_GRAYSCALE", cv2.WINDOW_NORMAL)
            cv2.imshow("IMREAD_GRAYSCALE", tmp)  # 10
            ss_sha = tmp.reshape((img.shape[0] * img.shape[1]))
            largethan0 = True
            count_largethan = 0
            print("====================================", ss_sha.shape[0])
            for zz in range(ss_sha.shape[0]):
                ss = ss_sha[zz]
                if ss <= 0:
                    if largethan0 is False:
                        count_largethan += 1
                    else:
                        print("+ ", count_largethan)
                        largethan0 = False
                        count_largethan = 1
                else:
                    if largethan0 is False:
                        print("- ", count_largethan)
                        largethan0 = True
                        count_largethan = 1
                    else:
                        largethan0 = True
                        count_largethan += 1
            cv2.waitKey(0)  # 11
    np.set_printoptions(threshold='nan')
    print(result_img.dtype)
    # result_img = result_img.astype(np.uint8)
    result_img = np.array(result_img, dtype=np.uint8)
    cv2.namedWindow("result_img", cv2.WINDOW_NORMAL)
    cv2.imshow("result_img", result_img)  # 10
    cv2.imwrite("E:\\BaiduNetdiskDownload\\fk-egp00676_job\\111.png", result_img)
    cv2.waitKey(0)  # 11

    conf_path = "E:\\BaiduNetdiskDownload\\fk-egp00676_job\\123\\1\\B.vrs"

    with open(conf_path, "r") as ngt_fp:
        ngp_list = ngt_fp.readlines()
    if len(ngp_list) >= 9:
        num_ngt = ngp_list[4]
        num_ngt = int(num_ngt.strip())
    origin_point = (-3942, -1371)
    count_save = 0
    for i, line in enumerate(ngp_list):
        if i < 11:
            continue
        count_save += 1
        line = line.strip()
        line_split = line.split(',')
        line_split = list(map(int, line_split))
        print(line_split)
        print(line_split[0]/0.035/1000, line_split[1]/0.035/1000)
        point = [(line_split[0]/0.035/1000 - origin_point[0]), (line_split[1]/0.035/1000 - origin_point[1])]
        point = list(map(int, point))
        print(point)
        #real_point =
        crop_img = result_img[result_img.shape[0] - point[1]-60:result_img.shape[0] - point[1]+60, point[0]-60:point[0]+60, :]
        #crop_img = result_img[result_img.shape[0] - point[1]:result_img.shape[0] - point[1]+120, point[0]:point[0]+120, :]
        #crop_img = result_img[point[0]:point[0]+120, point[1]:point[1]+120, :]
        print(crop_img.shape)
        save_dir = "E:\\BaiduNetdiskDownload\\fk-egp00676_job\\123\\1_std_save\\B"
        save_path = save_dir+str(count_save) + ".png"
        cv2.imwrite(save_path, crop_img)
        # cv2.namedWindow("crop_img", cv2.WINDOW_NORMAL)
        # cv2.imshow("crop_img", crop_img)  # 10
        # cv2.waitKey(0)

def sythetic_pic_1():
    img = cv2.imread("E:\\BaiduNetdiskDownload\\fk-egp00676_job\\fk-egp00676.b2\\Layer.bmp", cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    result_img = np.zeros((img.shape[0], img.shape[1], 3))

    intdex = (img == 0)
    print("result_img[:, :, 0].shape:", result_img[:, :, 0].shape)
    print("result_img[:, :, 0][intdex].shape:", result_img[:, :, 0][intdex].shape)
    # print(putcolor[0])
    putcolor = np.array([0, 0, 0])  # background
    result_img[:, :, 0][intdex] = putcolor[0]
    result_img[:, :, 1][intdex] = putcolor[1]
    result_img[:, :, 2][intdex] = putcolor[2]


    intdex = (img == 53)  # silk print
    putcolor = np.array([200, 200, 200])
    # result_img[intdex] = putcolor
    result_img[:, :, 0][intdex] = putcolor[0]
    result_img[:, :, 1][intdex] = putcolor[1]
    result_img[:, :, 2][intdex] = putcolor[2]
    bshow = True

    intdex_174 = (img == 174)  # copper
    putcolor = np.array([71, 93, 128])  # np.array([63, 82, 61])
    # result_img[intdex] = putcolor
    result_img[:, :, 0][intdex_174] = putcolor[0]
    result_img[:, :, 1][intdex_174] = putcolor[1]
    result_img[:, :, 2][intdex_174] = putcolor[2]
    bshow = True


    intdex_19 = (img == 19)  # green oil
    putcolor = np.array([39, 47, 40])
    print(putcolor[0])
    # result_img[intdex] = putcolor
    result_img[:, :, 0][intdex_19] = putcolor[0]
    result_img[:, :, 1][intdex_19] = putcolor[1]
    result_img[:, :, 2][intdex_19] = putcolor[2]
    bshow = True
    multi = intdex_19.astype(np.int) * intdex_174.astype(np.int)
    count__multi = np.count_nonzero(multi)
    if count__multi > 0 :
        print("count__multi cross")
    else:
        print("count__multi has no  cross")
        # if bshow:
        #     print(i)
        #     cv2.namedWindow("IMREAD_GRAYSCALE", cv2.WINDOW_NORMAL)
        #     cv2.imshow("IMREAD_GRAYSCALE", tmp)  # 10
        #     cv2.waitKey(0)  # 11
    np.set_printoptions(threshold='nan')
    print(result_img.dtype)
    # result_img = result_img.astype(np.uint8)
    result_img = np.array(result_img, dtype=np.uint8)
    cv2.namedWindow("result_img", cv2.WINDOW_NORMAL)
    cv2.imshow("result_img", result_img)  # 10
    cv2.imwrite("E:\\BaiduNetdiskDownload\\fk-egp00676_job\\111.png", result_img)
    cv2.waitKey(0)  # 11

import struct
def pack_unsigned_int(n):
    return struct.pack('I', n)

def unpack_unsigned_int(fp):
    bin_str = fp.read(4)

    (n,) = struct.unpack('I', bin_str)
    return n

def unpack_str(fp, size):
    bin_str = fp.read(size)
    (ss,) = struct.unpack('{}s'.format(size), bin_str)
    return bytes.decode(ss)


def unpack_float(fp):
    bin_str = fp.read(4)
    (n,) = struct.unpack('f', bin_str)
    return n

def unpack_point(fp):
    bin_str = fp.read(4 * 2)
    (x, y,) = struct.unpack('ii', bin_str)
    return x, y

def unpack_byte(fp):
    bin_str = fp.read(1)
    (n,) = struct.unpack('b', bin_str)
    return n

def unpack_int(fp):
    bin_str = fp.read(4)
    (n,) = struct.unpack('i', bin_str)
    return n

def unpack_unsigned_short(fp):
    bin_str = fp.read(2)
    (n,) = struct.unpack('H', bin_str)
    return n

def unpack_short(fp):
    bin_str = fp.read(2)
    (n,) = struct.unpack('h', bin_str)
    return n

def sythetic_pic_2(binary_img_path, savepath):
    with open(binary_img_path, 'rb') as fp:
        version_i = unpack_int(fp)
        forg_x = unpack_float(fp)
        forg_y = unpack_float(fp)
        width_i = unpack_int(fp)
        height_i = unpack_int(fp)
        fprecision = unpack_float(fp)
        # sss = unpack_int(fp)
        # print("sss", sss)
        fp.seek(0, 2)
        len_int = fp.tell()
        print("len_int", len_int)
        len_int = int((len_int-24)/2)
        print(version_i, forg_x, forg_y, width_i, height_i, fprecision, len_int)
        img = np.zeros((height_i * width_i))
        count = 0
        fp.seek(24, 0)

        for i in range(len_int):
            bin = unpack_short(fp)  # unpack_int(fp)
            #print(bin)
            if bin < 0:
                bin = -bin
                img[count:(count + bin)] = 0
            else:
                img[count: (count + bin)] = 1
            count += bin
        print(count)
        img = np.reshape(img, (height_i, width_i))
        cv2.namedWindow("result_img", cv2.WINDOW_NORMAL)
        cv2.imshow("result_img", 255 * img)  # 10
        img = np.array(img, dtype=np.uint8)
        print(img.shape)
        cv2.imwrite(savepath, 255 * img)
        cv2.waitKey(0)  # 11
    return img, fprecision, forg_x, forg_y


def sythetic_pic_3():
    fprecision = 0.0
    pic_list = []
    for i in range(4):
        binary_img_path = "E:\\doc\\kaimajitai\\Layer_{}.Img".format(i)
        savepath = "E:\\doc\\kaimajitai\\Layer_{}-save.bmp".format(i)
        print(savepath)
        img, fprecision, forg_x, forg_y = sythetic_pic_2(binary_img_path, savepath)
        pic_list.append(img)
    result_img_out = np.zeros((img.shape[0], img.shape[1], 3))
    mask_index_copper = (pic_list[0] == 0)
    putcolor_copper = np.array([71, 93, 128])  # np.array([63, 82, 61])
    # print(result_img_out[mask_index].shape)
    result_img_out[mask_index_copper] = putcolor_copper

    mask_index_oil = (pic_list[2] == 0)
    putcolor_oil = np.array([39, 47, 40])  # np.array([92, 121, 33])  # np.array([39, 47, 40])
    result_img_out[mask_index_oil] = putcolor_oil

    mask_line = mask_index_copper & mask_index_oil
    putcolor = np.array([63, 82, 61]) #  putcolor_copper * 0.2 + putcolor_oil * 0.7 #np.array([63, 82, 61])
    result_img_out[mask_line] = putcolor

    mask_index = (pic_list[1] == 0)
    putcolor = np.array([0, 0, 0])
    result_img_out[mask_index] = putcolor

    mask_index = (pic_list[3] == 0)
    putcolor = np.array([200, 200, 200])
    result_img_out[mask_index] = putcolor

    result_img_out = np.array(result_img_out, dtype=np.uint8)
    cv2.namedWindow("result_img_out", cv2.WINDOW_NORMAL)
    cv2.imshow("result_img_out", result_img_out)  # 10
    cv2.imwrite("E:\\doc\\kaimajitai\\synthetic_3.png", result_img_out)
    cv2.waitKey(0)  # 11

    conf_path = "E:\\BaiduNetdiskDownload\\fk-egp00676_job\\123\\1\\B.vrs"

    with open(conf_path, "r") as ngt_fp:
        ngp_list = ngt_fp.readlines()
    if len(ngp_list) >= 9:
        num_ngt = ngp_list[4]
        num_ngt = int(num_ngt.strip())
    origin_point = (-3942, -1371)
    origin_point = (forg_x*28.5, forg_y*28.5)
    count_save = 0
    for i, line in enumerate(ngp_list):
        if i < 11:
            continue
        count_save += 1
        line = line.strip()
        line_split = line.split(',')
        line_split = list(map(int, line_split))
        print(line_split)
        print(line_split[0]/fprecision/1000, line_split[1]/fprecision/1000)
        point = [(line_split[0]/fprecision/1000 - origin_point[0]), (line_split[1]/fprecision/1000 - origin_point[1])]
        point = list(map(int, point))
        print(point)
        #real_point =
        crop_img = result_img_out[result_img_out.shape[0] - point[1]-60:result_img_out.shape[0] - point[1]+60, point[0]-60:point[0]+60, :]
        #crop_img = result_img[result_img.shape[0] - point[1]:result_img.shape[0] - point[1]+120, point[0]:point[0]+120, :]
        #crop_img = result_img[point[0]:point[0]+120, point[1]:point[1]+120, :]
        print(crop_img.shape)
        save_dir = "E:\\BaiduNetdiskDownload\\fk-egp00676_job\\123\\2_std_save\\B"
        save_path = save_dir+str(count_save) + ".png"
        cv2.imwrite(save_path, crop_img)

if __name__ == "__main__":
    #sythetic_pic()
    #sythetic_pic_1()
    sythetic_pic_3()
    config_1 = configparser.ConfigParser()
    config_1.read()
    #import map
    #whole_pic = np.zeros((32600+1000, 17936+1000, 3), dtype=np.uint8)
    #whole_pic = np.zeros((32600, 17936, 3), dtype=np.uint8)
    whole_pic = np.zeros((10000, 10000, 3), dtype=np.uint8)
    root_dir = "E:\\BaiduNetdiskDownload\\ceshiziliao_20190822\\101245859-MSA-quexiandang\\7--_001\\Panel001\\SideA\\Shot0\\"
    config_file = root_dir + "ResultShot_0.ini"
    conf = configparser.ConfigParser()
    conf.read(config_file)
    defectNum = conf.get('Info', 'DefectNum')
    num = int(defectNum)
    mat = "{:20}\t{:28}\t{:32}"
    print(mat.format(defectNum, defectNum, defectNum))
    print(defectNum)
    dig = defectNum.zfill(4)
    for i in range(num):
        keyhead = "D_"+ str(i).zfill(4)
        print(keyhead)
        defect_roi_cam = conf.get(keyhead, 'DefectRoiCam')
        defect_roi_cam_split = defect_roi_cam.split(',')
        defect_roi_cam_split = list(map(int, defect_roi_cam_split))
        pic_path = root_dir + keyhead + ".jpg"
        print(pic_path)
        img = cv2.imread(pic_path)
        print(img.shape)
        line = conf.get(keyhead, 'NumSubDefect')
        detect_roi_gold = conf.get(keyhead, 'DefectRoiGold')
        detect_roi_gold_split = detect_roi_gold.split(',')
        detect_roi_gold_split = list(map(int, detect_roi_gold_split))
        print("DefectRoiCam:", defect_roi_cam_split)
        print("DefectRoiGold:", detect_roi_gold_split)
        if (detect_roi_gold_split[1]+detect_roi_gold_split[3])<10000 and (detect_roi_gold_split[0]+detect_roi_gold_split[2])<10000:
            whole_pic[detect_roi_gold_split[1]:detect_roi_gold_split[1]+detect_roi_gold_split[3], detect_roi_gold_split[0]:detect_roi_gold_split[0]+detect_roi_gold_split[2], :] = img
        num_line = int(line)
        polygon_list = []
        info_list = []
        bbox_list = []
        for j in range(num_line):
            sub_keyhead = "SD_"+ str(j).zfill(4)
            sub_lineinfo = conf.get(keyhead, sub_keyhead)
            sub_lineinfo = sub_lineinfo.split('|')[0]
            sub_lineinfo = sub_lineinfo.split('#')[-1]
            sub_lineinfo = sub_lineinfo.split(',')

            sub_lineinfo = list(map(float, sub_lineinfo))
            sub_lineinfo = list(map(int, sub_lineinfo))
            point_list = [sub_lineinfo[k:k + 2] for k in range(0, len(sub_lineinfo), 2)]
            polygon_list.append(point_list)
            info_list.append([point_list[1][0]-point_list[0][0], point_list[2][1]-point_list[0][1]])
            bbox_list.append([point_list[0][0], point_list[0][1], point_list[2][0], point_list[2][1]])
        print(polygon_list)
        print(info_list)
        green = (0, 255, 0)
        for z, point_list_tmp in enumerate(polygon_list):
            #cv2.circle(img, (point_list_tmp[0][0], point_list_tmp[0][1]), 3, (255, 0 , 0),thickness=-1)
            cv2.circle(img, (bbox_list[z][0], bbox_list[z][1]), 3, (255, 0 , 0),thickness=-1)
            cv2.circle(img, (bbox_list[z][2], bbox_list[z][3]), 3, (0, 0 , 255),thickness=-1)
            for k, point_tmp in enumerate(point_list_tmp):
                if (k + 1) < len(point_list_tmp):
                    point_tmp_end = point_list_tmp[k + 1]
                    #print(point_tmp[0], point_tmp[1])
                    img = cv2.line(img, (point_tmp[0], point_tmp[1]), (point_tmp_end[0],point_tmp_end[1]), green)

        cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
        cv2.imshow("Canvas", img)  # 10
        cv2.waitKey(0)  # 11

    cv2.namedWindow("whole_pic", cv2.WINDOW_NORMAL)
    cv2.imshow("whole_pic", whole_pic)  # 10
    cv2.waitKey(0)  # 11