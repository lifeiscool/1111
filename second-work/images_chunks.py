import os
import cv2
import numpy as np
import random


def get_chunks(c_num, isfineturn, img_path, gt_path, save_path, save_gt_path):
    for filename in os.listdir(img_path):
        print(img_path+filename)
        # img1 = cv2.imread(img_path+filename,1)
        # gt1 = cv2.imread(gt_path+filename[:-4]+"_gt.bmp")
        img1 = cv2.imdecode(np.fromfile(img_path+filename, dtype=np.uint8), 1)
        gt1 = cv2.imdecode(np.fromfile(gt_path+filename[:-4]+"_gt.bmp", dtype=np.uint8), 1)
        [h,w,c] = img1.shape
        each_width = h//c_num
        each_height = w//c_num
        for i in range(c_num):
            for j in range(c_num):
                temp_chunk = img1[each_height * i:each_height * (i + 1), each_width * j:each_width * (j + 1), :]
                temp_chunk_gt = gt1[each_height * i:each_height * (i + 1), each_width * j:each_width * (j + 1), :]
                cv2.imencode('.bmp', temp_chunk)[1].tofile(save_path + filename[:-4]+"_row"+str(i)+"_col"+str(j)+".bmp")
                cv2.imencode('.bmp', temp_chunk_gt)[1].tofile(save_gt_path + filename[:-4] + "_row" + str(i) + "_col" + str(j) + "_gt.bmp")
        if(isfineturn):
            half_w = each_width//2
            half_h = each_height // 2
            for i in range(c_num-1):
                for j in range(c_num-1):
                    temp_chunk = img1[half_h+each_height*i:half_h+each_height*(i+1),half_w+each_width*j:half_w+each_width*(j+1),:]
                    temp_chunk_gt = gt1[half_h+each_height*i:half_h+each_height*(i+1),half_w+each_width*j:half_w+each_width*(j+1),:]
                    cv2.imencode('.bmp', temp_chunk)[1].tofile(save_path + filename[:-4]+"_row"+str(i)+"_col"+str(j)+"_fine"+".bmp")
                    cv2.imencode('.bmp', temp_chunk_gt)[1].tofile(save_gt_path + filename[:-4] + "_row" + str(i) + "_col" + str(j) + "_gt_fine.bmp")



def get_montage(c_num, fun, img_path,save_path):
    last_name = ""
    for filename in os.listdir(img_path):
        if("fine" not in filename):
            i_name = filename.split("_")[0]
            i_row = int(filename.split("_")[1].split("row")[1])
            i_col = int(filename.split("_")[2].split(".")[0].split("col")[1])
            img1 = cv2.imread(img_path + filename, 1)
            [h, w, c] = img1.shape
            if last_name != i_name:
                print(filename)
                montage = np.zeros((h*c_num, w*c_num, c), np.uint8)
                last_name = i_name
            montage[i_row*h:(i_row+1)*h,i_col*w:(i_col+1)*w,:] = img1
            cv2.imwrite(save_path + i_name + ".bmp", montage)

    half_h = h//2
    half_w = w//2
    temp_im = np.zeros((h, w, c, 2))
    for filename in os.listdir(img_path):
        if("fine" in filename):
            i_name = filename.split("_")[0]
            i_row = int(filename.split("_")[1].split("row")[1])
            i_col = int(filename.split("_")[2].split(".")[0].split("col")[1])
            # img1 = cv2.imread(img_path + filename, 1)
            image = cv2.imdecode(np.fromfile(save_path + i_name + ".bmp", dtype=np.uint8), 1)
            temp_im[:,:,:,0] = image[half_h + i_row * h:half_h + (i_row + 1) * h, half_w + i_col * w:half_w + (i_col + 1) * w,:]
            temp_im[:, :, :, 1] = cv2.imdecode(np.fromfile(img_path + filename, dtype=np.uint8), 1)
            if (fun == 'ave'):
                image[half_h + i_row * h:half_h + (i_row + 1) * h, half_w + i_col * w:half_w + (i_col + 1) * w,
                :] = temp_im.mean(axis=3)
            if (fun == 'max'):
                image[half_h + i_row * h:half_h + (i_row + 1) * h, half_w + i_col * w:half_w + (i_col + 1) * w,
                :] = temp_im.max(axis=3)
            if (fun == 'min'):
                image[half_h + i_row * h:half_h + (i_row + 1) * h, half_w + i_col * w:half_w + (i_col + 1) * w,
                :] = temp_im.min(axis=3)
            cv2.imwrite(save_path + i_name + ".bmp", image)


# def get_preview(c_num,img_path,save_path):
#     for filename in os.listdir(img_path):
#         print(img_path+filename)
#         # img1 = cv2.imread(img_path+filename,1)
#         img1 = cv2.imdecode(np.fromfile(img_path+filename, dtype=np.uint8), 1)
#         [h,w,c] = img1.shape
#         each_width = h//c_num
#         each_height = w//c_num
#         for i in range(1,c_num):
#             img1[each_height * i:each_height * i + 1, :, 0] = 255
#             img1[each_height * i:each_height * i + 1, :, 1] = 150
#             img1[each_height * i:each_height * i + 1, :, 2] = 255
#         for j in range(1,c_num):
#             img1[:, each_width * j:each_width * j + 1, 0] = 255
#             img1[:, each_width * j:each_width * j + 1, 1] = 150
#             img1[:, each_width * j:each_width * j + 1, 2] = 255

        # cv2.imwrite(save_path + filename,img1)
        cv2.imencode('.bmp', img1)[1].tofile(save_path + filename)


def get_preview(c_num,img_path,save_path):
    for filename in os.listdir(img_path):
        print(img_path+filename)
        # img1 = cv2.imread(img_path+filename,1)
        img1 = cv2.imdecode(np.fromfile(img_path+filename, dtype=np.uint8), 1)
        [h,w,c] = img1.shape
        each_width = h//c_num
        each_height = w//c_num
        for i in range(1,c_num):
            img1[each_height * i:each_height * i + 1, :, 0] = 84
            img1[each_height * i:each_height * i + 1, :, 1] = 84
            img1[each_height * i:each_height * i + 1, :, 2] = 150
        for j in range(1,c_num):
            img1[:, each_width * j:each_width * j + 1, 0] = 84
            img1[:, each_width * j:each_width * j + 1, 1] = 84
            img1[:, each_width * j:each_width * j + 1, 2] = 150

        # cv2.imwrite(save_path + filename,img1)
        cv2.imencode('.bmp', img1)[1].tofile(save_path + filename)


def get_dir(img_path,train_path,test_path):

    with open(train_path, 'w') as train:
        with open(test_path, 'w') as test:
            imgs = os.listdir(img_path)
            imgs_train = os.listdir(img_path)
            resultList = random.sample(range(0, len(imgs)), int(len(imgs)*0.3));
            for n in resultList:
                test.write(imgs[n][:-4] + "\n")
                imgs_train.remove(imgs[n])
            for name in imgs_train:
                train.write(name[:-4]+"\n")

def get_val(img_path):

    '''
        D:\数据_轮胎\实验数据\val\val_bubble
        pics = ['s101102006','s101104027','s101104034','s101105041','s101105042','s111201063','s111202071','s111204081','s111204085','s111207101'
        ,'s111208104','s111208106','s111231015','s120101017','s120103030','s120103041','s120105001','s120105014','s120106036','s120106037'
        ,'s120106040','s120106044','s120106062','s120107074','s120107082','s120107087','s120107090','s120108105','s120109140','s120109141'
        ,'s120109145','s120110164','s120110166','s120110169','s120110172','s120110174','s120111177','s120111183','s120111191','s120111196'
        ,'s120112216','s120112217','s120113010','s120113011','s120113224','s120114020','s120114024','s120114027','s120114030','s120114036'
        ,'s120115042','s120115049','s120115051','s120115064','s120115068','s120115074','s120116078','s120116100','s120116103','s120116121'
        ,'s120118165','s120118167','s120118233','s120120260','s120120265','s120123315','s120123321','s120124327','s120124329','s120125336'
        ,'s120125337','s120126340','s120127355','s120128178','s120128181','s120129192','s120130214','s120130215','s120130218','s120130219'
        ,'s120131234','s120131235','s120131237','s120131246','s120201255','s120201263','s120202294','s120202297','s120202302','s120203321'
        ,'s120204081','s120204341','s120208189','s120212126','s120212135','s120213114','s120213118','s120213132','s120213141','s120214144'
        ,'s120214155','s120214160','s120214168','s120214169','s120214170','s120214172','s120214174','s120214176','s120214182','s120215194'
        ,'s120216025','s120216027','s120217034','s120217049','s120217053','s120217055','s120217059','s120218076','s120219113','s120219127'
        ,'s120220141','s120220145','s120221213','s120221215','s120221233','s120221237','s120221238','s120222249','s120222253','s120222257'
        ,'s120222261','s120223152','s120223159','s120225162','s120225199','s120225201','s120225206','s120226164','s120226173']

        D:\数据_轮胎\实验数据\val\val_impurity
        pics = ['s101102002','s101102005','s101102009','s101102010','s101103018','s101104025','s101104026','s101104033','s101104035','s101104036'
        ,'s101104037','s101104039','s101105051','s101105052','s111130061','s111202068','s111202070','s111204082','s111231010','s120103039'
        ,'s120104060','s120104063','s120104070','s120104072','s120105005','s120105007','s120105008','s120105011','s120105012','s120105013'
        ,'s120105015','s120105020','s120105024','s120105025','s120105026','s120105028','s120105030','s120106043','s120106056','s120107078'
        ,'s120107083','s120107084','s120107092','s120108104','s120108113','s120108120','s120109123','s120109124','s120109125','s120109128'
        ,'s120109129','s120109131','s120109137','s120109142','s120110146','s120110153','s120110155','s120110159','s120111187','s120111188'
        ,'s120112206','s120112207','s120112215','s120114035','s120115052','s120115058','s120115060','s120115065','s120116096','s120116112'
        ,'s120116125','s120116127','s120117140','s120117146','s120118168','s120118175','s120118231','s120119241','s120119242','s120120251'
        ,'s120120268','s120121277','s120121282','s120121284','s120123324','s120127343','s120127348','s120127350','s120127352','s120127354'
        ,'s120127356','s120129198','s120130209','s120130210','s120130232','s120131240','s120131247','s120131249','s120131251','s120201256'
        ,'s120201260','s120201267','s120201268','s120201270','s120201272','s120201281','s120201284','s120202286','s120202287','s120202292'
        ,'s120202301','s120203327','s120204342','s120204346','s120204347','s120206138','s120210092','s120210094','s120210096','s120210098'
        ,'s120211104','s120211109','s120212112','s120212131','s120212132','s120212144','s120212148','s120212152','s120213119','s120215188'
        ,'s120215193','s120215197','s120216012','s120216021','s120216022','s120217041','s120217045','s120217051','s120217056','s120218067'
        ,'s120218070','s120218079','s120218093','s120219099','s120219103','s120219108','s120219116','s120219118','s120219120','s120219128'
        ,'s120219131','s120220137','s120220204','s120221212','s120222260','s120223154','s120223163','s120223274','s120224172','s120224175'
        ,'s120224182','s120224183','s120224184','s120224186','s120225160','s120225205','s120225215','s120226171','s120226188','s120226189'
        ,'s120226192','s120226194','s120227213','s120228234','s120229281']
    '''




    val_str = "pics = ["
    for ii, filename in enumerate(os.listdir(img_path)):
        if((ii+1)%10 == 0):
            val_str = val_str + "'" + filename[:-4] + "'\n,"
        else:
            val_str = val_str +  "'"+filename[:-4] + "',"
    val_str = val_str[:-1] + "]"
    print(val_str)


if __name__ == "__main__":

    # test:

    c_num = 4
    isfineturn = False
    img_path = "test_images/"
    gt_path = "test_gt/"
    chunks_save = "test_save/"
    chunks_gt_save = "test_save_gt/"
    montage_save = "test_save2/"
    montage_gt_save = "test_save2_gt/"
    preview_save = "test_save3/"
    preview_gt_save = "test_save4/"
    get_chunks(c_num, isfineturn, img_path, gt_path, chunks_save, chunks_gt_save)
    # get_montage(c_num ,'ave',chunks_save, montage_save)

    # get_train:
    # c_num = 4
    # img_path = "D:/数据_轮胎/实验数据/images256/"
    # gt_path = "D:/数据_轮胎/实验数据/gt256/"
    # chunks_save = "D:/数据_轮胎/实验数据/chunk/"
    # chunks_gt_save = "D:/数据_轮胎/实验数据/chunk-gt/"
    # preview_save = "D:/数据_轮胎/实验数据/chunk-preview/"

    # get_chunks(c_num, img_path, gt_path, chunks_save, chunks_gt_save)
    # get_montage( c_num, chunks_save, montage_save)
    # get_preview(c_num, gt_path, preview_gt_save)


    # txt operation:
    # get_dir("D:/数据_轮胎/实验数据/chunk","D:/数据_轮胎/实验数据/train_chunk.txt","D:/数据_轮胎/实验数据/test_chunk.txt")
    # get_val(r"D:\数据_轮胎\实验数据\val\val_impurity")

