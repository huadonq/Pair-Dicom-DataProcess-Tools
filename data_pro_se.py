import os
import json
import tarfile
import numpy as np
import cv2
import SimpleITK as sitk

class DicomData(object):
    def __init__(self, source_dir, save_dir):
        self.source_dir = source_dir
        self.save_dir = save_dir
        self.file_names = []
        self.dir_names = []
        self.have_processed = []

    def __getitem__(self, idx):
        pass

    def process_leaf_file(func):
        def recursion(self, path, *args):
            lsdir = os.listdir(path)
            dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
            files = [i for i in lsdir if os.path.isfile(os.path.join(path, i))]
            if files:
                for f in files:
                    func(self, os.path.join(path, f), *args)
            if dirs:
                for d in dirs:
                    recursion(self, os.path.join(path, d), *args) 
        return recursion

    def mkdir(self, path):
        os.makedirs(path, exist_ok = True) if not os.path.exists(path) else None

    def source_to_save(self, path):
        return path.replace(self.source_dir, self.save_dir)

    @process_leaf_file
    def get_file_name(self, path):
        self.file_names.append(path)
    
    @process_leaf_file
    def get_sub_dir_name(self, path):
        self.dir_names.append(os.path.dirname(path))

    @process_leaf_file
    def un_tar_file(self, path):
        save_path = os.path.dirname(path)
        if '.tar' in path:
            tar = tarfile.open(path)
            names = tar.getnames()
            for name in names:
                tar.extract(name, save_path)
            tar.close()

    @process_leaf_file
    def json_to_txt(self, path):
        if 'json' in path:
            save_path = os.path.dirname(self.source_to_save(path))
            self.mkdir(save_path)
            with open(path, 'r') as f:
                data = json.load(f)
            w, h = data['FileInfo']['Width'], data['FileInfo']['Height']
            for Frame in data['Models']['BoundingBoxLabelModel']:
                self.process_Frame(Frame, w, h, save_path)

    def process_Frame(self, Frame, w, h, save_path):
        try:
            imgslice = Frame['FrameCount']
            x1,y1 = Frame['p1'][0],Frame['p1'][1]
            x2,y2 = Frame['p2'][0],Frame['p2'][1]
            bbox_w, bbox_h = (x2-x1) / w, (y2-y1) / h
            bbox_cx, bbox_cy = ((x2 + x1) / 2) / w, ((y2 + y1) / 2) / h
            imgcls = 0
            save_path = os.path.join(save_path, str(imgslice) + '.txt')
            with open(save_path,'a+') as file:
                file.write(str(imgcls))
                file.write(' ')
                file.write(str(bbox_cx))
                file.write(' ')
                file.write(str(bbox_cy))
                file.write(' ')
                file.write(str(bbox_w))
                file.write(' ')
                file.write(str(bbox_h))
                file.write('\n')
        except:
            print(f'💥💥💥 error ! path is {save_path}💥💥💥')
            pass
    
    def resampleSpacing(self, sitkImage, newspace=(1,1,1)):
        euler3d = sitk.Euler3DTransform()
        xsize, ysize, zsize = sitkImage.GetSize()
        xspacing, yspacing, zspacing = sitkImage.GetSpacing()
        origin = sitkImage.GetOrigin()
        direction = sitkImage.GetDirection()
        #新的X轴的Size = 旧X轴的Size *（原X轴的Spacing / 新设定的Spacing）
        new_size = (
                    int(xsize * xspacing / newspace[0]),
                    int(ysize * yspacing / newspace[1]),
                    int(zsize * zspacing / newspace[2])
                )
        #如果是对标签进行重采样，模式使用最近邻插值，避免增加不必要的像素值
        sitkImage = sitk.Resample(sitkImage, 
                                new_size,
                                euler3d,
                                sitk.sitkNearestNeighbor,
                                origin,
                                newspace,
                                direction
                            )
        return sitkImage,xspacing, yspacing, zspacing

    @process_leaf_file
    def sitk_resampleSpacing(self, path):

        img_path = os.path.dirname(path)
        if img_path in self.have_processed:
            return
        self.have_processed.append(img_path)

        try:
            reader = sitk.ImageSeriesReader()
            img_names = reader.GetGDCMSeriesFileNames(img_path)
            reader.SetFileNames(img_names)
            image = reader.Execute()
            newimg, xspacing, yspacing, zspacing = self.resampleSpacing(image, newspace=(1,1,1))
            image_array = sitk.GetArrayFromImage(image) # z, y, x
            if image_array is None:
                return
            for file in os.listdir(img_path):
                if 'json' in file:
                    self.read_json(os.path.join(img_path, file),
                                image_array,
                                xspacing, 
                                yspacing, 
                                zspacing
                            )
        except:
            # print(f'💥💥💥 error ! img path is {img_path} 💥💥💥')
            pass

    
    def read_json(self, 
                path, 
                image_array, 
                xspacing, 
                yspacing, 
                zspacing
            ):

        save_root_path = os.path.dirname(self.source_to_save(path))
        
        save_label_path = os.path.join(save_root_path, 'labels')
        self.mkdir(save_label_path)
        # print(save_label_path)
        save_img_path = os.path.join(save_root_path, 'images')
        self.mkdir(save_img_path)

        print(f'processing {save_img_path}...')

        with open(path, 'r') as f:
            data = json.load(f)
        shape = image_array.shape #z y x
        for Frame in data['Models']['BoundingBoxLabelModel']:
            
            if Frame['p1'][0] == Frame['p2'][0]:
                self.read_Frame(Frame, shape[0], shape[1], save_label_path, xspacing, yspacing, zspacing,'zy')
                self.save_index_img(int(Frame['p1'][0] / xspacing), save_img_path, image_array,'zy')
            elif Frame['p1'][1] == Frame['p2'][1]:
                self.read_Frame(Frame, shape[0], shape[2], save_label_path, xspacing, yspacing, zspacing,'zx')
                self.save_index_img(int(Frame['p1'][1] / yspacing), save_img_path, image_array,'zx')
            elif Frame['p1'][2] == Frame['p2'][2]:
                self.read_Frame(Frame, shape[1], shape[2], save_label_path, xspacing, yspacing, zspacing,'xy')
                self.save_index_img(int(Frame['p1'][2] / zspacing), save_img_path, image_array,'xy')


    def read_Frame(self,
                Frame,
                w,
                h,
                save_path,
                xspacing, 
                yspacing,
                zspacing,
                strtype = 'xy'
            ):
    #     try:
    #         imgslice = Frame['FrameCount']
            if strtype == 'xy':
                imgslice = int(Frame['p1'][2] / zspacing)
                x1, y1 = Frame['p1'][0] / xspacing,Frame['p1'][1] / yspacing
                x2, y2 = Frame['p2'][0] / xspacing,Frame['p2'][1] / yspacing
            elif strtype == 'zy':
                imgslice = int(Frame['p1'][0] / xspacing)
                x1, y1 = Frame['p1'][2] / zspacing,Frame['p1'][1] / yspacing
                x2, y2 = Frame['p2'][2] / zspacing,Frame['p2'][1] / yspacing
            else:
                imgslice = int(Frame['p1'][1] / yspacing)
                x1, y1 = Frame['p1'][2] / zspacing,Frame['p1'][0] / xspacing
                x2, y2 = Frame['p2'][2] / zspacing,Frame['p2'][0] / xspacing
            bbox_w, bbox_h = (x2 - x1) / w, (y2 - y1) / h
            bbox_cx, bbox_cy = ((x2 + x1) / 2) / w, ((y2 + y1) / 2) / h
            imgcls = 0
            save_path = os.path.join(save_path, str(imgslice) + '_' + strtype + '_' + '.txt')
            
            with open(save_path, 'a+') as file:
                file.write(str(imgcls))
                file.write(' ')
                file.write(str(bbox_cx))
                file.write(' ')
                file.write(str(bbox_cy))
                file.write(' ')
                file.write(str(bbox_w))
                file.write(' ')
                file.write(str(bbox_h))
                file.write('\n')


    def convert_from_dicom_to_jpg(self, 
                                img, 
                                low_window, 
                                high_window, 
                                save_path):
    
        minmax = np.array([low_window * 1., high_window * 1.])
        newimg = (img - minmax[0]) / (minmax[1] - minmax[0])  #归一化
        newimg = (newimg * 255).astype('uint8')        #将像素值扩展到[0,255]
        cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
      
    def save_index_img(self, 
                index, 
                save_path, 
                image_array, 
                mode = 'xy'):
                
        if mode == 'xy':
            image = image_array[index,:,:]
        elif mode == 'zy':
            image = image_array[:,:,index]
        elif mode == 'zx':
            image = image_array[:,index,:]

        shape = image_array.shape
        w, h = shape[1],shape[2]
        # img_array = np.reshape(image, (w,h))
        high, low = np.max(image), np.min(image)
        
        self.convert_from_dicom_to_jpg(image, 
                                    low, 
                                    high, 
                                    os.path.join(save_path, str(index) + '_' + mode + '_'+'.png')
        )





if __name__=='__main__':
    path = '/zhaojihuai/xiehe_multi_data/bz/PETCT'
    save_path = '/zhaojihuai/xiehe_multi_data/hello'
    data = DicomData(path, save_path)
    data.un_tar_file(path)
    data.sitk_resampleSpacing(path)
    
    
