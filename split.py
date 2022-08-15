import os
from sklearn.model_selection import train_test_split
import shutil
class SplitData(object):
    def __init__(self, path, save_path):
        self.path = path
        self.save_path = save_path
        self.patient_list = []
        self.img_hz = ['png', 'PNG', 'jpg', 'JPEG', 'tiff']
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
        return path.replace(self.path, self.save_path)

    def process_patient(self):
        for patient in os.listdir(self.path):
            path = os.path.join(self.path, patient)
            # save_path = self.source_to_save(path)
            self.patient_list.append(path)
    
    def get_train_val(self, ratio = 0.2):
        train, valid = train_test_split(
                self.patient_list, test_size = ratio)
        return train, valid
    
    def get_save_str(self, path):
        file_name = path.split('/')[-1]
        strs = path.replace(self.path, '').replace(file_name, '')
        result = ''
        for s in strs.split('/'):
            result += s
        return file_name, result

    @process_leaf_file
    def get_leaf_file(self, path, TYPE):
        file_name, patient_id = self.get_save_str(path)
        if file_name.split('.')[1] in self.img_hz:
            self.mkdir(os.path.join(self.save_path, TYPE, 'images'))
            shutil.copy(path, os.path.join(self.save_path, TYPE, 'images', patient_id + '_' +file_name))
        elif 'txt' in file_name:
            self.mkdir(os.path.join(self.save_path, TYPE, 'labels'))
            shutil.copy(path, os.path.join(self.save_path, TYPE, 'labels', patient_id + '_' +file_name))


    def save_train_val(self, path_list, TYPE):
        for patient in path_list:
            self.get_leaf_file(patient,TYPE)


        
    