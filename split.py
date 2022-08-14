import os

class SplitData(object):
    def __init__(self, path):
        self.path = path
        self.patient_list = []]
        
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

    @process_leaf_file
    def get_patient(self, path):
        if 'txt' in path:
            patient = os.path.dirname(os.path.dirname(os.path.dirname(path)))
            self.patient_list.append(patient)
    
        
    