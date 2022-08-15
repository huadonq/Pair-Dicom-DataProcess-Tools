import os
import argparse
from data_pro import DicomData
from split import SplitData

parser = argparse.ArgumentParser(
        description="convert Dicom Data to 2D img Data and generate YOLO format detection Dataset which label from Pair Tool."
    )
parser.add_argument(
    "--convert_root_path", 
    type=str, 
    help="Directory to Dicom img Data and label with tar files."
)
parser.add_argument(
    "--convert_save_path",
    type=str,
    help="Directory to save images and annotation txt files, no split train and val."
)

parser.add_argument(
    "--split_root_path",
    type=str,
    help="Directory to save train and val images and annotation txt files."
)
parser.add_argument(
    "--split_save_path",
    type=str,
    help="Directory to save train and val images and annotation txt files."
)
parser.add_argument(
    "--ratio",
    type=float,
    default=0.2,
    help="split ratio (test dataset ratio)."
)
parser.add_argument('--convert', action='store_true', default=False)
parser.add_argument('--split', action='store_true', default=False)
args = parser.parse_args()


def main(args):


    if args.convert:

        data = DicomData(args.convert_root_path, args.convert_save_path)
        data.un_tar_file(args.convert_root_path)
        data.sitk_resampleSpacing(args.convert_root_path)
    
    if args.split:

        split = SplitData(args.split_root_path, args.split_save_path)
        split.process_patient()
        train_list, val_list = split.get_train_val(args.ratio)
        split.save_train_val(train_list, 'train')
        split.save_train_val(val_list, 'val')

    




if __name__=='__main__':
    main(args)

