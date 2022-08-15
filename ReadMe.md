

# ReadMe
**This repository only support Linux OS.**

**This repository is used for from Dicom format detection dataset (label generate from Pair) convert to YOLO dataset format.**

**And this repository will auto split train and val dataset.**

**environments:**
ubuntu1~18.04, 4 * Tesla V100, Python Version:3.8.10, CUDA Version:11.4

Please make sure your Python version>=3.7.
**Use pip or conda to install those Packages:**

```
numpy
opencv-python
sklearn
SimpleITK
tarfile
argparse
```

# Usage
You just need to change the original Dicom data path and the save path in run.sh, then "sh run.sh" in the TERMINAL.

the "run.sh" contains:

if you want only split processed dataset, please add "--split" param.

```
python main.py --split --split_root_path hello_2/zhongliu/PET-CT --split_save_path yolov5-master/second/zhongliu/PETCT
```

the **"split_root_path"** is your processed dataset path, and i think you know what the **"split_save_path"**, it's the save path, this folder contains YOLO format data struct.

And if you want only convert Dicom dataset to 2D img datdaset, please add "--convert" param.

```
python main.py --convert --convert_root_path xiehe_multi_data/second -- convert_save_path xiehe_multi_data/second_processed
```

the **"convert_root_path"** is your origin Dicom dataset path, and i think you know what the **"convert_save_path"**, it's the save path, this folder contains origin data struct, but every last sub_dir have images and labels two subdir, contains image and txt files.

And you just "sh run.sh" in the TERMINAL.

```
sh run.sh
```
If you want to set train dataset and val dataset split ratio, please mod it in main.py, or add args in "run.sh".


# Prepare datasets
No matter how many subdir in your data root dir, you just need to make sure that the images and json files in each leaf subdir are placed together,
it's no problem even if there are subdir nested.


you need prepare labelme-format dataset and make sure the each leaf sub folder architecture as follows:
```
data
|
|----->sub1
|--------->sub2
|----------------->sub3
|---------------------------xxx.dcm
|---------------------------xxx.tar
...
|---------------------------xxx.dcm
|---------------------------xxx.tar
|----------------->sub4
|---------------------------xxx.dcm
|---------------------------xxx.tar
...
|---------------------------xxx.dcm
|---------------------------xxx.tar
|----->sub5
|--------->sub6
|----------------->xxx.dcm
|----------------->xxx.tar
...
|----------------->xxx.dcm
|----------------->xxx.tar
```

# Visual
If you want to visual YOLO-format annotations in single image, please replace single_visual.py's relative path.

Or you want to visual YOLO-format annotations in batch image, please replace batch_visual.py's relative path.

🙌To be honest, it is only support real annotations processing, I'll code for pred annotations quickly in the future. 🙌

# Citation
**If you find my work useful in your research, please consider citing:**

```
@inproceedings{zjh,
 title={Pair-Dicom-DataProcess-Tools},
 author={zjh},
 year={2022}
}
```