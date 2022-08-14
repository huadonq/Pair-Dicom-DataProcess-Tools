import os
import cv2

root_path = ''
save_path = ''

os.makedirs(save_path, exist_ok=True)

path = os.path.join(root_path, 'labels')
img_path = os.path.join(root_path, 'images')
point_color = (0, 255, 0) # BGR
thickness = 1
lineType = 8
img_hz = '.png'

for label in os.listdir(path):
    if 'ipy' in label:
        continue
    plot_img = cv2.imread(os.path.join(img_path, label.split('.')[0] + img_hz))
    if plot_img is None:
        continue
    H, W, C = plot_img.shape
    with open(os.path.join(path,label),'r') as file:
        for line in file.readlines():
            l = line.split('\n')[0]
            x = float(l.split(' ')[1])
            y = float(l.split(' ')[2])
            w = float(l.split(' ')[3])
            h = float(l.split(' ')[4])
            
            x1=int((x-w/2)*W)                                                                   #坐标转换
            y1=int((y-h/2)*H)
            x2=int((x+w/2)*W)
            y2=int((y+h/2)*H)
            w=w*W
            h=h*H
            cv2.rectangle(plot_img, (x1, y1), (x2, y2),  point_color, thickness, lineType)
    cv2.imwrite(os.path.join(save_path, label.split('.')[0] + img_hz), plot_img)