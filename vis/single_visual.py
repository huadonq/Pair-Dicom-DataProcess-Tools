import os
import cv2

img_path = ''
txt_path= ''
save_path = 'demo.png'

point_color = (0, 255, 0) # BGR
thickness = 1
lineType = 8

plot_img = cv2.imread(img_path)
assert plot_img is not None
H, W, C = plot_img.shape

with open(txt_path,'r') as file:
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
cv2.imwrite(save_path, plot_img)