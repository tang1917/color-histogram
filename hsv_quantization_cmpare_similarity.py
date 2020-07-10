import cv2
import numpy as np
import glob
import os


def get_hist(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h = hsv_img[...,0]
    s = hsv_img[...,1]
    v = hsv_img[...,2]
    height,width = h.shape
    h[(2*h>345)|(2*h<=15)] = 0
    h[(2*h>15)&(2*h<=25)] = 1
    h[(2*h>25)&(2*h<=45)] = 2
    h[(2*h>45)&(2*h<=55)] = 3
    h[(2*h>55)&2*(h<=80)] = 4
    h[(2*h>80)&(2*h<=108)] = 5
    h[(2*h>108)&(2*h<=140)] = 6
    h[(2*h>140)&(2*h<=165)] = 7
    h[(2*h>165)&(2*h<=190)] = 8
    h[(2*h>190)&(2*h<=220)] = 9
    h[(2*h>220)&(2*h<=255)] = 10
    h[(2*h>255)&(2*h<=275)] = 11
    h[(2*h>275)&(2*h<=290)] = 12
    h[(2*h>290)&(2*h<=316)] = 13
    h[(2*h>316)&(2*h<=330)] = 14
    h[(2*h>330)&(2*h<=345)] = 15


    s[((s/255)>0) & ((s/255)<= 0.15)] = 0
    s[((s/255)>0.15)&((s/255)<=0.4)] =1
    s[((s/255)>0.4)&((s/255)<=0.75)] = 2
    s[((s/255)>0.75)&((s/255)<=1)] = 3

    v[((v/255)>0) & ((v/255)<= 0.15)] = 0
    v[((v/255)>0.15)&((v/255)<=0.4)] =1
    v[((s/255)>0.4)&((v/255)<=0.75)] = 2
    v[((s/255)>0.75)&((v/255)<=1)] = 3
    g = 16*h +4*s + v
    hist = []
    #print(np.min(g),np.max(g))
    for i in range(256):
        a = (g==i).sum()/(height*width)
        hist.append(a)
    return np.array(hist)
def generate_score(tem_hist,img_hist):
    com = (tem_hist<img_hist)
    score = sum(tem_hist[com]) + sum(img_hist[~com])
    return score
def get_color_score(img,tem_path):
    tem_hist = np.loadtxt(tem_path,delimiter=',')
    img_hist = get_hist(img)
    score = generate_score(tem_hist,img_hist)
    return score
