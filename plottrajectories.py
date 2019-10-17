import numpy as np
import matplotlib
from scipy.ndimage import rotate, zoom
import math
import matplotlib.pyplot as plt
from matplotlib.image import BboxImage, AxesImage
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.widgets import Slider, Button
import streamlit as st
import torch
import pickle


CAR = {
    color: zoom(plt.imread('img/car-{}.png'.format(color)), [0.3, 0.3, 1.])
    for color in ['gray', 'orange', 'purple', 'red', 'white', 'yellow']
}

color = ['gray', 'orange', 'purple', 'red', 'yellow','white']
scale = 5./max(CAR['gray'].shape[:2])

def set_image(obj, data, scale, x=[0., 0., 0.]):
    ox = x[0]
    oy = x[1]
    angle = x[2]
    img = rotate(data, np.rad2deg(angle))
    h, w = img.shape[0], img.shape[1]
    obj.set_data(img)
    obj.set_extent([ox-scale*w*0.5, ox+scale*w*0.5, oy-scale*h*0.5, oy+scale*h*0.5])

def load_data():
    d = torch.load('short.pt')
    sc = d[0]
    gt = d[1]
    return sc,gt
st.subheader('Generated path in dotted')
fig, ax = plt.subplots()

if st.checkbox('load data'):
    sc,gt = load_data()
    rect = patches.Rectangle((-35,-15),70,30,linewidth=1,edgecolor='none',facecolor='gray')
    ax.add_patch(rect)
    n = st.selectbox('iteration#',range(0,len(sc)-1),1)
    sc = sc[n]
    gt = gt[n]
    sc = sc.permute(1,0,2)
    gt = gt.permute(1,0,2)
    angle = np.zeros((gt.size(0),gt.size(1)))
    sc_np = sc.cpu().detach().numpy()
    gt_np = gt.cpu().detach().numpy()
    seq_num = st.selectbox('scene number',range(0,sc.size(0)-4),1)
    c = 0
    for i in range(seq_num,seq_num+4):
        pos_x = sc_np[i,:,0]
        pos_y = sc_np[i,:,1]
        den = pos_x[1:] - pos_x[:-1]
        num = pos_y[1:] - pos_y[:-1]
        angle[i,1:] = num/den
        angle[i,:] = [0. if math.isnan(x) else x for x in angle[i,:]]
        ax.plot(sc[i,:,0].cpu().detach().numpy(),sc[i,:,1].cpu().detach().numpy(), color=color[c],linestyle='--')
        ax.plot(gt[i,:,0].cpu().detach().numpy(),gt[i,:,1].cpu().detach().numpy(),color=color[c])
        c += 1
    angle = angle.reshape(gt.size(0),gt.size(1),1)
    x_vec = np.concatenate((sc_np[seq_num:seq_num+4],angle[seq_num:seq_num+4]),axis=2)
    t = np.arange(x_vec.shape[1])
    t = st.slider('Time',0,x_vec.shape[1]-1,0)
    cars = [AxesImage(ax, interpolation='bicubic', zorder=100) for _ in np.arange(x_vec.shape[0])]
    c = 0
    for car in cars:
        ax.add_artist(car)
        set_image(car,CAR[color[c]],scale,x_vec[c,t,:])
        c += 1
plt.ylabel('Meters')
plt.xlabel('Meters')
ax.set_xlim(-35., 35.)
ax.set_ylim(-35., 35.)
st.write(fig)
