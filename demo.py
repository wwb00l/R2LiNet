import torch
import numpy as np
import time
import onnxruntime
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fx = 1495.468642
fy = 1495.468642
cx = 961.272442
cy = 624.89592
projection_matrix2 = np.array([[-0.013857  ,  -0.9997468,  0.01772762 , 0.05283124],
                               [ 0.10934269, -0.01913807, -0.99381983 , 0.98100483],
                               [ 0.99390751, -0.01183297,  0.1095802  , 1.44445002],
                               [ 0.        ,  0.        ,  0.         , 1.        ]])
                               
upsample_lidar = torch.nn.Upsample(size=1024, mode='nearest')
upsample_radar = torch.nn.Upsample(size=512, mode='nearest')
lidar = np.array(pd.read_hdf("00000.hdf",'lidar')).reshape((-1, 4)).T
radar = np.array(pd.read_hdf("00000.hdf",'radar')).reshape((-1, 7)).T
lidar = torch.from_numpy(lidar).float()
radar = torch.from_numpy(radar).float()
lidar = upsample_lidar(torch.unsqueeze(lidar,dim=0)).squeeze().T
radar = upsample_radar(torch.unsqueeze(radar,dim=0)).squeeze().T
lidar = lidar[:,:3].numpy()
radar = radar[:,:6].reshape(1,512,6).numpy()/1935
sess = onnxruntime.InferenceSession('enhance.onnx',providers=['CPUExecutionProvider'])
pred = sess.run([], {'input': radar})[0][0]*1935


x = radar[0,:,:3]*1935
pred = np.concatenate((pred,x),axis =0)
projection_matrix = np.linalg.inv(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]))
depth_lidar = lidar[:, 2].reshape(-1,1)
lidar = np.hstack((lidar[:, :2],np.ones((lidar.shape[0], 1),dtype=np.float32)))
lidar = depth_lidar * (lidar @ projection_matrix.T)
depth = pred[:, 2].reshape(-1,1)
pred = np.hstack((pred[:, :2],np.ones((pred.shape[0], 1),dtype=np.float32)))
pred = depth * (pred @ projection_matrix.T)
projection_matrix = projection_matrix2[:3,:3]
pred = pred[:,:3].dot(projection_matrix)
projection_matrix = projection_matrix2[:3,:3]
lidar = lidar[:,:3].dot(projection_matrix)
fig = plt.figure(figsize=(19, 12),dpi=120)
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(pred[512:,0],pred[512:,1],pred[512:,2],s = 2,edgecolor="black",marker=".")
ax1.set_title('Radar', fontsize=10)
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(pred[:,0],pred[:,1],pred[:,2],s = 2,edgecolor="black",marker=".")
ax2.set_title('Radar Enhancement', fontsize=10)
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(lidar[:,0],lidar[:,1],lidar[:,2],s = 2,edgecolor="black",marker=".")
ax3.set_title('Lidar', fontsize=10)
ax1.set_ylim(400,1200)
ax2.set_ylim(400,1200)
ax3.set_ylim(400,1200)
ax1.set_xlim(0,100)
ax2.set_xlim(0,100)
ax3.set_xlim(0,100)
ax1.set_ylim(-30,30)
ax2.set_ylim(-30,30)
ax3.set_ylim(-30,30)
plt.show()  
