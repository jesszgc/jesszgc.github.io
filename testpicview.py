# coding: utf-8
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import mobilenet_v1
import time
import numpy as np
import os.path as osp
# from skimage import io
# from mpl_toolkits.mplot3d import Axes3D
# import scipy.io as sio

from ddfa_utils import reconstruct_vertex

arch = 'mobilenet_1'
device_ids = [0]
checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'

num_classes=62

map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
torch.cuda.set_device(device_ids[0])
model = getattr(mobilenet_v1, arch)(num_classes=num_classes)
model = nn.DataParallel(model, device_ids=device_ids).cuda()
model.load_state_dict(checkpoint)
#cv.namedWindow('hello')
cv.namedWindow('shape')

cudnn.benchmark = True
model.eval()

end = time.time()
outputs = []
#cap=cv.VideoCapture('rtsp://admin:admin@192.168.2.180/cam/realmonitor?channel=1&subtype=0')
cap=cv.VideoCapture(0)
if cap.isOpened()!=False:
    while 27!=cv.waitKey(5):
        dd,frame=cap.read()
#        frame=cv.imread('')
        # read()
        frame=cv.resize(frame,(120,120))
#        cv.imshow('hello', frame)
#        cv.waitKey(0)
        imglist = [frame,cv.flip(frame,1)]
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,120,120))
            imglist[i] = (imglist[i]-127.5)/128.0

        img = np.vstack(imglist)
        inputs = torch.from_numpy(img).float()
        with torch.no_grad():
            inputs = inputs.cuda()
            output = model(inputs)
            # for i in range(output.shape[0]):
            param_prediction = output[0].cpu().numpy().flatten()
            #
            # outputs.append(param_prediction)
            outputs = np.array(param_prediction, dtype=np.float32)
            lms = reconstruct_vertex(outputs, dense=False)
            # print(lms)
#            print([lms[0, 1], lms[1, 1]])
            for i in range(68):
                cv.circle(frame, (lms[0, i], lms[1, i]), 2, (255, 255, 0))
                # print([lms[0, i], lms[1, i]])
#            cv.imwrite('res/AFLW-2000-3D/1.jpg', frame)

            # draw_landmarks(frame,outputs)
        cv.imshow('shape',frame)

else:
    print('open cam fail')
