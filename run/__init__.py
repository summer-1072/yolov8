import cv2
import math
import torch
import numpy as np
from util import color
import matplotlib.pyplot as plt

x1 = [round(x / 1e6) for x in [11129454, 25845550, 43614318, 68133198]]
p1 = [round(x * 1e2, 1) for x in [0.3842, 0.4021, 0.4085, 0.4098]]
s1 = [round(x * 1e3, 1) for x in [0.00577, 0.00611, 0.00858, 0.0101]]

x2 = [round(x / 1e6) for x in [19082040, 58248408, 72660920, 113512792]]
p2 = [round(x * 1e2, 1) for x in [0.3946, 0.4086, 0.4149, 0.4183]]
s2 = [round(x * 1e3, 1) for x in [0.0036, 0.00698, 0.018, 0.0194]]

plt.figure(figsize=(15, 6))

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.25, wspace=0.25)
plt.subplot(1, 2, 1)
plt.xlabel('Parameters(M)')
plt.ylabel('BDD100K mAP50-95(val)')

plt.plot(x1, p1, label='YOLOv8')
plt.plot(x2, p2, label='YOLOBI')
plt.scatter(x1, p1)
plt.scatter(x2, p2)
plt.legend()


plt.text(x1[0] - 1, p1[0] + 0.05, 's', color='steelblue')
plt.text(x1[1] - 1, p1[1] + 0.05, 'm', color='steelblue')
plt.text(x1[2] - 1, p1[2] + 0.05, 'l', color='steelblue')
plt.text(x1[3] - 1, p1[3] + 0.05, 'x', color='steelblue')

plt.text(x2[0] - 1, p2[0] + 0.05, 's', color='orange')
plt.text(x2[1] - 1, p2[1] + 0.05, 'm', color='orange')
plt.text(x2[2] - 1, p2[2] + 0.05, 'l', color='orange')
plt.text(x2[3] - 1, p2[3] + 0.05, 'x', color='orange')
plt.grid(True, linestyle="--", alpha=0.5)


plt.subplot(1, 2, 2)
plt.xlabel('Latency A100 NoTensorRT FP32(ms/img)')
plt.ylabel('BDD100K mAP50-95(val)')

plt.plot(s1, p1, label='YOLOv8')
plt.plot(s2, p2, label='YOLOBI')
plt.scatter(s1, p1)
plt.scatter(s2, p2)
plt.legend()


plt.text(s1[0] - 0.25, p1[0] + 0.05, 's', color='steelblue')
plt.text(s1[1] - 0.25, p1[1] + 0.05, 'm', color='steelblue')
plt.text(s1[2] - 0.25, p1[2] + 0.05, 'l', color='steelblue')
plt.text(s1[3] - 0.25, p1[3] + 0.05, 'x', color='steelblue')

plt.text(s2[0] - 0.25, p2[0] + 0.05, 's', color='orange')
plt.text(s2[1] - 0.25, p2[1] + 0.05, 'm', color='orange')
plt.text(s2[2] - 0.25, p2[2] + 0.05, 'l', color='orange')
plt.text(s2[3] - 0.25, p2[3] + 0.05, 'x', color='orange')
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()