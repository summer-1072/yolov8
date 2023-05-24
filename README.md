<img alt="yolov8.png" src="assert%2Fyolov8.png" width="1500"/>

Model: Ultralytics YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility.
Now, We have rewritten the YOLOv8 to make it simpler and easier to use. At the same time, we have tested the effectiveness of YOLOv8 in object detection for autonomous driving on the BDD100K dataset.
Finally, we designed the YOLOBI network to try to optimize the effectiveness of object detection.

<img alt="performance.png" src="assert%2Fperformance.png" width="1500"/>

Dataset: We released the largest and most diverse driving video dataset with rich annotations called BDD100K. You can access the data for research now at http://bdd-data.berkeley.edu. We have recently released an arXiv report on it. And there is still time to participate in our CVPR 2018 challenges!

<img alt="BDD100K.png" src="assert%2Fbdd100k.png" width="1500"/>

Show: We have presented some of the results of model inference to observe the detection effect in actual driving roads.

<img alt="bdd100k.gif" src="assert%2Fbdd100k.gif" width="1280"/>

Weight: We have provided pre training weights for everyone to use. 

| Model    | size(w * h) | mAP50-95 val | Speed A100 FP16(ms) | params(M) | FLOPs(G) |
|----------|-------------|--------------|---------------------|-----------|----------|
| YOLOv8s  | 960 * 544   | 38.4         | 2.5                 | 11.1      | 36.3     |
| YOLOv8m  | 960 * 544   | 40.2         | 3.5                 | 25.8      | 100.4    |
| YOLOv8l  | 960 * 544   | 40.9         | 4.8                 | 43.6      | 210.2    |
| YOLOv8x  | 960 * 544   | 41.0         | 6.5                 | 68.1      | 328.2    |
| YOLOBis  | 1280 * 704  | 39.5         | 3.5                 | 19.1      | 65.1     |
| YOLOBim  | 1280 * 704  | 40.9         | 5.8                 | 58.2      | 197.3    |
| YOLOBil  | 1280 * 704  | 41.5         | 7.5                 | 72.7      | 329.0    |
| YOLOBix  | 1280 * 704  | 42.0         | 11.1                | 113.5     | 513.7    |


Requirement: We suggest using Python =3.8, torch >=1.10.0, cuda >=11.3.

Use: You can start executing the program from the run folder, which contains YOLOv8 training, verification, and inference functions.
For example, you can execute "python train.py" to train your YOLOv8

