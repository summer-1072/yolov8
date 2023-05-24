<img alt="yolov8.png" src="assert%2Fyolov8.png" width="1500"/>

Model: Ultralytics YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility.
Now, We have rewritten the YOLOv8 to make it simpler and easier to use. At the same time, we have tested the effectiveness of YOLOv8 in object detection for autonomous driving on the BDD100K dataset.
Finally, we designed the YOLOBI network to try to optimize the effectiveness of object detection.

<img alt="performance.png" src="assert%2Fperformance.png" width="1500"/>

Dataset: We released the largest and most diverse driving video dataset with rich annotations called BDD100K. You can access the data for research now at http://bdd-data.berkeley.edu. We have recently released an arXiv report on it. And there is still time to participate in our CVPR 2018 challenges!

<img alt="BDD100K.png" src="assert%2Fbdd100k.png" width="1280"/>

Show: We used the model to infer a portion of the data to demonstrate the actual detection effect.

<img alt="bdd100k.gif" src="assert%2Fbdd100k.gif" width="1280"/>

Weight and Dataset: We have provided pre training weights for everyone to use. We have provided a download address for the dataset and pre trained model weights.
Address: https://pan.baidu.com/s/1gnPDkFhPCL7-n4Y_VME2yw   password: 1111

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

We provide detection results for YOLOv8s and YOLOBix in each category to evaluate the upper and lower limits of the model detection ability.

| class        | instances | YOLOv8s P | YOLOv8s R | YOLOv8s mAP50 | YOLOv8s mAP50-95 |
|--------------|-----------|-----------|-----------|---------------|------------------|
| person       | 13085     | 0.7592    | 0.5951    | 0.708         | 0.4169           |
| bike         | 1000      | 0.6139    | 0.477     | 0.5683        | 0.3259           |
| car          | 101893    | 0.808     | 0.7606    | 0.8296        | 0.583            |
| motor        | 447       | 0.6871    | 0.4273    | 0.5433        | 0.3014           |
| rider        | 646       | 0.6628    | 0.4381    | 0.5783        | 0.3383           |
| bus          | 1597      | 0.6923    | 0.5636    | 0.6683        | 0.5624           |
| train        | 14        | 0.0       | 0.0       | 0.0           | 0.0              |
| truck        | 4233      | 0.6893    | 0.5863    | 0.6631        | 0.5295           |
| traffic sign | 33914     | 0.7506    | 0.6633    | 0.7312        | 0.4535           |
| traffic light| 24869     | 0.7392    | 0.6573    | 0.7117        | 0.3312           |

| class        | instances | YOLOBix  P | YOLOBix  R | YOLOBix mAP50 | YOLOBix mAP50-95 |
|--------------|-----------|------------|------------|---------------|------------------|
| person       | 13220     | 0.7481     | 0.6965     | 0.7655        | 0.4649           |
| bike         | 1004      | 0.6182     | 0.5627     | 0.6192        | 0.3721           |
| car          | 102341    | 0.813      | 0.8018     | 0.854         | 0.6016           |
| motor        | 449       | 0.6687     | 0.4944     | 0.5927        | 0.3467           |
| rider        | 648       | 0.712      | 0.5417     | 0.6597        | 0.4043           |
| bus          | 1597      | 0.7023     | 0.6293     | 0.7128        | 0.6029           |
| train        | 14        | 0.0        | 0.0        | 0.0           | 0.0              |
| truck        | 4241      | 0.6836     | 0.6496     | 0.7008        | 0.5588           |
| traffic sign | 34615     | 0.7155     | 0.756      | 0.7751        | 0.4807           |
| traffic light| 26355     | 0.7349     | 0.7275     | 0.7488        | 0.3508           |


Requirement: We suggest using Python =3.8, torch >=1.10.0, cuda >=11.3.

Use:
1) Download Dataset Address: https://bdd-data.berkeley.edu/
   You can download the data and place it in the following directory structure. The images folder contains images related to train, val, and test, the labels folder contains label files related to train and val, and the videos folder contains video files that require inference.
   You can also download the dataset directly from Baidu Netdisk. Address: https://pan.baidu.com/s/1gnPDkFhPCL7-n4Y_VME2yw   password: 1111
 
   <img alt="data.png" src="assert%2Fdata.png"/>
  
   run "python /utils/dataset.py" to generate train and val data labels that are compatible with YOLO in the labels folder.

2) Train Model run "python /run/train.py" to train model.

3) Fuse Conv and BN run "python /model/tools.py" to fuse Conv and BN layer.

4) Valid Model run "python /run/valid.py" to valid model performance.

5) Predict run "python /run/predict.py" to predict data, the model supports inference functions for images and videos.

6) If you want to directly use the trained model to infer on the BDD100K dataset, please download the model weights directly from the above address and place them in the following directory structure.
   
   <img alt="weight.jpg" src="assert%2Fweight.jpg"/>