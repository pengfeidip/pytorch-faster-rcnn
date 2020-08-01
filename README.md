# An Object Detection Framework
The project started off as a task of reimplementing Faster RCNN and later becomes a detection framework that integrates some of the most popular detection models like Faster RCNN, Cascade RCNN, RetinaNet and etc. It borrows many stuff from various open-source projects and among them mmdetection is the one that it follows the most.



### Design

The design very much follows the design of mmdetection. A detector is made of detection  components like backbone, neck and many kinds of heads. It uses registry + builder to connect these components. Trainer is a separate component that uses hooks to add events and keep track of training process. Tester supports multi-image inference which is something mmdetection has not implemented yet. Both dataset and dataloader simply uses mmdetection's which means data preprocessing is also using mmdetection's. While both training and inferencing support multi-image, distributed training is not supported. 



### Implementation of models

Implementation of some models very much follows mmdetection, e.g. FPN, Cascade RCNN, RetinaNet etc. Other models I added some of my thoughts, they are Libra RCNN, FCOS-ATSS, Generalized Focal Loss etc. For example in ATSS, mmdet still uses bbox delta as regression output while my implementation uses LTRB(left, top, right, bottom) as output of regression. We both achieve similar performance on VOC dataset. 



### Supported

Backbones:

- VGG16
- ResNet(50, 101)

Two stage models:

- Faster RCNN
- Cascade RCNN
- Libra RCNN
- Double Head

Single stage models

- RetinaNet
- FCOS
- FCOS+ATSS
- Generalized Focal Loss



### Benchmark

Training, testing, validating and benchmarking are all using VOC2007 dataset, specifically VOC2007 trainval as training dataset and VOC2007 test as test dataset. Benchmarks on coco dataset are not available. 

- Models that gives mAP and AP50 for mmdet's results are trained on a single GPU 2080Ti. The configs are all using default. 
- All the results are taken from epoch 12 except for ATSS, GFL models.
- Configs are trying to follow mmdet's config whenever they can. 



| Faster RCNN          | mAP   | AP50  | MMdet mAP | MMdet AP50 |
| -------------------- | :---- | ----- | --------- | ---------- |
| FasterRCNN_R50_FPN   | 0.392 | 0.741 | 0.398     | 0.753      |
| CascadeRCNN_R50_FPN  | 0.467 | 0.757 | 0.473     | 0.761      |
| RetinaNet_R50_FPN    | 0.41  | 0.719 | 0.411     | 0.717      |
| DoubleHead_R50_FPN   | 0.415 | 0.744 | 0.432     | 0.758      |
| LibraRCNN_R50_FPN    | 0.426 | 0.759 | 0.425     | 0.763      |
| FCOS_R50_FPN         | 0.422 | 0.731 | 0.434     | 0.74       |
| ATSS_R50_FPN         | 0.443 | 0.72  | 0.442     | 0.72       |
| ATSS+QFL_R50_FPN     |       |       | 0.451     | 0.718      |
| ATSS+QFL+DFL_R50_FPN |       |       | 0.449     | 0.702      |
|                      |       |       |           |            |







Thanks to the following amazing projects:

[mmdetection]: https://github.com/open-mmlab/mmdetection
[simple-faster-rcnn-pytorch]: https://github.com/chenyuntc/simple-faster-rcnn-pytorch
[pytorch-retinanet]: https://github.com/kuangliu/pytorch-retinanet
[ATSS]: https://github.com/sfzhang15/ATSS

