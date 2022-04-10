# opencv-yolov3
Started this to learn more about OpenCV's CuDNN module, running on yolo-v3 model trained on the coco dataset

Weights file not included as it's too big. Refer to [this article](https://pjreddie.com/darknet/yolo/) by YoloV3 creator [Joseph Redmon](https://pjreddie.com/)

Most code (shamefully) copied from this [Article](https://www.thepythoncode.com/article/yolo-object-detection-with-opencv-and-pytorch-in-python)

Authored by [Abdou/Rockickz](https://github.com/x4nth055)

Coded to run on WSL2 currently. HOWTO to follow, but will need to run MJPEG Streamer(Windows App) and Xming/any Xserver running on host OS(Windows)

OpenCV will also need to be compiled on WSL2, as well as having Cuda and CuDNN installed and configured correctly