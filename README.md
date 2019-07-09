
## **Aerial Algorithms & Aerial Traversability and Planning Dataset**

#### A strategy and evaluation method for ground global path planning based on aerial images ####

This repository contains the algorithms and dataset developed in our research about strategies and evaluation methods for ground global path planning based on aerial images. 

If you use this repository, please cite our [paper](https://doi.org/10.1016/j.eswa.2019.06.067).

### **Algorithms**

The module trav does traversability calculation, while graphmapx generates traversability graphs and paths. The examples script contains small snippets that show how to use the aforementioned modules. The experiments script contains the code required to reproduce the experiments in our paper.

### **Traversability and Planning Dataset**

The Aerial Traversability and Planning Dataset is in the dataset directory.
Following is a brief description of its contents.

- images: aerial RGB images with resolution 1000x1000 pixels and approx. 0.3 x 0.3 m²/p;
- labels: binary traversability labels for each sample in images;
- keypoints-reachable: marked points of interest to use as source and target points;
- keypoints-unreachable: marked points to test whether a path planning approach is able to identify infeasible paths.

### **Path Planning Samples**

![alt text](https://raw.githubusercontent.com/cosmonautd/Aerial/master/samples/1.jpg)
![alt text](https://raw.githubusercontent.com/cosmonautd/Aerial/master/samples/2.jpg)
![alt text](https://raw.githubusercontent.com/cosmonautd/Aerial/master/samples/3.jpg)
![alt text](https://raw.githubusercontent.com/cosmonautd/Aerial/master/samples/4.jpg)
![alt text](https://raw.githubusercontent.com/cosmonautd/Aerial/master/samples/5.jpg)
![alt text](https://raw.githubusercontent.com/cosmonautd/Aerial/master/samples/6.jpg)
![alt text](https://raw.githubusercontent.com/cosmonautd/Aerial/master/samples/7.jpg)
![alt text](https://raw.githubusercontent.com/cosmonautd/Aerial/master/samples/8.jpg)