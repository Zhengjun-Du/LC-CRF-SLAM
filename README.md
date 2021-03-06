![LC-CRF-SLAM](https://github.com/Zhengjun-Du/LC-CRF-SLAM/blob/master/LC-CRF-SALM.png)

# Accurate Dynamic SLAM using CRF-based Long-term Consistency
This is the source code of the paper: [Accurate Dynamic SLAM using CRF-based Long-term Consistency](https://cg.cs.tsinghua.edu.cn/people/~shisheng/Papers/OC-CRF/lccrf_tvcg.pdf)  
**Authors:** Zheng-Jun Du, Shi-Sheng Huang, Tai-Jiang Mu, Qunhe Zhao, Ralph R. Martin and Kun Xu

Our implementation is based on the framework of [ORBSLAM2](https://github.com/raulmur/ORB_SLAM2), any question about the code please contact me via my email: duzj19@mails.tsinghua.edu.cn.

## Build
We compile the project on Ubuntu 16.04 LTS, and the compiling is similar to ORBSLAM2, please refer to https://github.com/raulmur/ORB_SLAM2#3-building-orb-slam2-library-and-examples for more detail.
All dependency libraries are included in the Thirdparty directory.

## Dataset
We test our algorithm on two dynamic dataset: [TUM RGB-D dynamic dataset](https://vision.in.tum.de/data/datasets) and [BONN RGB-D dynamic dataset](http://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/).

## Usage
>./rgbd_tum path_to_sequence path_to_association path_to_settings  

more detail see the file: rgbd_tum.cc

## Evaluate
The tools of ate/rpe evaluation locate in /Examples/RGB-D  
usage: (python 2.7)
> python evaluate_ate/evaluate_rpe.py groundtruth.txt trajectory.txt --verbose

## Notes
If the frame viewers are frozen, close it, it will be reactived.
