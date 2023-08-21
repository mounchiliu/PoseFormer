### 1. Data Preparation - Human3.6M
Refer to videoPose: https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md

1. download the dataset in its original format.
    You only need to download Poses -> D3 Positions for each subject (1, 5, 6, 7, 8, 9, 11)

   链接: https://pan.baidu.com/s/1gwUlzr1l-7CM8H6chCvxeQ?pwd=mtea

2. install `cdflib` Python library via `pip install cdflib`.

3. Extract the archives named `Poses_D3_Positions_S*.tgz` (subjects 1, 5, 6, 7, 8, 9, 11) to a common directory. 
    ```commandline
   cd data/
   for file in *.tgz; do tar -xvzf $file; done
   rm *.tgz
   ```
   Your directory tree should look like this:
   ```commandline
    /path/to/dataset/S1/MyPoseFeatures/D3_Positions/Directions 1.cdf
    /path/to/dataset/S1/MyPoseFeatures/D3_Positions/Directions.cdf
    ```
    Then, run the preprocessing script:
    ```commandline
    cd data
    python ./data/prepare_data_h36m.py --from-source-cdf ./data/
    cd ..
    ```

    If everything goes well, you are ready to go.
      
   注意：当生成一次npz文件后，需要删除原文件才能运行此脚本再次生成数据文件
   

#### Intro to Human3.6M
Human3.6M 是目前 3D HPE 任务最为常用的数据集之一，包含了 360 万帧图像和对应的 2D/3D 人体姿态。该数据集在实验室环境下采集，通过 **4 个高清相机**同步记录 4 个视角下的场景，并通过 MoCap 系统获取精确的人体三维关键点坐标及关节角。

Human3.6M 的评价指标主要有 Mean Per Joint Position Error (MPJPE) 和 P-MPJPE。其中，MPJPE 是所有关键点预测坐标与 ground truth 坐标之间的平均欧式距离，一般会事先将预测结果与 ground truth 的根关节对齐；P-MPJPE 则是将预测结果通过 Procrustes Analysis 与 ground truth 对齐，再计算 MPJPE。

### 2. 2D detections for Human3.6M

see in https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md


### 3. Evaluating pre-trained models

We provide the pre-trained 81-frame model (CPN detected 2D pose as input) [here](https://drive.google.com/file/d/1oX5H5QpVoFzyD-Qz9aaP3RDWDb1v1sIy/view?usp=sharing). To evaluate it, put it into the `./checkpoint` directory and run:

Note: we need to download 2D detection, as shown in https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md, and place the `.npz` file in data directory

```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 81 -c checkpoint --evaluate detected81f.bin
```

We also provide pre-trained 81-frame model (Ground truth 2D pose as input) [here](https://drive.google.com/file/d/18wW4TdNYxF-zdt9oInmwQK9hEdRJnXzu/view?usp=sharing). To evaluate it, put it into the `./checkpoint` directory and run:

```bash
python run_poseformer.py -k gt -f 81 -c checkpoint --evaluate gt81f.bin
```

### 4. Training new models

* To train a model from scratch (CPN detected 2D pose as input), run:

```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 27 -lr 0.00004 -lrd 0.99
```

`-f` controls how many frames are used as input. 27 frames achieves 47.0 mm, 81 frames achieves achieves 44.3 mm. 

* To train a model from scratch (Ground truth 2D pose as input), run:

```bash
python run_poseformer.py -k gt -f 81 -lr 0.0004 -lrd 0.99
```

81 frames achieves 31.3 mm (MPJPE). 

