# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import sys
import zipfile
from glob import glob
from shutil import rmtree

import h5py
import numpy as np

sys.path.append('../')
from common.h36m_dataset import Human36mDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from common.utils import wrap

# work dir 设置在PoseFormer目录下
output_filename = os.getcwd() + '/data/data_3d_h36m'  # 将生成的数据存放在data目录下
output_filename_2d = os.getcwd() + '/data/data_2d_h36m_gt'  # 将生成的数据存放在data目录下
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

# 这个文件由于处理human3.6M原始数据
if __name__ == '__main__':
    # if os.path.basename(os.getcwd()) != 'data':
    #     print('This script must be launched from the "data" directory')
    #     exit(0)

    parser = argparse.ArgumentParser(description='Human3.6M dataset downloader/converter')

    # Convert dataset preprocessed by Martinez et al. in https://github.com/una-dinosauria/3d-pose-baseline
    parser.add_argument('--from-archive', default='', type=str, metavar='PATH', help='convert preprocessed dataset')

    # Convert dataset from original source, using files converted to .mat (the Human3.6M dataset path must be specified manually)
    # This option requires MATLAB to convert files using the provided script
    parser.add_argument('--from-source', default='', type=str, metavar='PATH', help='convert original dataset')

    # Convert dataset from original source, using original .cdf files (the Human3.6M dataset path must be specified manually)
    # This option does not require MATLAB, but the Python library cdflib must be installed
    parser.add_argument('--from-source-cdf', default='', type=str, metavar='PATH', help='convert original dataset')

    args = parser.parse_args()

    if args.from_archive and args.from_source:
        print('Please specify only one argument')
        exit(0)

    if os.path.exists(output_filename + '.npz'):
        print('The dataset already exists at', output_filename + '.npz')
        exit(0)

    # 处理3D关键点Pose数据
    # 根据不同格式的数据进行处理
    if args.from_archive:
        print('Extracting Human3.6M dataset from', args.from_archive)
        with zipfile.ZipFile(args.from_archive, 'r') as archive:
            archive.extractall()

        print('Converting...')
        output = {}
        for subject in subjects:
            output[subject] = {}
            file_list = glob('h36m/' + subject + '/MyPoses/3D_positions/*.h5')
            assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
            for f in file_list:
                action = os.path.splitext(os.path.basename(f))[0]

                if subject == 'S11' and action == 'Directions':
                    continue  # Discard corrupted video

                with h5py.File(f) as hf:
                    positions = hf['3D_positions'].value.reshape(32, 3, -1).transpose(2, 0, 1)
                    positions /= 1000  # Meters instead of millimeters
                    output[subject][action] = positions.astype('float32')

        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=output)

        print('Cleaning up...')
        rmtree('h36m')

        print('Done.')

    elif args.from_source:
        print('Converting original Human3.6M dataset from', args.from_source)
        output = {}

        from scipy.io import loadmat

        for subject in subjects:
            output[subject] = {}
            file_list = glob(args.from_source + '/' + subject + '/MyPoseFeatures/D3_Positions/*.cdf.mat')
            assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
            for f in file_list:
                action = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]

                if subject == 'S11' and action == 'Directions':
                    continue  # Discard corrupted video

                # Use consistent naming convention
                canonical_name = action.replace('TakingPhoto', 'Photo') \
                    .replace('WalkingDog', 'WalkDog')

                hf = loadmat(f)
                positions = hf['data'][0, 0].reshape(-1, 32, 3)
                positions /= 1000  # Meters instead of millimeters
                output[subject][canonical_name] = positions.astype('float32')

        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=output)

        print('Done.')

    elif args.from_source_cdf:  # human3.6M使用 cdf
        print('Converting original Human3.6M dataset from', args.from_source_cdf, '(CDF files)')
        output = {}  # 初始化一个list存放所有subject的结果
        # 按照每个subject整理存储数据
        import cdflib

        for subject in subjects:  # 对于每一组数据，e.g. S1, S5, ...
            output[subject] = {}  # 初始化list for this subject
            file_list = glob(args.from_source_cdf + '/' + subject + '/MyPoseFeatures/D3_Positions/*.cdf')  # 获取所有cdf文件
            assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
            for f in file_list:  # 处理每个cdf
                action = os.path.splitext(os.path.basename(f))[0]  # 从文件名中获取该数据对应action

                # for debug, loading sample samples of data for training
                if subject != "S1" and subject != "S9":
                    continue
                if action != 'Directions':
                    continue
                # end

                if subject == 'S11' and action == 'Directions':
                    continue  # Discard corrupted video

                # Use consistent naming convention
                canonical_name = action.replace('TakingPhoto', 'Photo') \
                    .replace('WalkingDog', 'WalkDog')

                hf = cdflib.CDF(f)  # read cdf file into system # hf['Pose']： 1 x num_frame x (32*3)
                positions = hf['Pose'].reshape(-1, 32, 3)  # 获得32个点的所有3D坐标，并转化为相应格式
                positions /= 1000  # Meters instead of millimeters
                output[subject][canonical_name] = positions.astype('float32')  # save in float32

        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=output)

        print('Done.')

    else:
        print('Please specify the dataset source')
        exit(0)

    # 从3Dgt数据中提取处理2D关键点数据: 根据相机坐标，将3D投影到2D上，获得2D gt
    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = Human36mDataset(output_filename + '.npz')  # 读取D数据,load human36m Dataset 格式(data[subject][action] = {position, camera})
    output_2d_poses = {}
    for subject in dataset.subjects():  # 对于每一组数据，e.g. S1, S5, ...
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():  # 获取这组数据里的某个action的所有数据（对应一个视频）
            anim = dataset[subject][action]

            positions_2d = []
            for cam in anim['cameras']:  # 获取相机参数
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])  # 将3D gt position转换到世界坐标系下
                pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)  # 定义func：project_to_2d 和使用func的参数, 将3D坐标转换到2D上
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d

    print('Saving...')
    # save metadata for summarized info of kp
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)

    print('Done.')