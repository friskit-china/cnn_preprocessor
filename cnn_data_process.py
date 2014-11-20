# -*- coding: utf-8
__author__ = 'shibotian'

import scipy.io as sio
import numpy as np
import h5py, os, sys

class DataFile:
    """
    处理单个文件
    """

    def __init__(self, setting_obj, x_filename, y_filename):
        # 文件名不能为空
        assert x_filename is not None
        assert y_filename is not None

        # 初始化一些成员变量
        self.data_x = None
        self.data_y = None
        self.out_data_x = None
        self.out_data_y = None
        self.out_dim_x = None
        self.out_dim_y = None
        self.in_dim_x = None
        self.in_dim_y = None
        self.x_dtype = setting_obj.x_dtype
        self.y_dtype = setting_obj.y_dtype
        self.setting_obj = setting_obj
        self.is_processed = False

        # 输入文件格式为mat
        if self.setting_obj.type_str is 'mat':
            self.data_x = sio.loadmat(x_filename)[self.setting_obj.x_name]
            self.data_y = sio.loadmat(y_filename)[self.setting_obj.y_name]

        # 输入文件格式为hdf5
        elif self.setting_obj.type_str is 'hdf5':
            pass
        pass

        # 获取数据总长度
        self.in_dim_x = self.data_x.shape
        self.in_dim_y = self.data_y.shape

        # 如果x是复数，则进行abs处理               #############复数abs？
        if self.setting_obj.is_x_complex:
            self.data_x = np.abs(self.data_x)

        # 如果输入y是多维，则进行argmax处理
        if not self.setting_obj.is_y_1d:
            if self.setting_obj.is_y_row:
                self.data_y = self.data_y.argmax(axis=0)
            else:
                self.data_y = self.data_y.argmax(axis=1)
            pass

        # 将y展平
        self.data_y = self.data_y.reshape(-1)

        # 类型转换
        self.data_x = self.data_x.astype(self.setting_obj.x_dtype)
        self.data_y = self.data_y.astype(self.setting_obj.y_dtype)

        # 数据归一化处理, 针对每一张图片？还是每一段语音处理之后的 ?? * 2560
        if self.setting_obj.is_need_normalize:
            if self.setting_obj.normalize_str == 'z-score':  # 使用z-score归一化      #####对每个文件进行还是对整个语音进行？
                mean = self.data_x.mean()
                std = self.data_x.std()
                self.data_x = (self.data_x - mean) / std
            elif self.setting_obj.normalize_str == 'min-max':   # 使用min-max归一化
                max_x = self.data_x.max()
                min_x = self.data_x.min()
                self.data_x = (self.data_x - min_x) / (max_x - min_x)
                ## 可以补充其他归一化方法

        # 如果frame_before 或 frame_after有一个不等于零，则说明需要进行拼接，变成二维图
        if self.setting_obj.frame_before is not 0 or self.setting_obj.frame_after is not 0:
            # 获取到前后保留帧数
            frame_before = self.setting_obj.frame_before
            frame_after = self.setting_obj.frame_after

            # 计算每一维拼接的尺度
            pic_dim = (self.in_dim_x[0], frame_before + frame_after + 1)

            # 计算输出数据的维度(前后边界真补零)
            # self.out_dim_x = (self.in_dim_x[1] - frame_before - frame_after, pic_dim[0] * pic_dim[1])
            # self.out_dim_y = (self.in_dim_x[1] - frame_before - frame_after)

            # 对于二维图来说是需要前后补零的，创建补零矩阵，并进行拼接
            zeros_before = np.zeros((self.in_dim_x[0], frame_before))
            zeros_after = np.zeros((self.in_dim_x[0], frame_after))
            self.data_x = np.hstack((zeros_before, self.data_x, zeros_after))

            # 计算输出数据的维度(前后边界帧抛弃)
            self.out_dim_x = (self.in_dim_x[1], pic_dim[0] * pic_dim[1])
            self.out_dim_y = (self.in_dim_x[1], pic_dim[0] * pic_dim[1])

            # 初始化一个数组，尺寸是(self.out_data_dim[0], pic_dim[0], pic_dim[1])
            self.out_data_x = np.zeros((self.out_dim_x[0], pic_dim[0] * pic_dim[1]), dtype=self.setting_obj.x_dtype)
            self.out_data_y = np.zeros((self.out_dim_x[0]), dtype=self.setting_obj.y_dtype)

            # 从frame_before帧开始遍历到总帧数-frame_after(self.data_dim[1] = frame_after)为止
            # for i in range(frame_before, self.in_dim_x[1] - frame_after):
            #     # 获取整个切片，并转换成(1, dim[0]*dim[1])维数据
            #     self.out_data_x[i - frame_before] = self.data_x[:, i - frame_before: i + frame_after + 1].reshape(1, -1)
            #     self.out_data_y[i - frame_before] = self.data_y[i - frame_before]
            #     pass
            # pass

            # 补零版遍历
            for i in range(frame_before, self.in_dim_x[1] + frame_before):
                self.out_data_x[i - frame_before] = self.data_x[:, i - frame_before: i + frame_after + 1].reshape(1, -1)
                self.out_data_y[i - frame_before] = self.data_y[i - frame_before]
                pass




class SettingObj:
    """
    选项管理类
    """
    def __init__(self, setting_filename=None):
        # 默认配置

        self.is_echo = True                 # 是否输出信息
        self.frame_before = 7               # 当前帧前数量
        self.frame_after = 8                # 当前帧后数量
        self.pic_dim = (160, 16)            # 图片维度 （行数/高 * 列/宽）
        self.is_x_complex = True            # x是否是复数（如果是则要经过abs运算）
        self.is_y_1d = False                # y的格式是否是1维（还有可能是n维0-1值）
        self.is_y_row = True                # y是否是行排列？dim=(预测值, 帧数)，False则为每行为预测输出即dim=(帧数,预测值)
        self.is_need_normalize = True       # 是否需要归一化
        self.normalize_str = 'z-score'      # 归一化方法
        self.x_dtype = 'float32'            # 输出X类型
        self.y_dtype = 'int32'              # 输出y类型
        self.x_name = 'fftdata'             # 输出X名称
        self.y_name = 'label'               # 输出y名称
        self.type_str = 'mat'               # 文件格式
        self.is_rand_dataset = True         # 是否随机重排数据集

        # 读取配置文件
        if setting_filename:
            pass


class Dataset:
    """
    数据集类
    """
    def __init__(self, x_file_list, y_file_list, output_dataset, setting_obj=SettingObj()):
        self.x_files = []
        self.y_files = []
        self.setting_obj = setting_obj
        self.x_out_dim = (0, 0)
        self.y_out_dim = (0,)
        self.dataset_file = h5py.File(output_dataset, 'w')
        pic_size = self.setting_obj.pic_dim[0] * self.setting_obj.pic_dim[1]
        self.x_dataset = self.dataset_file.create_dataset('X', shape=(0, pic_size),
                                                          maxshape=(None, pic_size), dtype=setting_obj.x_dtype)
        self.y_dataset = self.dataset_file.create_dataset('y', shape=(0,),
                                                          maxshape=(None,), dtype=setting_obj.y_dtype)

        with open(x_file_list) as xfl:
            lines = xfl.readlines()
            self.x_files = [line.strip('\n') for line in lines]

        with open(y_file_list) as yfl:
            lines = yfl.readlines()
            self.y_files = [line.strip('\n') for line in lines]

            # 需要确保两个列表文件内文件个数相同
        assert len(self.x_files) == len(self.y_files)

        index = 0
        for item in zip(self.x_files, self.y_files):
            x_filename = item[0]
            y_filename = item[1]
            if setting_obj.is_echo:
                sys.stdout.write('    processing file: ' + x_filename + ' ........')
                sys.stdout.flush()
            df = DataFile(x_filename=x_filename, y_filename=y_filename, setting_obj=setting_obj)
            if setting_obj.is_echo:
                print 'done'
            # 更新数据集的维度
            self.x_out_dim = (self.x_out_dim[0] + df.out_dim_x[0], df.out_dim_x[1])
            self.y_out_dim = (self.y_out_dim[0] + df.out_dim_y[0], )

            # 更新数据集大小
            self.x_dataset.resize(self.x_out_dim)
            self.y_dataset.resize(self.y_out_dim)
            for item_x, item_y in zip(df.out_data_x, df.out_data_y):
                self.x_dataset[index, :] = item_x
                self.y_dataset[index] = item_y
                index += 1
                pass
        self.dataset_file.close()

        # 随机乱序
        if self.setting_obj.is_rand_dataset:
            if self.setting_obj.is_echo:
                print '    randomization',
            # 将原来的数据文件改名（后缀加tmp），并创建一个新的数据文件
            tmp_dataset_filename = output_dataset + '.tmp'
            os.rename(output_dataset, tmp_dataset_filename)
            tmp_dataset = h5py.File(tmp_dataset_filename, 'r')
            self.dataset_file = h5py.File(output_dataset, 'w')

            # 为新数据文件创建数据集
            self.x_dataset = self.dataset_file.create_dataset('X', shape=tmp_dataset['X'].shape)
            self.y_dataset = self.dataset_file.create_dataset('y', shape=tmp_dataset['y'].shape)

            # 生成随机索引
            rand_indices = np.random.permutation(tmp_dataset['X'].shape[0])

            # 按照随机索引，将数据重排到新的数据文件中
            for i in range(len(rand_indices)):
                if self.setting_obj.is_echo:
                    if i % 100 == 0:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                self.x_dataset[i] = tmp_dataset['X'][rand_indices[i]]
                self.y_dataset[i] = tmp_dataset['y'][rand_indices[i]]
            self.dataset_file.close()
            # 删除临时文件
            os.remove(tmp_dataset_filename)

        if self.setting_obj.is_echo:
            print ''
            print '    all works done'

if __name__ == '__main__':
    setting_obj = SettingObj()
    if len(sys.argv) == 1:
        print 'usage: python %s [x file list name] [y file list name] [output dataset name] [True|False]' % sys.argv[0]
        print '        e.g. python %s x.lst y.lst dataset_train.h5 False' % sys.argv[0]
        # setting_obj.is_rand_dataset = False
        # ds = Dataset(x_file_list='train/train_x.lst', y_file_list='train/train_label.lst', output_dataset='dataset/train/dataset_train.h5')
    else:
        setting_obj.is_rand_dataset = True if sys.argv[4] == 'True' else False
        ds = Dataset(x_file_list=sys.argv[1], y_file_list=sys.argv[2],
                     output_dataset=sys.argv[3], setting_obj=setting_obj)
        pass