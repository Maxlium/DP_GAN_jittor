# 基于DP_GAN的风景生成器



## 简介

​	图像生成任务一直以来都是十分具有应用场景的计算机视觉任务，从语义分割图生成有意义、高质量的图片仍然存在诸多挑战，如保证生成图片的真实性、清晰程度、多样性、美观性等。

​	其中，条件图像合成，即输入图片数据，合成真实感图片，在内容生成与图片编辑领域有广泛应用。一种条件图像合成的方式是，用两张图片作为输入，经过处理转换后生成一张新的图片，其中一张输入为语义分割图片（称为mask图），指示生成图片（称为gen图）的语义信息；另一张输入为参考风格图片（称为ref图），从色调等方面指示gen图的风格信息：

<div align="center">
  <img src="https://s3.bmp.ovh/imgs/2023/08/21/52509d62c844d223.jpg">
</div>

​	清华大学计算机系图形学实验室从Flickr官网收集了12000张高清（宽512、高384）的风景图片，并制作了它们的语义分割图。其中，10000对图片被用来训练。采用1000张图像进行测试。



## 配置环境

### 运行环境

- Ubuntu 20.04.6 LTS
- python>=3.7.0
- jittor>=1.3.8

在单张3090上训练了4天。

### 安装依赖

执行下面命令安装python等依赖

```python
pip install -r requirements.txt
```

### 训练数据集

训练数据一共使用10000张图片

[训练数据集](https://cloud.tsinghua.edu.cn/f/063e7fcfe6a04184904d/?dl=1)

### 测试数据集测

测试数据集使用A和B榜两种数据集1000张图片

[A榜数据集](https://cloud.tsinghua.edu.cn/d/cb748039138145f2b971/)

[B榜数据集](https://cloud.tsinghua.edu.cn/d/9dd48340bbde4d9b9ffa/)

## 训练

- 下载好训练数据集并保存在项目文件夹下

​		—DP_GAN_jittor/train_resized

​		——imgs

​		——labels					

- 预训练模型采用的是 `Jittor` 框架自带的 `vgg19` 模型，无需额外下载，在代码运行的过程中会载入到内存里。					

- 在终端执行下列命令训练代码
  - --name:保存训练时的modal、loss和figure相关模型内容到checkpoints/name文件夹下
  - --input_path:{训练数据集路径（即train_resized文件夹所在路径）}
  - --batch_size: 训练时的批大小
  

```shell
CUDA_VISIBLE_DEVICES="2" python3 train.py --name jittor_train --input_path ./train_resized --batch_size 4
```

- 训练之前会进行数据划分，将**训练集**和**验证集**保存在datasets文件夹下。
- 训练好的权重：链接: https://pan.baidu.com/s/1dehotg6d9J2mSxp91DSqHA?pwd=1341 提取码: 1341 ；下载好并放入**checkpoints/jittor_train4** 文件夹下

## 测试

- 下载**A榜**或者**B榜**的数据保存在项目文件夹下

​		—DP_GAN_jittor/A(or B)

​		——val_B_labels_resized

​		——label_to_img.json	

- 在终端执行下行命令测试代码，并将生成的结果保存到output_dir中
  - name表示之前训练好的模型保存的一个文件夹的名字；
  - input_path表示label的路径，也可以直接放在项目文件夹下；
  - json_path表示label_to_img的一个对应关系的路径，和input_path放在同一个路径下；
  - img_path训练数据集的图片路径（即train_resized/imgs文件夹所在路径，它提供ref图）；
  - output_path表示测试结果存放的路径。

```shell
CUDA_VISIBLE_DEVICES="2" python3 test.py --name jittor_train4 --input_path ./B/val_B_labels_resized  --json_path ./B/label_to_img.json --img_path ./train_resized/imgs --output_path ./results
```
## 训练损失函数和FID指标
![combined](https://github.com/Maxlium/DP_GAN_jittor/assets/89024317/0b310a1c-1d0d-4341-92b9-c37949bf1e7c)
![plot_fid](https://github.com/Maxlium/DP_GAN_jittor/assets/89024317/29b4c517-b394-4f26-a814-04668b1fb929)










