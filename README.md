# 违规发言检测：

这是使用PaddleHub完成的一个违规发言检测的项目，由于小老弟初学水平有限所以该项目还有很大的改进空间，以后可以慢慢改慢慢做
## 一、PaddleHub简介
PaddleHub是百度飞桨PaddlePaddle开源深度学习平台的预训练模型应用工具，使用该工具就可以很便捷的使用少量的代码量快速部署各种模型。从而大大节省开发人员的时间；PaddleHub可以便捷地获取PaddlePaddle生态下的预训练模型，完成模型的管理和一键预测。配合使用Fine-tune API，可以基于大规模预训练模型快速完成迁移学习，让预训练模型能更好地服务于用户特定场景的应用。更多详情可查看[PaddleHub官网](https://www.paddlepaddle.org.cn/hub)
## 二、模型简介
该模型使用的是PaddleHub中的**porn_detection_lstm**预训练模型，该预训练模型使用的是LSTM网络结构并按字粒度进行切词，具有较高的分类精度。数据集是百度自建的数据集。该模型最大句子长度为256字，仅支持预测。
以下是使用的各模型的版本号：
> porn_detection_lstm - 1.1.0
> Python - 3.7.6
> PaddleHub - 1.8.1
## 三、代码实现
### 1、模型的安装
首先我们要安装PaddleHub工具，PaddleHub需要与飞桨一起使用，其硬件和操作系统的适用范围与飞桨相同。注意：飞桨版本需要>= 1.7.0。

```bash
$pip install paddlehub --upgrade -i https://mirror.baidu.com/pypi/simple  # 安装最新版本，使用百度源
```
之后就可以便捷的部署和使用各种预先训练好的模型，下方代码是安装该项目的模型

```bash
$ hub install porn_detection_lstm==1.1.0
```
### 2、引入相关库
```python
from __future__ import print_function
import json
import six
import paddlehub as hub
import sys
```
其中sys模块是很常用的模块， 它封装了与python解释器相关的数据，例如sys.modules里面有已经加载了的所有模块信息，sys.path里面是PYTHONPATH的内容，而sys.argv则封装了传入的参数数据。
使用sys.argv可以在命令行模式运行的时候接收相关输入数据。
### 3、命令行方式运行

```python
$ python filename.py inputdata
```
在filename.py下打开CMD并执行上述语句即可实现命令行方式运行，其中filename.py为本项目的程序文件，在本例中为demo.py。inputdata为待检测的语句，输入格式为一段文字加上双引号，在本例中为“打击黄牛党”。

### 4、使用API方式运行
使用PornDetectionLSTM的API接口的实现方式如下所示。其它有关与该预训练模型的详细情况请查看[PaddleHub中相关介绍](https://www.paddlepaddle.org.cn/hubdetail?name=porn_detection_lstm&en_category=TextCensorship)

```python
from __future__ import print_function
import json
import six
import paddlehub as hub

if __name__ == "__main__":
    # Load porn_detection_lstm module
    porn_detection_lstm = hub.Module(name="porn_detection_lstm")

    test_text = ["黄片下载"]

    input_dict = {"text": test_text}

    results = porn_detection_lstm.detection(data=input_dict,use_gpu=True, batch_size=1)

    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        if six.PY2:
            print(
                json.dumps(results[index], encoding="utf8", ensure_ascii=False))
        else:
            print("您的发言为:",test_text[0])
            
    lst = list(results[0].values())
    if lst[1]==1:
        print("您的发言涉嫌违规，已被屏蔽！")
    else:
        print("已发送！")
```

## 四、效果展示
两种方式的运行效果如图所示，在验证多种语句之后发现该模型都能达到较为好的效果，之后还可以更进一步的将过程给可视化的显示出来。
![命令行运行](https://img-blog.csdnimg.cn/20200824163604195.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4NTM2MzEz,size_16,color_FFFFFF,t_70#pic_center)
![API方式运行](https://img-blog.csdnimg.cn/20200824164120117.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4NTM2MzEz,size_16,color_FFFFFF,t_70#pic_center)

AI Stdio链接：https://aistudio.baidu.com/aistudio/projectdetail/753581

CSDN链接：https://blog.csdn.net/qq_38536313/article/details/108200653
