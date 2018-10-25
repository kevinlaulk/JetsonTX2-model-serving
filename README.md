# JetsonTX2-model-serving
something about JetsonTX2 and the appliaction of model-serving (Tensorflow) on it
# tensorflow serving使用指引及注意事项:

## 一、主机上的faster rcnn (res101 \ mobile) tensorflow serving存放位置：
> /home/spaci/liukai/tf-faster-rcnn-export_tensorflow_serving/

**其中保存的模型将存放于其子目录下**
##  2、tensorflow serving相关运行程序存放于：
> /home/spaci/liukai/tf-faster-rcnn-export_tensorflow_serving/tools/
**中**:
### 1、 其中

```
tf_faster_rcnn_client_for_jetson2.py------适用jetson上运行的客户端程序（能画图，单张，首选！！！）
tf_faster_rcnn_client_for_jetson3.py------适用jetson上运行的客户端程序（能画图，自动搜索默认路径下所有jpg并输入到模型中）
tf_faster_rcnn_client_for_jetson.py------适用jetson上运行的客户端程序 (稳定基础，不用)
tf_faster_rcnn_client_for_jetson1.py-------适用jetson上运行的客户端程序 (原版复制部分修改)
tf_faster_rcnn_client.py------适用主机上运行的客户端程序（最优秀）（能输出jetson格式）
export_tf_serving.py--------适用主机上运行的模型保存程序
```

### 2、注意：
> tf-faster-rcnn-export_tensorflow_serving

**文件内相关不能更改与替换，与其他faster rcnn版本代码有较多不同，主要为lib文件中网络定义部分全部使用tensorflow相关函数，不使用py_func自定义函数方能在客户端使用！**

### 三、jetson上的tensorlfow serving存放位置：
> /home/nvidia/TrainedModel/FasterRcnn/

**或**
 
>  /home/nvidia/tf-faster-rcnn-export_tensorflow_serving/ （host文件完整拷贝，理论上可导出res101 和 mobile网络，推荐）
### 1、其中

```
tf_serving_export_mobile_pascal_voc-------初始版本
tf_serving_export_mobile_pascal_voc_cpu-----导出模型全部指定为cpu
tf_serving_export_mobile_pascal_voc_forcegpu------导出模型强制指定为gpu（不可用）
tf_serving_export_mobile_pascal_voc_gpu------导出模型默认为gpu
tf_serving_export_mobile_pascal_voc_noconfig------导出模型时session无config，即 tensorflow_model_serving 开启时强制占用全部gpu (不推荐)
tf_serving_export_mobile_pascal_voc_onlygpu –------导出模型时较为合理指定gpu0 （推荐！！！）
tf_serving_export_res101_pascal_voc------res101版本不推荐
```

### 2、 客户端程序存放于：
> /home/nvidia/TrainedModel/FasterRcnn/Demo/

**或**

> /home/nvidia/tf-faster-rcnn-export_tensorflow_serving/tools/ (推荐)

**要运行**

```
tf_faster_rcnn_client_for_jetson2.py  -------  （可画图，推荐!!!!)
```

## 四、运行流程：
### 1、在主机端生成模型：

```
$tf-faster-rcnn-export_tensorflow_serving/
python ./tools/export_tf_serving.py
```

**在当前目录下生成模型，注意生成模型前需要删除同名文件夹否则会报错，注意在其下新建子文件夹下新建文件夹‘1’并将其两个文件移动至其中**
### 2、将生成模型拷贝至jetson中，存放目录任意$PATH (可存放于/tmp/)
### 3、运行tensorflow_server_model（位置任意）：

```
tensorflow_model_server --port=9000 –model_name=fasterrcnn –model_base_path=/tmp/tf_serving_export_mobile_pascal_voc_onlygpu/   （推荐！！！！不做任何修改！！）
```


```
tensorflow_model_server --port=9000 –model_name=fasterrcnn –model_base_path=/tmp/tf_serving_export_res101_pascal_voc/
```

### 4、运行客户端：

```
$ tf-faster-rcnn-export_tensorflow_serving/
python3 ./tools/tf_faster_rcnn_client.py  --server=localhost:9000
```
### 5、并列加载多个模型
使用配置文件的方式启动即可

首先编写模型配置文件，命名为models.json

```
model_config_list: {
  config: {
    name: "fasterrcnn_res",
    base_path: "/tmp/model",
    model_platform: "tensorflow"
  },
  config: {
     name: "fasterrcnn_res",
     base_path: "/tmp/model2",
     model_platform: "tensorflow"
  }
}
```

保存后，启动即可

```
tensorflow_model_server --model_config_file=/root/models.json --port=9000
```




## 五、待解决的问题：
- 1、为了能够解析结果，取消了一步直接生成最后（类别、预测概率、预测框）的方法，通过抽取中间预测结果进行分析，发现由于环境等问题，tensorflow不支持nms（即非极大抑制函数）需要重新编写，目前已能使用但结果与原来使用gpu_nms函数结果有部分差异，这部分可以通过理解nms及tensorflow gpunms解决

- 2、根据分析对比可知，在完全相同模型、相同客户端的情况下，host与jeston预测产生结果部分不同，对比所有结果可知丢失部分的结果包含了预测概率很高的结果，而这部分的不同导致了最终没有出现理想结果，此部分导致了host有结果而jeston无结果的现象，由图可知预测结果较低

- 3、加载模型model_server问题，难以启动，此部分无法解决，基本没有可参考帮助且非代码问题，
如果将jetson长时间放置能增加启动成功的概率！！

## 六、改进参考
==2018/02/24==

根据详细的分析结果可知：
![image](https://raw.githubusercontent.com/OldGentleman/picture/master/untitled.jpg)
![image](https://raw.githubusercontent.com/OldGentleman/picture/master/untitled1.jpg)

> 1、根据 bbox_pred 和 rois 产生的矩形框在主机端和jetson结果几乎完全一致。

![image](https://raw.githubusercontent.com/OldGentleman/picture/master/20180224140008572.jpg)

> 2、但 cls_prod 两者结果完全不一致，其中jetson产生的概率全部集中在backgound

造成的原因是由于 dropout 导致的，具体参考：

https://github.com/tensorflow/serving/issues/689

https://github.com/tensorflow/serving/issues/602

https://github.com/tensorflow/serving/issues/9
