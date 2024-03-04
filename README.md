## 面部识别模型  

### 2.1  数据源      

本项目面部识别所使用的数据集是AVEC2013/2014。AVEC（Audio/Visual recognition Challenge）2013数据集是一个用于情感分析的大型语料库，由英国普利茅斯大学的心理语言学研究中心收集。它包含超过400名受试者的情绪变化数据，其中包括文本和音视频数据。AVEC2014包含来自不同国家和文化背景的108名受试者的音视频和文本数据，其中受试者观看了一些激发情感的视频剪辑，实验人员记录了他们的情感状态。数据通过Webcam和麦克风记录。  

### 2.1.1  数据预处理      

为得到高质量的面部数据，需要对数据源中的视频作预处理。预处理主要为以下几个步骤：  

（1）**视频抽帧：**  由于大部分情况下，视频相邻帧的表情是几乎不变的，因此为了获取有效变化的表情图像，需要对视频进行抽帧。本项目所使用的工具是FFmpeg，一套用于记录和转换数字音频、视频的开源程序，它提供了视频抽帧的解决方案，保证了高移植性和编解码质量。使用FFmpeg工具，将帧率设定为15fps，视频被解码并保存为多帧jpg格式的图片。  

（2）**检测和对齐人脸：**  得到包含人像的视频帧后，需要去除图片集中由于遮挡、人像姿态变化、大块背景以及无意识运动所带来的影响。这些影响因素往往会降低特征网络的提取效果,甚至降低最终的识别精度。因此本项目选择多任务级联卷积神经网络（Multi-task  Cascaded Convolutional Networks, MTCNN）来同时检测和对齐人脸。  


MTCNN的结构如下：

![image-20240304110122468](C:\Users\苏俊\AppData\Roaming\Typora\typora-user-images\image-20240304110122468.png)

图3-1 MTCNN结构图     

NMS：非极大值抑制  

  其中：     Resize：将原始图像缩放成不同的尺度，生成图像金字塔（Image  Pyramid）。然后将不同尺度的图像送入到三个子网络中训练，目的是为了可以检测到不同大小的人脸，从而实现多尺度目标检测。     

P-Net（Proposal Network, 建议网络）：P-Net得到人脸区域的候选网络，给出了人脸区域的候选框和边界框的回归向量。基于预测边界框回归向量对候选框进行矫正，并用NMS来合并重叠率高的候选框。

 使用python实现P-Net的代码如下：  ![image-20240304110622554](C:\Users\苏俊\AppData\Roaming\Typora\typora-user-images\image-20240304110622554.png)

  **图3-2 P-Net的代码**  

其中stage1代表MTCNN的第一步。  R-Net（Refine Network, 细化网络）：将P-Net中得到的所有候选框都输入到R-Net中，继续通过边界框回归和NMS去除大量false-positive（FP）区域。  同上文，使用python实现R-Net：  

![image-20240304110721254](C:\Users\苏俊\AppData\Roaming\Typora\typora-user-images\image-20240304110721254.png)

O-Net（Output Network，输出网络）：相比于上一层，O-Net多一层卷积层，因此处理的结果会更加精细。输出的结果包括N个边界框的坐标信息、score和关键点位置。        

以下是python实现O-Net的函数：  

![image-20240304110741419](C:\Users\苏俊\AppData\Roaming\Typora\typora-user-images\image-20240304110741419.png)

**图3-4 O-Net**     

在损失函数方面，对于人脸识别，直接使用交叉熵代价函数；对于框回归和关键点定位，使用L2损失。最后把这三部分的损失各自乘以自身的权重累加起来，形成最后的总损失。其中各个损失函数的权重是不同的。     

### 2.2  注意力机制提取关键点属性      

传统人脸识别方法将整张人脸图像输入网络中进行计算，预测表情状态。这 种方法通常情况下，都会受到图像中非表情区域的干扰，很难获得最佳的识别效果。本项目使用基于注意力的人脸关键点属性表征的特征描述符，通过CNN回归获得人脸中的关键点以及对应的关键点特征向量，然后通过Transformer模块进行编码，从而进行表情状态的识别。     

### 2.2.1  通道注意力模块      

CNN网络在逐层运算中，通道数会有所增加，产生信息冗余。为了解决这一问题，本项目使用通道注意力模块，应用不同的池化策略压缩输入特征的空间维度。通道注意力模块的结构如下图：  

![image-20240304110937538](C:\Users\苏俊\AppData\Roaming\Typora\typora-user-images\image-20240304110937538.png)



![image-20240304111108800](C:\Users\苏俊\AppData\Roaming\Typora\typora-user-images\image-20240304111108800.png)

4-1  通道注意力结构图

element-wise  addition：逐项相加  

matrix  multiplication：矩阵乘法  

Maxpool：最大池化  

AvgPool：平均池化  

Channel  Attention：通道注意力  

Channel-Refined  Feature：通道精细化特征     

  使用通道注意力主要分为以下几个步骤：     

​     （1）将大小为![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image002.gif)的输入特征![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image004.gif)馈送至通道注意力块，对![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image004.gif)沿空间轴并行进行全局平均池化和最大值池化操作得到两个特征向量![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image006.gif)和![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image008.gif)，通过对两个向量逐项求和得到具有特征聚合属性的![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image010.gif)，并用![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image012.gif)的卷积运算处理之，随后执行PReLU和BatchNorm操作，得到特征图![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image014.gif)。  

（2）去掉![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image014.gif)的冗余维度，对其进行转置，得到大小为![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image016.gif)和![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image018.gif)的两个特征图，将他们相乘得到![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image020.gif)矩阵，进而通过softmax运算得到最终的注意力矩阵![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image022.gif)。  

（3）最后，将输入矩阵![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image004.gif)与通道注意力矩阵![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image022.gif)相乘，并经过残差学习得到通道精细后的特征![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image024.gif)，大小为![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image002.gif)。  通道注意力矩阵可以看作是一个选择器，筛选出捕获人脸特征的最佳滤波器。   

### 2.2.2  空间注意力模块  

​       在人脸中，五官的位置是有一定空间位置关系的，故不同的位置具有不同的重要性，但在传统的方法中，卷积核对它们的处理是相同的。本项目在通道注意力的基础上，加入空间注意力模块与之结合，可以同时获得重要的通道特征及特征之间的关系，从而使得到的特征图更加细化。空间注意力模块的结构如下图所示：      

![image-20240304111500969](C:\Users\苏俊\AppData\Roaming\Typora\typora-user-images\image-20240304111500969.png)

  图4-2 ![图示  低可信度描述已自动生成](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image002.gif)   通道注意力结构图  

element-wise  addition：逐项相加  

matrix  multiplication：矩阵乘法  

Maxpool：最大池化  

AvgPool：平均池化  

Channel  Attention：通道注意力  

Channel-Refined  Feature：通道精细化特征  

  使用空间注意力主要分为以下几个步骤：  

  （1）将通道细化特征![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image002.gif)馈送至模块，对其沿通道轴并行采取全局平均池化和最大池化操作，得到两个尺寸均为![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image004.gif)的特征向量![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image006.gif)和![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image008.gif)。采用通道级联的方式合并构成聚合特征![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image010.gif)。然后使用![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image012.gif)卷积作用于![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image010.gif)，卷积步长和填充值均设置为1。进行PReLU和BatchNorm运算得到中间特征图![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image014.gif)。  

（2）对![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image014.gif)进行维度变换，将其维度转换为![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image016.gif)，再进行转置得到![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image018.gif)的特征图。对两者执行矩阵乘法和softmax运算，得到空间注意矩阵![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image020.gif)。  

（3）最后将通道细化特征![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image002.gif)与空间注意矩阵![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image020.gif)相乘，经过残差学习得到空间精细后的特征![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image022.gif)。  现在特征图已能够获取到人脸中具有空间关系的重要特征。  

### 2.2.3  基于注意力机制的神经网络（CS-ResNet）总结构      

将上文中的通道注意力和空间注意力附加到每个残差块后，可以得到基于注意力机制的神经网络（CS-ResNet）的残差结构如下图所示：                             

![image-20240304112752468](C:\Users\苏俊\AppData\Roaming\Typora\typora-user-images\image-20240304112752468.png)                

图4-3 CS-ResNet残差网络结构  

Channel Att：通道注意力机制     

Spatial Att：空间注意力机制     

Conv layer：卷积层     

Relu：激活函数        

经过训练，输出端可以获得关键点的位置坐标，也可以得到每个点的特征属性，为使用Transformer模块去融合关键点之间的相关特征奠定基础。     

### 2.3  使用Transformer框架识别表情  

![image-20240304112801902](C:\Users\苏俊\AppData\Roaming\Typora\typora-user-images\image-20240304112801902.png)      

Transformer框架结构如下图所示：  

图4-4   Transformer框架结构图  

MLP：多层感知机  

Position Embedding：位置编码  

Cross-entropy Loss：交叉熵损失  

Multi-Head Attention：多头注意力模块  

Norm：归一化操作     

### 2.3.1  网络框架      

标准Transformer的输入是一维的词嵌入向量，通过获取网络全连接层512维的向量，参照ViT（Vision  Transformer）中的参数设置，利用可训练的线性投影变换将原始特征映射为768维的向量，图中称为Embedded Keypoints。Transformer层由Multi-Head Attention（多头注意力）和Multi-Layer  Perception（MLP，多层感知机）块组成。并且在每个块之前进行Norm（归一化）操作，每个块之后使用残差连接。     

### 2.3.2  多头自注意力机制      

根据注意力机制原理，模型在对当前位置的信息进行编码时，会过度地将注意力集中于自身的位置，而忽略了其他位置。因此在Transformer Layer中，加入了多头注意力机制。其机制实质是将原始输入序列进行多组自注意力处理过程，再将每一组结果进行拼接、线性变换的结果，这是Transformer的核心部分。     

### 2.3.3  位置编码      

  与传统CNN不同，Transformer需要位置编码来编码每个词向量的位置信息。为了更好区分不同位置的关键点之间的差异以便更好地训练，本项目将位置编码加入Transformer Layer底部的输入词嵌入中（即结构图中的Position  Embedding）。位置编码与关键点嵌入向量具有相同的维度，将二者相加，并使用不同频率的正弦和余弦函数构造Position Embedding，用公式表示：  ![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image002.gif)  其中：  ![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image004.gif)  ![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image006.gif)

表示特征点在序列中的位置，![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image008.gif)为关键点序列长度。![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image010.gif)为线性变换后嵌入向量的维度，![img](file:///C:/Users/苏俊/AppData/Local/Temp/msohtmlclip1/01/clip_image012.gif)表示嵌入向量的位置。每个位置对应一个正弦信号，加入经过线性映射后的输入向量中，实现位置编码的引入。  