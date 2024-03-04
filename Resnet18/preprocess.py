import os
import pandas as pd
import cv2
from tqdm import tqdm
from mtcnn import MTCNN


def get_files(path):
    file_info = os.walk(path)
    file_list = []
    for r, d, f in file_info:
        file_list += f
    return file_list
'''

get_files函数的作用是获取指定路径下的所有文件。它的工作原理和流程如下：

首先，它使用os.walk(path)函数获取指定路径下的所有目录和文件的信息。os.walk(path)函数会返回一个生成器，每次迭代都会返回一个元组，元组中包含三个元素：当前目录的路径，当前目录下的所有子目录名，和当前目录下的所有文件名。

然后，它定义了一个空列表file_list，用于存储所有文件名。

接着，它使用一个for循环遍历os.walk(path)返回的所有元组。在每次迭代中，它将当前目录下的所有文件名添加到file_list中。

最后，它返回file_list，即指定路径下的所有文件名。

'''

def get_dirs(path):
    file_info = os.walk(path)
    dirs = []
    for d, r, f in file_info:
        dirs.append(d)
    return dirs[1:]

'''

get_dirs函数的作用是获取指定路径下的所有子目录。它的工作原理和流程如下：

首先，它使用os.walk(path)函数获取指定路径下的所有目录和文件的信息。

然后，它定义了一个空列表dirs，用于存储所有子目录的路径。

接着，它使用一个for循环遍历os.walk(path)返回的所有元组。在每次迭代中，它将当前目录的路径添加到dirs中。

最后，它返回dirs[1:]，即指定路径下的所有子目录的路径。这里使用[1:]是因为os.walk(path)的第一个元素是指定路径本身，我们通常只关心其下的子目录，所以需要排除第一个元素。

'''



def generate_label_file():
    print('get label....')
    base_dirs = [
        'C:/Users/苏俊/Desktop/大创/AVEC/AEVC 2014/AVEC2014/dev/Freeform/frames',
        'C:/Users/苏俊/Desktop/大创/AVEC/AEVC 2014/AVEC2014/dev/Northwind/frames',
        'C:/Users/苏俊/Desktop/大创/AVEC/AEVC 2014/AVEC2014/test/Freeform/frames',
        'C:/Users/苏俊/Desktop/大创/AVEC/AEVC 2014/AVEC2014/test/Northwind/frames',
        'C:/Users/苏俊/Desktop/大创/AVEC/AEVC 2014/AVEC2014/train/Freeform/frames',
        'C:/Users/苏俊/Desktop/大创/AVEC/AEVC 2014/AVEC2014/train/Northwind/frames'
    ]
    label_base_url = 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/label/DepressionLabels/'
    labels = []
    for base_dir in base_dirs:
        img_files = get_files(base_dir)
        loader = tqdm(img_files)
        for img_file in loader:
            img_name = os.path.basename(img_file)
            video_id = img_name.split('_')[0]
            label_file = video_id + '_Depression.csv'
            label_path = os.path.join(label_base_url, label_file)
            if os.path.exists(label_path):
                label = pd.read_csv(label_path, header=None)
                labels.append([img_name, label[0][0]])
                loader.set_description(f'Processing {img_name}')
    
    output_path = 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/label/processed/label.csv'
    pd.DataFrame(labels, columns=['file', 'label']).to_csv(output_path, index=False)
    return labels
'''
1. `labels`是一个列表，其中每个元素都是一个包含图像文件名和对应标签的列表。`pd.DataFrame(labels, columns=['image', 'label'])`将这个列表转换为一个pandas DataFrame，其中每一行对应一个图像文件，有两列，分别是'image'（图像文件名）和'label'（对应的标签）。`df.to_csv('labels.csv', index=False)`将这个DataFrame保存为一个CSV文件，文件名为'labels.csv'。`index=False`表示在保存时不包含行索引。这样，生成的CSV文件的每一行就包含一个图像文件的名称和对应的标签。

2. `base_dirs`是一个包含所有图像文件所在目录的路径的列表。这个列表是预先定义的，包含了所有需要处理的图像文件的路径。`label_base_url`是标签文件所在的基础路径，用于构造每个图像文件对应的标签文件的路径。

3. 对于每个图像文件，首先根据其文件名和`label_base_url`构造出对应的标签文件的路径。然后，检查这个标签文件是否存在。如果存在，那么使用`np.loadtxt(label_path)`读取这个标签文件，得到一个numpy数组，然后取出数组的第一个元素作为标签值。然后，将图像文件名和标签值作为一个列表添加到`labels`中。这样，`labels`就包含了所有图像文件的名称和对应的标签。
'''


def generate_img(path, v_type, img_path):
    videos = get_files(path)
    loader = tqdm(videos)
    for video in loader:
        name = video[:5]
        save_path = img_path + v_type + '/' + name
        os.makedirs(save_path, exist_ok=True)
        cap = cv2.VideoCapture(path + video)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        gap = int(n_frames / 100)
        for i in range(n_frames):
            success, frame = cap.read()
            if success and i % gap == 0:
                cv2.imwrite(save_path + '/{}.jpg'.format(int(i / gap)), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                loader.set_description("data:{} type:{} video:{} frame:{}".format(path.split('/')[2], v_type, name, i))
        cap.release()


def get_img():
    print('get video frames....')
    train_f = 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/train/Freeform/'
    train_n = 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/train/Northwind/'
    test_f = 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/test/Freeform/'
    test_n = 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/test/Northwind/'
    validate_f = 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/dev/Freeform/'
    validate_n = 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/dev/Northwind/'
    dirs = [train_f, train_n, test_f, test_n, validate_f, validate_n]
    types = ['Freeform', 'Northwind', 'Freeform', 'Northwind', 'Freeform', 'Northwind']
    img_path = ['C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/train/Freeform/', 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/train/Northwind/', 
                'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/test/Freeform/', 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/test/Northwind/', 
                'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/validate/Freeform/', 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/validate/Northwind/']
    os.makedirs('C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/train', exist_ok=True)
    os.makedirs('C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/test', exist_ok=True)
    os.makedirs('C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/validate', exist_ok=True)
    for i in range(6):
        generate_img(dirs[i], types[i], img_path[i])


def get_face():
    print('get frame faces....')
    detector = MTCNN()
    save_path = ['C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/processed/train/Freeform/', 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/processed/train/Northwind/', 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/processed/test/Freeform/',
                 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/processed/test/Northwind/', 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/processed/validate/Freeform/', 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/processed/validate/Northwind/']
    paths = ['C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/train/Freeform/', 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/train/Northwind/',
            'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/test/Freeform/', 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/test/Northwind/',
             'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/validate/Freeform/', 'C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img/validate/Northwind/']
    for index, path in enumerate(paths):
        dirs = get_dirs(path)
        loader = tqdm(dirs)
        for d in loader:
            os.makedirs(save_path[index] + d.split('/')[-1], exist_ok=True)
            files = get_files(d)
            for file in files:
                img_path = d + '/' + file
                s_path = save_path[index] + d.split('/')[-1] + '/' + file
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                info = detector.detect_faces(img)
                if (len(info) > 0):
                    x, y, width, height = info[0]['box']
                    confidence = info[0]['confidence']
                    b, g, r = cv2.split(img)
                    img = cv2.merge([r, g, b])
                    img = img[y:y + height, x:x + width, :]
                    cv2.imwrite(s_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    loader.set_description('confidence:{:4f} img:{}'.format(confidence, img_path))


if __name__ == '__main__':
    os.makedirs('C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/img', exist_ok=True)
    os.makedirs('C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/processed', exist_ok=True)
    os.makedirs('C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/processed/train', exist_ok=True)
    os.makedirs('C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/processed/test', exist_ok=True)
    os.makedirs('C:/Users/苏俊/Desktop/Depression-detect-ResNet-main/AVEC2014/processed/validate', exist_ok=True)
    label = generate_label_file() #将运来的label合并为一个csv文件
    get_img()                     #抽取视频帧，每个视频按间隔抽取100-105帧
    get_face()                    #使用MTCNN提取人脸，并分割图片




