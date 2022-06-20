import torch
import os,random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


class Iconclass(Dataset):
    def __init__(self, root, mode,transform=None):
        super(Iconclass, self).__init__()
        # 文件根路径
        self.root = root
        
        self.transform = transform

        # 打标签
        # self.name2label = {}
        # for name in sorted(os.listdir((os.path.join(root)))):
        #     if not os.path.isdir(os.path.join(root, name)):
        #         continue
        #     self.name2label[name] = len(self.name2label.keys())

        # 加载文件路径与标签
        
        with open(os.path.join(self.root,'data for classify-final.json'),'r') as load_f:
            load_dict = json.load(load_f)
        
        self.images=[(load_dict[i]['image'],load_dict[i]['label']) for i in load_dict]
        label=[]
        for i in self.images:
            label.append(i[1])
        
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(label)
        self.classes=self.mlb.classes_
        
        # keys_train=random.sample(self.images.keys(), int(0.8 * len(self.images)))
        # keys_=list(set(self.images.keys()) - set(keys_train))
        # keys_val=random.sample(keys_, int(0.5 * len(keys_)))
        # keys_test=list(set(keys_) - set(keys_val))
        
        self.mode=mode
        # 按作用取数据集
        if mode == 'train':  
            self.images=self.images[:int(0.8*len(self.images))]
        elif mode == 'val': 
            i=random.randint(0,len(self.images))
            # self.images=self.images[int(0.8*len(self.images)):int(0.9*len(self.images))]
            self.images=self.images[i:i+1]
            # print("aaaaa",self.images[0][0])
            # print("aaaaa",self.images[0][1])
            
        else:  
            self.images=self.images[int(0.9*len(self.images)):]

    # 数据集大小
    def __len__(self):
        # if self.mode == 'train':
        #     return len(self.train)
        # elif self.mode=='val':
        #     return len(self.val)
        # elif self.mode=='test':
        #     return len(self.test)
        return len(self.images)

    # 取图片路径与标签
    def __getitem__(self, idx):
        label = self.images[idx][1]
        image = Image.open(os.path.join(self.root,self.images[idx][0])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # print([label))
        label=self.mlb.transform([label])
        # print(type(label))
        return image, np.squeeze(label).astype(np.float16)

# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from torchvision.transforms import InterpolationMode
def main():
    path=r'C:\Users\82439\Desktop\dataset\iconclass'
    t = []

    t.append(
        transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
        # to maintain same ratio w.r.t. 224 images
    )
            

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transformer=transforms.Compose(t)
    db = Iconclass(path,  'train',transformer)
    # 用迭代器迭代，调用了getitem方法
    # print(len(db.classes))
    # for i in range(10):
    #     x, y = next(iter(db))
    #     print('sample:', x.shape, y.shape)
    # sample: torch.Size([3, 224, 224]) torch.Size([]) tensor(7)
    # print(db.mlb.transform([['57A6', '57B2', '41D266', '86(...)', '11Q01', '11L222', '41A711', '25D11']]).shape)

    # db = torchvision.datasets.ImageFolder(root='ADS-B', transform=tf)
    # print(db.class_to_idx)

    # 使用DataLoader加载数据集
    loader = DataLoader(db, batch_size=32, shuffle=True)
    # loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8) # 多线程取数据
    # len of loader: 75 = 400 * 10 * 0.6 / 32 = batch数量
    print('len of loader:', len(loader))
    for i in loader:
        print(i[1].shape)
        break
    


if __name__ == '__main__':
    main()
    