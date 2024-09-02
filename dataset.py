import numpy as np
from einops import rearrange
# from decord import VideoReader
from PIL import Image
import os
import os.path as osp

import torch
import torchvision.transforms as transforms
import torch.utils.data as data


def get_transform():
    image_transforms = transforms.Compose(
        [
        transforms.Resize((512, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    return image_transforms


class FashionDataset(data.Dataset):
    def __init__(self,
                 video_data_root,
                 video_tensor_root,
                 poses_data_root,
                 motion_text_root,
                 frameLimit=16):
        super(FashionDataset, self).__init__()

        self.video_tensor_root   = video_tensor_root
        self.video_data_root = video_data_root

        self.poses_data_root   = poses_data_root
        self.motion_text_root  = motion_text_root

        self.frameLimit = frameLimit
        self.dataset = list()
        for txts in os.listdir(motion_text_root):
            txt_path = os.path.join(motion_text_root, txts)
            video_name = txts.split('.')[0]

            with open(txt_path, 'r') as t:
                for line in t:
                    line = line.strip().split()
                    # start = line[0:1]
                    # end = line[1:2]
                    start = line[0]
                    end = line[1]
                    if (int(end) - int(start) < self.frameLimit): 
                        # print(f"{video_name} : {start} - {end}")
                        # a += 1
                        continue

                    txt = " ".join(line[2:])
                    self.dataset.append([video_name, start, end, txt])

        self.transform = get_transform()
        print(len(self.dataset))

    def __getitem__(self, index):

        # 读取数据
        data = self.dataset[index]

        # 文本
        mt_txt = data[-1]

        # 视频
        video = torch.load(osp.join(self.video_tensor_root, data[0] + ".pt"), map_location='cpu')

        start, end = int(data[1]), int(data[2])

        # start = np.random.randint(start, end-15)

        img_path = osp.join(self.video_data_root, data[0] + "/" + str(start).zfill(3) + '.png')
        assert os.path.isfile(img_path), "img_path is not exists!!!"

        image = Image.open(osp.join(self.video_data_root, data[0] + "/" + str(start).zfill(3) + '.png'))
        # if not image.mode == "RGB":
        #     image = image.convert("RGB")
        image = self.transform(image)
        
        # pose
        pose = []
        for i in range(start, start + self.frameLimit):

            path = osp.join(self.poses_data_root, data[0] + "/" + str(i).zfill(3) + '.png')
            assert os.path.isfile(path), "pose_path is not exists!!!"

            p = Image.open(path)
            # if not pose.mode == "RGB":
            #     pose = pose.convert("RGB")
            p = self.transform(p)
            p = p.unsqueeze(0)
            pose.append(p)

        pose = torch.cat(pose, 0)

        video = video[start:start+self.frameLimit, :, :, :]


        return {"mt_txt": mt_txt, "video": video, "pose": pose, "image": image}


    def __len__(self):
        return len(self.dataset)
    


class FashionTestDataset(data.Dataset):
    def __init__(self,
                 video_data_root,
                 video_tensor_root,
                 poses_data_root,
                 motion_text_root,
                 frameLimit=16):
        super(FashionTestDataset, self).__init__()

        self.video_tensor_root   = video_tensor_root
        self.video_data_root = video_data_root

        self.poses_data_root   = poses_data_root
        self.motion_text_root  = motion_text_root

        self.frameLimit = frameLimit
        self.dataset = list()
        for txts in os.listdir(motion_text_root):
            txt_path = os.path.join(motion_text_root, txts)
            video_name = txts.split('.')[0]

            with open(txt_path, 'r') as t:
                for line in t:
                    line = line.strip().split()
                    # start = line[0:1]
                    # end = line[1:2]
                    start = line[0]
                    end = line[1]
                    if (int(end) - int(start) < self.frameLimit): 
                        # print(f"{video_name} : {start} - {end}")
                        # a += 1
                        continue

                    txt = " ".join(line[2:])
                    self.dataset.append([video_name, start, end, txt])

        self.transform = get_transform()
        print(len(self.dataset))

    def __getitem__(self, index):

        # 读取数据
        data = self.dataset[index]

        # 文本
        mt_txt = data[-1]

        # 视频
        # video = torch.load(osp.join(self.video_tensor_root, data[0] + ".pt"), map_location='cpu')

        start, end = int(data[1]), int(data[2])

        # start = np.random.randint(start, end-15)

        img_path = osp.join(self.video_data_root, data[0] + "/" + str(start).zfill(3) + '.png')
        assert os.path.isfile(img_path), "img_path is not exists!!!"

        image = Image.open(osp.join(self.video_data_root, data[0] + "/" + str(start).zfill(3) + '.png'))
        # if not image.mode == "RGB":
        #     image = image.convert("RGB")
        image = self.transform(image)
        
        # pose
        pose = []
        for i in range(start, start + self.frameLimit):

            path = osp.join(self.poses_data_root, data[0] + "/" + str(i).zfill(3) + '.png')
            assert os.path.isfile(path), "pose_path is not exists!!!"

            p = Image.open(path)
            # if not pose.mode == "RGB":
            #     pose = pose.convert("RGB")
            p = self.transform(p)
            p = p.unsqueeze(0)
            pose.append(p)

        pose = torch.cat(pose, 0)

        # video = video[start:start+self.frameLimit, :, :, :]


        # return {"mt_txt": mt_txt, "video": video, "pose": pose, "image": image}
        return {"name": data[0], "mt_txt": mt_txt, "pose": pose, "image": image, "start": start, "end": end}

    def __len__(self):
        return len(self.dataset)



if __name__ == "__main__":

    dataset = FashionTestDataset(
        video_data_root="fashion_dataset/fashion_crop",
        video_tensor_root="fashion_dataset_tensor/fashion_crop",
        poses_data_root="fashion_dataset/fashion_poses",
        motion_text_root="fashion_dataset/test",
        frameLimit=24
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,)
    print(len(dataloader))

    for i, batch in enumerate(dataloader):
 
        # control = torch.stack(batch["video"], dim=0)
        
        print(i,": Image's shape:", batch["image"].shape, "Poses's shape:", batch['pose'].shape)
        
        pose = batch['pose'][0:1, :, :, :, :]
        # save_video_frames_as_mp4(batch['pose'], 24, "pose_video.mp4")
        print(batch['mt_txt'][0])
        print(batch['name'][0])
