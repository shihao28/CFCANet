"""
Copy CompCarsWeb train images to another dir
"""
# import os
# from shutil import copy
# from tqdm import tqdm


# root = r'D:\OneDrive - PETRONAS\Exercise\data\CompCars\web_nature'
# with open(os.path.join(root, 'train_test_split', 'classification_train.txt')) as f:
#     train_images = f.readlines()
# for i in tqdm(train_images):
#     makemodel_dir = os.path.join('D:/train', '_'.join(i.split('/')[:2]))
#     os.makedirs(makemodel_dir, exist_ok=True)
#     copy(os.path.join(root, 'image', i).strip(), makemodel_dir)


"""
Get CompCarsWeb vehicle model name
"""
# import os
# import scipy
# from scipy import io


# root = '../yolov5-master/data/CompCars/web_nature'
# make_name = scipy.io.loadmat(f'{root}/misc/make_model_name.mat')['make_names']
# model_name = scipy.io.loadmat(f'{root}/misc/make_model_name.mat')['model_names']
# type_name = scipy.io.loadmat(f'{root}/misc/car_type.mat')
# modelid_to_makeid_mapping = dict()
# modelid_to_filename_mapping = dict()
# for makeid in os.listdir(f'{root}/image'):
#     for modelid in os.listdir(f'{root}/image/{makeid}'):
#         modelid_to_makeid_mapping[int(modelid)] = int(makeid)

#         filename = []
#         for _, _, files in os.walk(f'{root}/image/{makeid}/{modelid}'):
#             filename.extend(files)
#         modelid_to_makeid_mapping[int(modelid)] = filename

# # Get make name from make id
# make_id = 1  # start from 1, it's the name of dir
# make_name[make_id-1]

# # Get model name from model id
# model_id = 1327  # start from 1, it's the name of dir
# model_name[model_id-1]

# # Get makeid from model id
# model_id = 1  # start from 1, it's the name of dir
# modelid_to_makeid_mapping[model_id]

"""
Copy Stanford Car train images to another dir
"""
# import os
# import scipy
# from scipy import io
# from shutil import copy


# root = r'D:\OneDrive - PETRONAS\Exercise\data\stanford-car'
# img_path_ori = scipy.io.loadmat(os.path.join(root, 'devkit', 'cars_train_annos.mat'))['annotations'][0]
# for img_path_ori_tmp in img_path_ori:
#     filename = img_path_ori_tmp[-1].item()
#     modelid = img_path_ori_tmp[-2].item() - 1
#     makemodel_dir = f'D:/sc/train/{modelid}'
#     os.makedirs(makemodel_dir, exist_ok=True)
#     copy(os.path.join(root, 'cars_train', filename), makemodel_dir)

"""
Get Stanford Car vehicle model name
"""
# import scipy
# from scipy import io
# import numpy as np


# cars_meta = scipy.io.loadmat('../yolov5-master/data/stanford-car/devkit/cars_meta.mat')['class_names'][0]
# model_name = scipy.io.loadmat('../yolov5-master/data/stanford-car/devkit/cars_test_annos_withlabels.mat')['annotations'][0]
# filename_to_modelid_mapping = dict()
# modelid_to_filename_mapping = dict()
# for model_name_tmp in model_name:
#     filename = model_name_tmp[-1].item()
#     modelid = model_name_tmp[-2].item() - 1
#     filename_to_modelid_mapping[filename] = modelid
#     if modelid_to_filename_mapping.get(modelid) is None:
#         modelid_to_filename_mapping[modelid] = [filename]
#     else:
#         modelid_to_filename_mapping[modelid].append(filename)

# # Get model name from filename
# filename = '00001.jpg'
# cars_meta[filename_to_modelid_mapping[filename]]

# # Get filename from model id
# # modelid starts from 0
# modelid = 100
# modelid_to_filename_mapping[modelid]

# # Get model id from model name
# model_name = 'Land Rover Range Rover SUV 2012'
# modelid = np.where(cars_meta==model_name)[0].item()


"""
Get image that have bbox wrongly labeled
"""
# import os
# import cv2
# from tqdm import tqdm


# with open('/data/CompCars/web_nature/train_test_split/classification_test.txt', 'r') as f:
#     label_paths = f.readlines()

# wrong_img_paths = []
# wrong_bboxes = []
# for label_path in tqdm(label_paths):
#     label_path = label_path.replace('jpg', 'txt').strip()
#     label_path = os.path.join('/data/CompCars/web_nature/label', label_path)
#     with open(label_path, 'r') as f:
#         label = f.readlines()
#     bbox = label[2]
#     x1, y1, x2, y2 = bbox.strip().split()
#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#     diff_x = x2 - x1
#     diff_y = y2 - y1
#     if diff_x == 0 or diff_y == 0:
#         wrong_img_paths.append(label_path.replace('label', 'image').replace('txt', 'jpg'))
#         wrong_bboxes.append([x1, y1, x2, y2])
# print('Wong labels count: ', len(wrong_img_paths))

# for wrong_img_path, wrong_bbox in zip(wrong_img_paths, wrong_bboxes):
#     img = cv2.imread(wrong_img_path)
#     x1, y1, x2, y2 = wrong_bbox
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
#     cv2.namedWindow(wrong_img_path, cv2.WINDOWNORMAL)
#     cv2.imshow(wrong_img_path, img)
#     cv2.waitKey()


"""
GNN Example
"""
# import torch
# from torch_geometric.data import Data


# x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
# y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

# edge_index = torch.tensor([[0, 2, 1, 0, 3],
#                            [3, 1, 0, 1, 2]], dtype=torch.long)


# data = Data(x=x, y=y, edge_index=edge_index)

