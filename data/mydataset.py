# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:32:29 2021

@author: tan.shihao
"""

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, Sampler, BatchSampler
import scipy
from scipy import io
from tqdm import tqdm
import numpy as np
from PIL import Image
import json
from datetime import datetime
import pandas as pd


class MyDataset(Dataset):
    def __init__(
        self, root, is_train, use_bbox, use_model, use_make, 
        use_type, use_viewpoint, transform=None,
        **kwargs):
        assert isinstance(is_train, bool)
        assert isinstance(use_bbox, bool)
        assert isinstance(use_model, bool)
        assert isinstance(use_make, bool)
        assert isinstance(use_type, bool)
        assert isinstance(use_viewpoint, bool)
        
        self.use_bbox = use_bbox
        self.use_model = use_model
        self.use_make = use_make
        self.use_type = use_type
        self.use_viewpoint = use_viewpoint

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform
        
        img_path = []
        label_model = []
        label_make = []
        bbox = []
        label_viewpoint = []
        label_type = []

        if root.find('web') != -1:
            file = open(os.path.join(root, 'train_test_split', 'typemodel.txt'), 'r')
            vmmr_to_vtr_raw = file.readlines()[2:]
            file.close()
            vmmr_to_vtr = dict()
            for vmmr_to_vtr_raw_tmp in vmmr_to_vtr_raw:
                vmmr_label, vtr_label = vmmr_to_vtr_raw_tmp.strip().split(',')
                vmmr_to_vtr[int(vmmr_label)] = int(vtr_label)
            if "combine_vtr_label" in kwargs:
                if kwargs["combine_vtr_label"]:
                    vtr_annot_simple = {
                        1:1, 2:1, 3:2, 4:1, 5:3, 6:5, 7:1, 8:4, 9:5,
                        10:5, 11:1, 12:5}
            if is_train:
                file = open(os.path.join(root, 'train_test_split', 'classification_train.txt'))
                img_path_ori = file.readlines()
                file.close()
                if kwargs.get("small_dataset"):
                    file = open(os.path.join(root, 'train_test_split', 'classification', 'train.txt'))
                    img_path_ori = file.readlines()
                    file.close()
                if os.path.exists(os.path.join(root, 'class_mapping_model_webnature.npy')):
                    print('Loading class mapping file...')
                    class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_webnature.npy'),
                                                  allow_pickle=True).flatten()[0]
                    class_mapping_make = np.load(os.path.join(root, 'class_mapping_make_webnature.npy'),
                                                  allow_pickle=True).flatten()[0]
                    save_class_mapping = False
                else:
                    print('Class mapping file not found...')
                    class_mapping_model = dict()
                    class_mapping_make = dict()
                    save_class_mapping = True
                
                for i in tqdm(img_path_ori):
                    # Get viewpoint
                    file = open(os.path.join(root, 'label', i.replace('jpg', 'txt').strip()))
                    label_viewpoint_tmp = int(file.readlines()[0].strip())
                    file.close()

                    # -1 means unknown viewpoint, these images are poorly captured
                    # Skip these images
                    if kwargs.get("remove_poor_images") and label_viewpoint_tmp == -1:
                        # label_viewpoint_tmp = 0
                        continue
                    
                    img_path.append(os.path.join(root, 'image', i.strip()))

                    model = '/'.join(i.split('/')[:2])
                    make = i.split('/')[0]
                    label_model_tmp = class_mapping_model.get(model)
                    label_make_tmp = class_mapping_make.get(make)
                    if label_model_tmp is None:
                        class_mapping_model[model] = len(class_mapping_model) + 1
                        label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                    if label_make_tmp is None:
                        class_mapping_make[make] = len(class_mapping_make) + 1
                        label_make_tmp = class_mapping_make.get(make)
                    label_make.append(label_make_tmp)
                    
                    vtr_type = int(model.split('/')[1])                    
                    label_type_tmp = vmmr_to_vtr[vtr_type]
                    if "combine_vtr_label" in kwargs:
                        if kwargs["combine_vtr_label"]:
                            label_type_tmp = vtr_annot_simple[label_type_tmp]
                    label_type.append(label_type_tmp)
                    
                    if use_bbox:
                        file = open(os.path.join(root, 'label', i.replace('jpg', 'txt').strip()))
                        xmin, xmax, ymin, ymax = file.readlines()[-1].strip().split()
                        file.close()
                        bbox.append((int(xmin), int(xmax), int(ymin), int(ymax)))
                    
                    if use_viewpoint:
                        label_viewpoint.append(label_viewpoint_tmp)
                        
                if save_class_mapping:
                    print('Saving class mapping file...')
                    np.save(os.path.join(root, 'class_mapping_model_webnature.npy'), class_mapping_model)
                    np.save(os.path.join(root, 'class_mapping_make_webnature.npy'), class_mapping_make)

            elif not is_train:
                file = open(os.path.join(root, 'train_test_split', 'classification_test.txt'))
                img_path_ori = file.readlines()
                file.close()
                if kwargs.get("small_dataset"):
                    file = open(os.path.join(root, 'train_test_split', 'classification', 'test.txt'))
                    img_path_ori = file.readlines()
                    file.close()
                assert os.path.exists(os.path.join(root, 'class_mapping_model_webnature.npy')), 'Class mapping (model) file not found...'
                assert os.path.exists(os.path.join(root, 'class_mapping_make_webnature.npy')), 'Class mapping (make) file not found...'
                class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_webnature.npy'),
                                                            allow_pickle=True).flatten()[0]
                class_mapping_make = np.load(os.path.join(root, 'class_mapping_make_webnature.npy'),
                                                            allow_pickle=True).flatten()[0]
                
                for i in tqdm(img_path_ori):
                    # Get viewpoint
                    file = open(os.path.join(root, 'label', i.replace('jpg', 'txt').strip()))
                    label_viewpoint_tmp = int(file.readlines()[0].strip())
                    file.close()

                    # -1 means unknown viewpoint, these images are poorly captured
                    # Skip these images
                    if kwargs.get("remove_poor_images") and label_viewpoint_tmp == -1:
                        # label_viewpoint_tmp = 0
                        continue
                    
                    img_path.append(os.path.join(root, 'image', i.strip()))

                    model = '/'.join(i.split('/')[:2])
                    make = i.split('/')[0]
                    label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                    label_make_tmp = class_mapping_make.get(make)
                    label_make.append(label_make_tmp)
                    
                    vtr_type = int(model.split('/')[1])                  
                    label_type_tmp = vmmr_to_vtr[vtr_type]
                    if "combine_vtr_label" in kwargs:
                        if kwargs["combine_vtr_label"]:
                            label_type_tmp = vtr_annot_simple[label_type_tmp]
                    label_type.append(label_type_tmp)
                    
                    if use_bbox:
                        file = open(os.path.join(root, 'label', i.replace('jpg', 'txt').strip()))
                        xmin, xmax, ymin, ymax = file.readlines()[-1].strip().split()
                        file.close()
                        bbox.append((int(xmin), int(xmax), int(ymin), int(ymax)))
                    
                    if use_viewpoint:
                        label_viewpoint.append(label_viewpoint_tmp)

        elif root.find('sv_data') != -1:
            if use_bbox or use_type or use_viewpoint:
                raise Exception('No such annotation')
            annotation = scipy.io.loadmat(os.path.join(root, 'sv_make_model_name.mat'))['sv_make_model_name']
            if is_train:
                file = open(os.path.join(root, 'train_surveillance.txt'))
                img_path_ori = file.readlines()
                file.close()
                if os.path.exists(os.path.join(root, 'class_mapping_model_sv.npy')):
                    print('Loading class mapping file...')
                    class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_sv.npy'),
                                                                allow_pickle=True).flatten()[0]
                    class_mapping_make = np.load(os.path.join(root, 'class_mapping_make_sv.npy'),
                                                                allow_pickle=True).flatten()[0]
                    save_class_mapping = False
                else:
                    print('Class mapping file not found...')
                    class_mapping_model = dict()
                    class_mapping_make = dict()
                    save_class_mapping = True
                    
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, 'image', i.strip()))
                    model = i.split('/')[0]
                    make = annotation[int(model) - 1][0].item()
                    label_model_tmp = class_mapping_model.get(model)
                    label_make_tmp = class_mapping_make.get(make)
                    if label_model_tmp is None:
                        class_mapping_model[model] = len(class_mapping_model) + 1
                        label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                    if label_make_tmp is None:
                        class_mapping_make[make] = len(class_mapping_make) + 1
                        label_make_tmp = class_mapping_make.get(make)
                    label_make.append(label_make_tmp)

                if save_class_mapping:
                    print('Saving class mapping file...')
                    np.save(os.path.join(root, 'class_mapping_model_sv.npy'), class_mapping_model)
                    np.save(os.path.join(root, 'class_mapping_make_sv.npy'), class_mapping_make)
            
            elif not is_train:
                file = open(os.path.join(root, 'test_surveillance.txt'))
                img_path_ori = file.readlines()
                file.close()
                assert os.path.exists(os.path.join(root, 'class_mapping_model_sv.npy')), 'Class mapping (model) file not found...'
                assert os.path.exists(os.path.join(root, 'class_mapping_make_sv.npy')), 'Class mapping (make) file not found...'
                class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_sv.npy'),
                                                            allow_pickle=True).flatten()[0]
                class_mapping_make = np.load(os.path.join(root, 'class_mapping_make_sv.npy'),
                                                            allow_pickle=True).flatten()[0]
                
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, 'image', i.strip()))
                    model = i.split('/')[0]
                    make = annotation[int(model) - 1][0].item()
                    label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                    label_make_tmp = class_mapping_make.get(make)
                    label_make.append(label_make_tmp)      
        
        elif root.find('stanford-car') != -1:
            cars_meta = scipy.io.loadmat(os.path.join(root, 'devkit', 'cars_meta.mat'))['class_names'][0]
            with open(os.path.join(root, 'devkit', 'vmmr_to_vtr.json'), 'r') as file:
                vmmr_to_vtr = json.load(file)["vmmr2vtr"]
            if "combine_vtr_label" in kwargs:
                if kwargs["combine_vtr_label"]:
                    vtr_annot_simple = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 3, 8: 5, 9: 3}
            if is_train:
                with open(os.path.join(root, 'car_poor_images_train.txt'), 'r') as f:
                    poor_images = f.readlines()
                poor_images = [i.strip() for i in poor_images]
                img_path_ori = scipy.io.loadmat(os.path.join(root, 'devkit', 'cars_train_annos.mat'))['annotations'][0]
                viewpoint_annot = np.load(os.path.join(root, 'label_viewpoint_train.npy'), allow_pickle=True).flatten()[0]
                if os.path.exists(os.path.join(root, 'class_mapping_model_cars.npy')):
                    print('Loading class mapping file...')
                    class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_cars.npy'),
                                                                allow_pickle=True).flatten()[0]
                    class_mapping_make = np.load(os.path.join(root, 'class_mapping_make_cars.npy'),
                                                                allow_pickle=True).flatten()[0]
                    save_class_mapping = False
                else:
                    print('Class mapping file not found...')
                    class_mapping_model = dict()
                    class_mapping_make = dict()
                    save_class_mapping = True
                    
                for i in tqdm(img_path_ori):
                    
                    if kwargs.get("remove_poor_images") and i[-1].item() in poor_images:
                        # label_viewpoint_tmp = 0
                        continue

                    img_path.append(os.path.join(root, 'cars_train', i[-1].item()))
                    model = i[-2].item()
                    make = cars_meta[model - 1].item().split()[0]
                    label_model_tmp = class_mapping_model.get(model)
                    label_make_tmp = class_mapping_make.get(make)
                    if label_model_tmp is None:
                        class_mapping_model[model] = model
                        label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                    if label_make_tmp is None:
                        class_mapping_make[make] = len(class_mapping_make) + 1
                        label_make_tmp = class_mapping_make.get(make)
                    label_make.append(label_make_tmp)
                    
                    label_type_tmp = vmmr_to_vtr[str(label_model_tmp)]
                    if "combine_vtr_label" in kwargs:
                        if kwargs["combine_vtr_label"]:
                            label_type_tmp = vtr_annot_simple[label_type_tmp]
                    label_type.append(label_type_tmp)

                    if use_bbox:
                        xmin, ymin, xmax, ymax = i[0].item(), i[2].item(), i[1].item(), i[3].item()
                        bbox.append((xmin, xmax, ymin, ymax))

                    if use_viewpoint:
                        label_viewpoint_tmp = viewpoint_annot[i[-1].item()]
                        label_viewpoint.append(label_viewpoint_tmp)

                if save_class_mapping:
                    print('Saving class mapping file...')
                    np.save(os.path.join(root, 'class_mapping_model_cars.npy'), class_mapping_model)
                    np.save(os.path.join(root, 'class_mapping_make_cars.npy'), class_mapping_make)

            elif not is_train:
                img_path_ori = scipy.io.loadmat(os.path.join(root, 'devkit', 'cars_test_annos_withlabels.mat'))['annotations'][0]
                viewpoint_annot = np.load(os.path.join(root, 'label_viewpoint_train.npy'), allow_pickle=True).flatten()[0]
                assert os.path.exists(os.path.join(root, 'class_mapping_model_cars.npy')), 'Class mapping (model) file not found...'
                assert os.path.exists(os.path.join(root, 'class_mapping_make_cars.npy')), 'Class mapping (make) file not found...'
                class_mapping_model = np.load(os.path.join(root, 'class_mapping_model_cars.npy'),
                                                            allow_pickle=True).flatten()[0]
                class_mapping_make = np.load(os.path.join(root, 'class_mapping_make_cars.npy'),
                                                            allow_pickle=True).flatten()[0]
                
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, 'cars_test', i[-1].item()))
                    model = i[-2].item()
                    make = cars_meta[model - 1].item().split()[0]
                    label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                    label_make_tmp = class_mapping_make.get(make)
                    label_make.append(label_make_tmp)
                    
                    label_type_tmp = vmmr_to_vtr[str(label_model_tmp)]
                    if "combine_vtr_label" in kwargs:
                        if kwargs["combine_vtr_label"]:
                            label_type_tmp = vtr_annot_simple[label_type_tmp]
                    label_type.append(label_type_tmp)

                    if use_bbox:
                        xmin, ymin, xmax, ymax = i[0].item(), i[2].item(), i[1].item(), i[3].item()
                        bbox.append((xmin, xmax, ymin, ymax))
                    
                    if use_viewpoint:
                        label_viewpoint_tmp = viewpoint_annot[i[-1].item()]
                        label_viewpoint.append(label_viewpoint_tmp)

        elif root.find('BIT') !=-1:
            if use_bbox or use_model or use_make or use_viewpoint:
                raise Exception('No such annotation')
            assert kwargs['exp'] >=0 and kwargs['exp'] <=3
            if is_train:
                file = open(os.path.join(root, f"train_exp{kwargs['exp']}.txt"), 'r')
                img_path_ori = file.readlines()
                file.close()
                
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, i.strip()))
                    label_type_tmp = int(i.split('/')[0])
                    label_type.append(label_type_tmp)
            
            elif not is_train:
                file = open(os.path.join(root, f"test_exp{kwargs['exp']}.txt"), 'r')
                img_path_ori = file.readlines()
                file.close()
                
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, i.strip()))
                    label_type_tmp = int(i.split('/')[0])
                    label_type.append(label_type_tmp)

        elif root.find("MIO-TCD") != -1:
            if use_bbox or use_model or use_make or use_viewpoint:
                raise Exception('No such annotation')
            if is_train:
                train_dataset = datasets.ImageFolder(root=os.path.join(root, "train"))
                for samples in train_dataset.samples:
                    img_path.append(samples[0])
                    label_type.append(samples[1])
            elif not is_train:
                val_dataset = datasets.ImageFolder(root=os.path.join(root, "test"))
                for samples in train_dataset.samples:
                    img_path.append(samples[0])
                    label_type.append(samples[1])

        elif root.find("Car-FG3K") != -1:
            if use_bbox or use_type or use_viewpoint:
                raise Exception('No such annotation')
            if is_train:
                with open(os.path.join(root, "191train.txt"), "r") as file:
                    img_path_ori = file.readlines()
                if os.path.exists(os.path.join(root, 'class_mapping_model.npy')):
                    print('Loading class mapping file...')
                    class_mapping_model = np.load(os.path.join(root, 'class_mapping_model.npy'),
                                                  allow_pickle=True).flatten()[0]
                    class_mapping_make = np.load(os.path.join(root, 'class_mapping_make.npy'),
                                                  allow_pickle=True).flatten()[0]
                    save_class_mapping = False
                else:
                    print('Class mapping file not found...')
                    class_mapping_model = dict()
                    class_mapping_make = dict()
                    save_class_mapping = True
                
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, i.split('\t')[0]))
                    model = '/'.join(i.split('/')[:2])
                    make = i.split('/')[0]
                    label_model_tmp = class_mapping_model.get(model)
                    label_make_tmp = class_mapping_make.get(make)
                    if label_model_tmp is None:
                        class_mapping_model[model] = len(class_mapping_model) + 1
                        label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                    if label_make_tmp is None:
                        class_mapping_make[make] = len(class_mapping_make) + 1
                        label_make_tmp = class_mapping_make.get(make)
                    label_make.append(label_make_tmp)
                        
                if save_class_mapping:
                    print('Saving class mapping file...')
                    np.save(os.path.join(root, 'class_mapping_model.npy'), class_mapping_model)
                    np.save(os.path.join(root, 'class_mapping_make.npy'), class_mapping_make)
            
            elif not is_train:
                with open(os.path.join(root, "191test.txt"), "r") as file:
                    img_path_ori = file.readlines()
                assert os.path.exists(os.path.join(root, 'class_mapping_model.npy')), 'Class mapping (model) file not found...'
                assert os.path.exists(os.path.join(root, 'class_mapping_make.npy')), 'Class mapping (make) file not found...'
                class_mapping_model = np.load(os.path.join(root, 'class_mapping_model.npy'),
                                                            allow_pickle=True).flatten()[0]
                class_mapping_make = np.load(os.path.join(root, 'class_mapping_make.npy'),
                                                            allow_pickle=True).flatten()[0]
                
                for i in tqdm(img_path_ori):
                    img_path.append(os.path.join(root, i.split('\t')[0]))
                    model = '/'.join(i.split('/')[:2])
                    make = i.split('/')[0]
                    label_model_tmp = class_mapping_model.get(model)
                    label_model.append(label_model_tmp)
                    label_make_tmp = class_mapping_make.get(make)
                    label_make.append(label_make_tmp)
        
        elif root.find("mohsin_vmmr") != -1:
            if use_bbox or use_make or use_type or use_viewpoint:
                raise Exception("No such annotation")
            
            if is_train:
                if os.path.exists(os.path.join(root, 'class_mapping_model.npy')):
                    print('Loading class mapping file...')
                    class_mapping_model = np.load(os.path.join(root, 'class_mapping_model.npy'),
                                                  allow_pickle=True).flatten()[0]
                    save_class_mapping = False
                else:
                    print('Class mapping file not found...')
                    class_mapping_model = dict()
                    save_class_mapping = True

                for root_tmp, dirs, files in tqdm(os.walk(root)):
                    if len(dirs) == 0 and 'train' in root_tmp:
                        # Get model
                        model = os.path.split(root_tmp)[-1]
                        label_model_tmp = class_mapping_model.get(model)
                        if label_model_tmp is None:
                            class_mapping_model[model] = len(class_mapping_model) + 1
                            label_model_tmp = class_mapping_model.get(model)

                        for file in files:
                            img_path.append(os.path.join(root_tmp, file))
                            label_model.append(label_model_tmp)

                if save_class_mapping:
                    print('Saving class mapping file...')
                    np.save(os.path.join(root, 'class_mapping_model.npy'), class_mapping_model)

            elif not is_train:
                assert os.path.exists(os.path.join(root, 'class_mapping_model.npy')), 'Class mapping (model) file not found...'
                class_mapping_model = np.load(os.path.join(root, 'class_mapping_model.npy'),
                                                            allow_pickle=True).flatten()[0]
                for root_tmp, dirs, files in tqdm(os.walk(root)):
                    if len(dirs) == 0 and 'test' in root_tmp:
                        # Get model
                        model = os.path.split(root_tmp)[-1]
                        label_model_tmp = class_mapping_model.get(model)

                        for file in files:
                            img_path.append(os.path.join(root_tmp, file))
                            label_model.append(label_model_tmp)

        elif root.find("brcars") != -1:
            if use_bbox or use_type or use_viewpoint:
                raise Exception("No such annotation")
            
            class_mapping_model = pd.read_csv(os.path.join(root, f"brcars{kwargs['num_classes']}.csv"))
            
            # Get only external view, remove cockpit view
            class_mapping_model = class_mapping_model.loc[class_mapping_model['persp'] == 'e', ]

            if is_train:
                class_mapping_model = class_mapping_model.loc[class_mapping_model['split'] == 0,]
                for i, row in tqdm(class_mapping_model.iterrows()):
                    img_path.append(os.path.join(root, row['uri']))
                    label_model.append(row['model_id'])
                    label_make.append(row['make_id'])

            elif not is_train:
                class_mapping_model = class_mapping_model.loc[class_mapping_model['split'] == 1,]

                for i, row in tqdm(class_mapping_model.iterrows()):
                    img_path.append(os.path.join(root, row['uri']))
                    label_model.append(row['model_id'])
                    label_make.append(row['make_id'])

        self.img_path = np.array(img_path)
        self.bbox = np.array(bbox)
        self.label_model = np.array(label_model) - 1
        self.label_make = np.array(label_make) - 1
        self.label_type = np.array(label_type) - 1
        self.label_viewpoint = np.array(label_viewpoint) - 1

        if 'use_model_label' in kwargs:
            assert isinstance(kwargs['use_model_label'], list)
            idx = [np.where(self.label_model == i)[0] for i in kwargs['use_model_label']]
            idx = np.hstack(idx)
            self.img_path = self.img_path[idx]
            if len(self.bbox) > 0: self.bbox = self.bbox[idx]
            self.label_model = self.label_model[idx]
            remap_label_dict = {label: no for no, label in enumerate(kwargs['use_model_label'])}
            self.label_model = np.array([remap_label_dict[i] for i in self.label_model])
            
            self.label_make = self.label_make[idx]
            self.label_type = self.label_type[idx]
            if len(self.label_viewpoint) > 0: self.label_viewpoint = self.label_viewpoint[idx]

        if "use_n_img_per_cls" in kwargs:
            assert(isinstance(kwargs["use_n_img_per_cls"], int))
            idx_all = []
            for cls_ in np.unique(self.label_type):
                idx = np.where(self.label_type == cls_)[0]
                np.random.shuffle(idx)
                idx = idx[:kwargs["use_n_img_per_cls"]]
                idx_all.append(idx)
            idx_all = np.concatenate(idx_all, 0)
            self.img_path = self.img_path[idx_all]
            self.label_type = self.label_type[idx_all]
            dt_now = datetime.now().strftime("%Y_%m_%d_%H_%M")
            with open(f"idx_{is_train}_{dt_now}.txt", "w") as f:
                f.writelines("\n".join(idx_all.astype(str)))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image = Image.open(self.img_path[idx]).convert(mode='RGB')
        if self.use_bbox and len(self.bbox) > 0:
            image = image.crop((self.bbox[idx][0], self.bbox[idx][1], 
                                self.bbox[idx][2], self.bbox[idx][3]))
        try:
            image = self.transform(image)
        except Exception as e:
            print('Image Transform Error: ', self.img_path[idx])
        label = ()
        if self.use_model:
            label_model = self.label_model[idx]
            label += (label_model,)
        if self.use_make:
            label_make = self.label_make[idx]
            label += (label_make,)
        if self.use_type:
            label_type = self.label_type[idx]
            label += (label_type,)
        if self.use_viewpoint:
            label_viewpoint = self.label_viewpoint[idx]
            label += (label_viewpoint,)
        return self.img_path[idx], image, label
        

class BalancedBatchSampler(BatchSampler):
    """
    https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/4
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, label, batch_size, n_samples_per_cls):
        self.labels_list = label
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        assert batch_size >= n_samples_per_cls
        self.batch_size = batch_size
        self.n_samples_per_cls = n_samples_per_cls
        self.n_classes_to_sample = batch_size // n_samples_per_cls
        self.dataset = dataset

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes_to_sample, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples_per_cls])
                self.used_label_indices_count[class_] += self.n_samples_per_cls
                if self.used_label_indices_count[class_] + self.n_samples_per_cls > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes_to_sample * self.n_samples_per_cls

    def __len__(self):
        return len(self.dataset) // self.batch_size


if __name__=='__main__':
    # root = 'data/car'
    root = '/data/CompCars/web_nature'
    # root = 'data/CompCars/sv_data'
    # root = 'data/BITVehicle_Dataset_mine'
    # root = "data/MIO-TCD"
    # root = "data/Car-FG3K"
    # root = "data/brcars"
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    img_dataset = MyDataset(root=root, is_train=False, use_bbox=True, 
                            use_model=True, use_make=False, use_type=False,
                            use_viewpoint=False, transform=data_transform, 
                            # use_topn_model=[2, 8, 10], 
                            # exp=0
                            # combine_vtr_label=True
                            # use_n_img_per_cls=250
                            num_classes=431
                            )
    _, count = np.unique(img_dataset.label_model, return_counts=True)
    print(count.min())
    print(count.mean())
    print(np.percentile(count, 50))
    print(count.max())
    print(np.std(count))
    
    # sampler = BalancedBatchSampler(img_dataset, 64, 4)
    mydataloader = torch.utils.data.DataLoader(img_dataset, batch_size=64,
                                                shuffle=True, num_workers=8, 
                                                drop_last=False)
    # mydataloader = torch.utils.data.DataLoader(
    #     img_dataset, batch_sampler=sampler,num_workers=4
    #     )
    for _, inputs, labels in mydataloader:
        print(inputs.size(), labels[0].size())
