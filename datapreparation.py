from torch.utils.data import DataLoader, SequentialSampler, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from albumentations import Cutout
import imageio
import os
import torch
import random
from imgaug import augmenters as iaa
import imgaug as ia
import shutil
from skimage import exposure

def seed_torch(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def dataprepare():
    from torch.utils.data import DataLoader, SequentialSampler, random_split
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import numpy as np
    from albumentations import Cutout
    import imageio
    import os
    import torch
    import random
    from imgaug import augmenters as iaa
    import imgaug as ia
    import shutil

    seed_torch(200)

    train_dataset_path = os.path.join(os.getcwd(), 'kaggle_dataset', 'Covid19-dataset', 'train')#r'G:\softwarelife\LJMU\covid19\Covid19-dataset\train'
    test_dataset_path = os.path.join(os.getcwd(), 'kaggle_dataset', 'Covid19-dataset', 'test')#r'G:\softwarelife\LJMU\covid19\Covid19-dataset\test'
    k_data_path = os.path.join(os.getcwd(), 'dataset')
    delete_paths = [k_data_path, os.path.join(os.getcwd(), 'kaggle_dataset', 'Covid19-dataset', 'data')]
    for delete_path in delete_paths:
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)
            print(delete_path, 'removed')
        else:
            print(delete_path, 'not exist to delete')

    #combined data
    data_path = os.path.join(os.getcwd(), 'kaggle_dataset', 'Covid19-dataset')
    for root, dirs, files in os.walk(data_path, topdown=False):
        for filename in files:
            file_path=os.path.join(root, filename)
            req_dir = root.replace('train','data').replace('test', 'data')
            os.makedirs(req_dir, exist_ok=True)
            req_path = os.path.join(req_dir, filename)
            try:
                shutil.copy2(file_path, req_path)
            except Exception as sf:
                print('error is' , sf)
                print(file_path, 'file_path')

    dataset_path = os.path.join(os.getcwd(), 'kaggle_dataset', 'Covid19-dataset', 'data')

    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        """

        # override the __getitem__ method. this is the method that dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path

    def pre_process(dataset_path):
        transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                       ])
        return ImageFolderWithPaths(dataset_path, transform=transform)

    class ImgAugTransform:#cutout augmentation #

      def __init__(self):
        self.aug_s = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0.5, 3.0)),
        ])
        self.aug_f = iaa.Sequential([
            iaa.Fliplr(1),
        ])
        self.aug_a = iaa.Sequential([
            iaa.Affine(rotate=(-10, 10), mode='symmetric'),
        ])
        self.aug_h = iaa.Sequential([
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

        self.aug_h2 = iaa.Sequential([
            iaa.AddToHueAndSaturation(value=(-20, 20), per_channel=True)
        ])
    #     self.aug_c = Cutout()
        self.aug_ot = Cutout(num_holes=15, max_h_size=4, max_w_size=4, always_apply=True)

        self.aug_ot2 = Cutout(num_holes=10, max_h_size=6, max_w_size=6, always_apply=True)

        self.img_eq = exposure.equalize_hist

        self.img_eq_ada = exposure.equalize_adapthist

      def __call__(self, img):
        img = np.array(img)
        img = [(self.aug_a.augment_image(img), 'affine'), (self.aug_f.augment_image(img), 'fliplr'), (self.aug_s.augment_image(img), 'gaussian'), (self.aug_h.augment_image(img), 'hue_saturation'), (self.aug_h2.augment_image(img), 'hue_saturation2'), (self.aug_ot(image=img)['image'], 'cut_out'), (self.aug_ot2(image=img)['image'], 'cut_out2'), (self.img_eq(image=img), 'img_eq'), (self.img_eq_ada(image=img, clip_limit=0.03), 'img_eq_ada')]
        return img
#
# transforms = ImgAugTransform()
#     class ImgAugTransform:#cutout augmentation #

#       def __init__(self):
#         self.aug_s = iaa.Sequential([
#             iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
#         ])
#         self.aug_f = iaa.Sequential([
#             iaa.Fliplr(0.1),
#         ])
#         self.aug_a = iaa.Sequential([
#             iaa.Affine(rotate=(-10, 10), mode='symmetric'),
#         ])
#         self.aug_h = iaa.Sequential([
#             iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
#         ])
#     #     self.aug_c = Cutout()
#         self.aug_ot = Cutout(num_holes=10, max_h_size=6, max_w_size=6)
#       def __call__(self, img):
#         img = np.array(img)
#         img = [(self.aug_a.augment_image(img), 'affine'), (self.aug_f.augment_image(img), 'fliplr'), (self.aug_s.augment_image(img), 'gaussian'), (self.aug_h.augment_image(img), 'hue_saturation'), (self.aug_ot(image=img)['image'], 'cut_out')]
#         return img
    #
    # transforms = ImgAugTransform()

    def preprocess_augment_save_data(train_dataloader, id_to_class, folder='dataset', subfolder='train'):
        counter = 0
        for image, class_label, image_path in train_dataloader:
            new_path = os.path.join(os.getcwd(), folder , subfolder, id_to_class[class_label])
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            counter += 1
            imageio.imwrite(os.path.join(new_path, os.path.basename(image_path).split('.')[0]+'.jpeg'), np.array(image))
            image_aug_transform = ImgAugTransform()
            aug_images = image_aug_transform.__call__(image)
            for img_arr, subname in aug_images:
                imageio.imwrite(os.path.join(new_path, os.path.basename(image_path).split('.')[0]+f'_{subname}.jpeg'), img_arr)
                counter += 1

    def save_preprocess_data(val_dataloader, id_to_class, folder='dataset', subfolder='train'):
        counter = 0
        for image, class_label, image_path in val_dataloader:
            new_path = os.path.join(os.getcwd(), folder, subfolder, id_to_class[class_label])
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            counter += 1
            image = np.array(image)
            imageio.imwrite(os.path.join(new_path, os.path.basename(image_path).split('.')[0]+'.jpeg'), image)

    # Train
    dataset = pre_process(dataset_path)
    n = len(dataset)  
    n_test = int(0.3 * n)
    train_dataset, test_dataset  = random_split(dataset, [n-n_test, n_test])
    n = len(train_dataset)  
    n_test = int(0.3 * n)
    train_dataloader, val_dataloader  = random_split(train_dataset, [n-n_test, n_test])
    class_to_id = dataset.class_to_idx
    id_to_class = dict(zip(class_to_id.values(), class_to_id.keys()))
    preprocess_augment_save_data(train_dataloader, id_to_class, folder='dataset', subfolder='train')
    save_preprocess_data(val_dataloader, id_to_class, folder='dataset', subfolder='val')
    #test
    class_to_id = dataset.class_to_idx
    id_to_class = dict(zip(class_to_id.values(), class_to_id.keys()))
    test_dataset_loader = torch.utils.data.SubsetRandomSampler(test_dataset)
    save_preprocess_data(test_dataset_loader, id_to_class, folder='dataset', subfolder='test')

    #split data for binary and multi classification
    import shutil
    import os
    for new_folder in ['binary', 'multi']:
        classes = ['Covid', 'Normal', 'Viral Pneumonia']
    #     if new_folder == 'binary':
    #         classes.remove('Viral Pneumonia')
        for req_folder in classes:
            clas = req_folder
            for type_ in ['train', 'test', 'val']:
                src = os.path.join(os.getcwd(), 'dataset', type_ ,req_folder)
                if new_folder == 'binary':
                    if req_folder == 'Viral Pneumonia':
                        req_folder = 'Normal'
                dst = os.path.join(os.getcwd(), 'dataset', new_folder, type_ ,req_folder)
    #             shutil.copytree(src, dst)

                for root, dirs, files in os.walk(src, topdown=False):
                    for filename in files:
                        file_path=os.path.join(root, filename)
    #                     req_dir = root.replace('train','data').replace('test', 'data')
                        os.makedirs(dst, exist_ok=True)
                        req_path = os.path.join(dst, clas+'_'+filename)
                        try:
                            shutil.copy2(file_path, req_path)
                        except Exception as sf:
                            print('error is' , sf)
                            print(file_path, 'file_path')