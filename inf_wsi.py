import os
import numpy as np
import cv2
import pyvips
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn

from math import log2
from utils.model_ch import model_load


import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

import time

def norm_scale(x, **kwargs):
    return x.to(torch.float32) / 255.0

def get_preprocess():
    _transform = [
        ToTensorV2(transpose_mask=True),
        albu.Lambda(image=norm_scale),
    ]
    return albu.Compose(
        _transform,
        additional_targets={'lrimage': 'image', 'lrmask': 'mask'}
        )

class WSIPatchDataset(Dataset):
    def __init__(self, wsi_path, patch_size, stride, tifpage, lrratio=None):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.tifpage = tifpage
        self.without_white = True #setting about skipping space place

        self.lrratio = lrratio
        self.ckratio = 7
        self.lrpage = self.tifpage+int(log2(self.lrratio))
        self.lrshift = self.patch_size // 2 -self.patch_size // self.lrratio // 2
        
        if self.wsi_path[-3:]=='tif':
            self.wsi_image  = pyvips.Image.tiffload(self.wsi_path, page=self.tifpage)
            self.wsi_lrimage = pyvips.Image.tiffload(self.wsi_path, page=self.lrpage)
            self.wsi_image_checkRegion = pyvips.Image.tiffload(self.wsi_path, page=self.ckratio)
        
        else:
            self.wsi_image  = pyvips.Image.new_from_file(self.wsi_path, level=self.tifpage)
            self.wsi_lrimage = pyvips.Image.new_from_file(self.wsi_path, level=self.lrpage)
            self.wsi_image_checkRegion = pyvips.Image.new_from_file(self.wsi_path, level=self.ckratio)
        self.chmask = None
        
        # if not lrratio is None:
            # self.wsi_lrimage = pyvips.Image.new_from_file(self.wsi_path, path=tifpage)
            # self.img_lrregion = pyvips.Region.new(wsi_lrimage)

        self.patch_locs = self._generate_patch_loc()

        self.preprocess = get_preprocess()
        # self.patch_list, self.location_list = self._generate_patch_list()

    def __len__(self):
        return len(self.patch_locs)

    def __getitem__(self, idx):
        location = self.patch_locs[idx]
        x, y = location[0], location[1]

        image = self._tiffcrop(self.wsi_image, x, y, self.patch_size, self.patch_size)

        # multiscale
        if not self.lrratio is None:
            lrimage = self._tiffcheckcrop(  
                self.wsi_lrimage, x//self.lrratio-self.lrshift, y//self.lrratio-self.lrshift, self.patch_size, self.patch_size)
            sample = self.preprocess(image=image, lrimage=lrimage)
            image, lrimage = sample['image'], sample['lrimage']
            return x, y, image, lrimage

        sample = self.preprocess(image=image)
        image, lrimage = sample['image']

        return x, y, image

    def _generate_patch_loc(self, bg_limit_upper=240, bg_limit_lower=10):
        #without skip background
        # width = self.wsi_image.width
        # height = self.wsi_image.height
        # datas = []
        # for sy in range(0, height-self.patch_size, self.stride):
        #     for sx in range(0, width-self.patch_size, self.stride):
        #         datas.append([sx, sy])
        

        #原始影像大小
        width = self.wsi_image.width
        height = self.wsi_image.height
        datas = []

        #檢測倍率之影像讀取(pyvips.image -> memory)
        image = self.wsi_image_checkRegion.numpy()
        # image = np.ndarray(buffer=pyvips.Image.write_to_memory(self.wsi_image_checkRegion),
        #                     dtype=np.uint8,
        #                     shape=[height, width, self.wsi_image_checkRegion.bands])

        #調整大小 -> 原始影像大小/patch size 每一格表示1個patch的代表值
        #注意:當細胞太小可能會被忽視
        image_resize = cv2.resize(image, [int(width/self.patch_size), int(height/self.patch_size)])
        # print(int(width_orig/self.patch_size), int(height_orig/self.patch_size))

        #根據patch pixel ARGB平均決定是否保留 (生成mask)
        avg = np.average(image_resize, axis=2)
        # print(avg.shape)
        mask = np.ones(avg.shape)
        mask[avg < bg_limit_lower] = 0
        mask[avg > bg_limit_upper] = 0

        #使用膨脹增加邊緣與中間的彌補
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)

        #根據mask 選取標註位置 (依照patch size調回原始倍率之對應位置)
        for sy in range(mask.shape[0]):
            for sx in range(mask.shape[1]):
                if mask[sy, sx]==0:
                    continue
                datas.append([sx*self.patch_size, sy*self.patch_size])

        self.chmask = mask
        return datas

    def _tiffcheckcrop(self, slide, x, y, width, height):
        maxwidth, maxheight = slide.width, slide.height
        maxwidth -= x
        maxheight -= y
        # print(f'>>{slide.width},{slide.height}, {x}, {y}, {width}, {height}')
        if x>=0 and y>=0 and (width)<=maxwidth and (height)<=maxheight:
            image = self._tiffcrop(slide, x, y, width, height)
        else:
            # lefttop smaller than 0 crop location start from 0,0 but not over largest size
            w_comp = -x if x < 0 else 0
            h_comp = -y if y < 0 else 0
            w_crop = np.clip(width-w_comp, None, maxwidth)  
            h_crop = np.clip(height-h_comp, None, maxheight)
            crop = self._tiffcrop(slide, np.clip(x, 0, None), np.clip(y, 0, None), w_crop, h_crop)

            image = np.full((height, width, 3), 255, dtype=np.uint8) if crop.shape[2] == 3 \
                else np.zeros((height, width, 1))
            image[h_comp:h_comp+h_crop, w_comp:w_comp+w_crop] = crop

        return image

    def _tiffcrop(self, slide, x, y, width, height):
        #crop
        # vi = slide.crop(x, y, width, height)    #lefttop
        # image = np.ndarray(buffer=vi.write_to_memory(),
        #                  dtype=np.uint8,
        #                  shape=[vi.height, vi.width, vi.bands])

        #fetch (more fast)
        region = pyvips.Region.new(slide)
        vi = region.fetch(x, y, width, height)
        image = np.ndarray(buffer=vi,
                            dtype=np.uint8,
                            shape=[height, width, slide.bands])
        
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
        return image

    
# def rgba2rgb(rgba, background=(255, 255, 255)):
#     row, col, ch = rgba.shape
#     if ch == 3:
#         return rgba
#     assert ch == 4, 'RGBA image has 4 channels.'
#     # rgb = np.zeros((row, col, 3), dtype='float32')
#     # r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
#     # a = np.asarray(a, dtype='float32') / 255.0
#     # R, G, B = background
#     # rgb[:, :, 0] = r * a + (1.0 - a) * R
#     # rgb[:, :, 1] = g * a + (1.0 - a) * G
#     # rgb[:, :, 2] = b * a + (1.0 - a) * B
#     # return np.asarray(rgb, dtype='uint8')
#     return rgba[:,:,0:3]

# --other tool--
def draw_mask(result, mask, x, y, lowerbound, upperbound):
    _, h, w = result[:, y+lowerbound:y+upperbound, x+lowerbound:x+upperbound].shape
    result[:, y+lowerbound:y+upperbound, x+lowerbound:x+upperbound] = \
        torch.logical_or(result[:, y+lowerbound:y+upperbound, x+lowerbound:x+upperbound], mask[lowerbound:lowerbound+h, lowerbound:lowerbound+w])

def numpy2vips(a):
    dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
    }

    height, width = a.shape
    linear = a.reshape(width * height * 1)*255
    vi = pyvips.Image.new_from_array(a*255)
    # vi = pyvips.Image.new_from_memory(linear.data, width, height, 1,
    #                                   dtype_to_format[str(a.dtype)])
    return vi

def img_tensor2pillow(img_tensor):
    img_tensor = img_tensor.permute(1, 2, 0)
    img_np = img_tensor.numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    return img_pil

def img_tensor2pillow_mask(img_tensor):
    img_np = img_tensor.numpy()
    print(img_np.max())
    img_np = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np, mode='L')
    return img_pil


import argparse
import torch.backends.cudnn as cudnn
import torch
import yaml
from torch.utils.data import DataLoader

def main():


    MODEL_NAME = "MRCPSMixModel"                  # choosed model (need assign in utils/model_ch.py)

    #model config
    MODEL_CONF_PATH = './used_models/MRCPS/cfg/traincfg_MRCPS.yaml'      # model config path

    DATA_PATH  = '/work/u2676425/dataset/tiger_dataset/test/input'
    WEIGHT_PATH  = './model_weights_record/fl_switch_tiger_doubleCH2_3epoch_1131111/1.pt'
    RESULT_PATH     = '/work/u2676425/dataset/tiger_dataset/test/LGCFL_inf_tiger_ch2re_sw'
    MASK_PATH     = '/work/u2676425/dataset/tiger_dataset/test/LGCFL_inf_tiger_ch2re_sw'
    
    print(WEIGHT_PATH)
    print(RESULT_PATH)

    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH, exist_ok=True)
    if not os.path.isdir(MASK_PATH):
        os.makedirs(MASK_PATH, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # --model loading--
    model = model_load("MRCPSMixModel")
    model.load_state_dict(torch.load(WEIGHT_PATH))
    model= nn.DataParallel(model)
    model.cuda()

    #--patch setting--
    stride=512
    patch_size=512
    lowerbound = (patch_size - stride) // 2
    upperbound = stride + (patch_size - stride) // 2
    tifpage = 0
    #multiscale
    lrdiff = 3
    lrratio = 8
    
    #sliding to get patch location


    for tif_image in os.listdir(DATA_PATH):
        print(tif_image)
        # if tif_image[:4]=='TCGA':
        #     continue

        if tif_image[-3:]!='tif' and tif_image[-4:]!='ndpi' and tif_image[-4:]!='mrxs':
            continue
        #load tiff and prepare dataloader
        tif_path = os.path.join(DATA_PATH, tif_image)
        save_path = os.path.join(RESULT_PATH, tif_image)
        if os.path.isfile(save_path):
            continue

        wsi_dataset = WSIPatchDataset(tif_path, patch_size, stride, tifpage, lrratio)
        #save checkmask
        if not (MASK_PATH is None):
            chmask_path = os.path.join(MASK_PATH, tif_image[:-len(tif_image.split('.')[-1])]+'.png')
            cv2.imwrite(chmask_path, wsi_dataset.chmask*255) 

        # gen_cells_patches(tif_path, save_dir=None, save=True, stride=stride, patch_size=patch_size)
        # infds = InferDataset(inf_dict, tif_path, preprocess)
        dataloader = DataLoader(
                        dataset=wsi_dataset,
                        batch_size=16,
                        pin_memory=True,
                        num_workers=0
                        )


            #result tif size
        print(wsi_dataset.wsi_image.height, wsi_dataset.wsi_image.width)
        result = torch.zeros(
            (1, wsi_dataset.wsi_image.height, wsi_dataset.wsi_image.width), dtype=torch.uint8).cuda()

        model.eval()
        with torch.no_grad():
            processCount=0
            # time_record_preprocess = time.time()  #time check
            for batch in dataloader:
                # print('prepare data time:', time.time()-time_record_preprocess)   #time check
                processCount+=1
                print(f'{processCount}/{len(dataloader)}', end='\r')
                if not lrratio is None:
                    xs, ys, imgs, lrimgs = batch
                else:
                    xs, ys, imgs = batch
                #test for img input
                # count=0
                # for img in imgs:
                #     save_path = os.path.join(args.save_path, f'a{0}.png')
                #     img.save(save_path)

                imgs = imgs.cuda()
                lrimgs = lrimgs.cuda()
                # time_record_inf = time.time()   #time check
                masks = model(imgs,lrimgs)
                # print('inference time:', time.time()-time_record_inf)   #time check
                # masks_cofidence, masks_index = torch.max(masks, 1)
                # print(masks.shape)
                # masks_norm = torch.nn.functional.softmax(masks, 1)
                # masks_cofidence, masks_index = torch.max(masks_norm, 1)
                # print(torch.sum(masks_index))

                # masks = masks.argmax(dim=1)
                # print(masks.shape)

                # if masks.max()==0:
                #     continue

                #patch image save
                # for img, x, y in zip(imgs, xs, ys):
                #     img_tensor = img.cpu()
                #     img_pil = img_tensor2pillow(img_tensor)
                #     temp_ori_save_path = os.path.join(args.save_path, f'ori_{tif_image[:-4]}_{x}_{y}.png')
                #     print(temp_ori_save_path)
                #     img_pil.save(temp_ori_save_path)
            
                
                # time_record_draw = time.time()  #time check
                for mask, x, y in zip(masks, xs, ys):
                    #patch mask save
                    # mask_tensor = mask.cpu()
                    # mask_pil = img_tensor2pillow_mask(mask_tensor)
                    # temp_mask_save_path = os.path.join(RESULT_PATH, f'mask_{tif_image[:-4]}_{x}_{y}.png')
                    # print(temp_mask_save_path)
                    # mask_pil.save(temp_mask_save_path)

                    #whole slide image
                    draw_mask(result, mask, x, y, lowerbound, upperbound)

                # print('draw time:', time.time()-time_record_draw)   #time check
                # time_record_preprocess = time.time()    #time check


        #transfer to tif
        result = result.squeeze(dim=0).cpu().numpy()
        # print(result.shape)
        vips_img = numpy2vips(result)
        # print(vips_img.width, vips_img.height)
        vips_img.tiffsave(save_path, compression='deflate', tile=True, bigtiff=True, pyramid=True)
        # vips_img.write_to_file(save_path, tile=True, bigtiff=True, pyramid=True)
        print('finish')


if __name__ == '__main__':
    main()



                
   