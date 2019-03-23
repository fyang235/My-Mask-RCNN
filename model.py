from pycocotools.coco import COCO
import numpy as np
import os
import skimage
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import datetime
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.engine as KE
import tensorflow as tf

#from BN16 import BatchNormalizationF16
K.set_floatx('float32')
K.set_epsilon(1e-7)
############################################################
#  Dataset class
############################################################
class Dataset(object):
    def __init__(self, path, subset, config):
        self.path = path
        self.subset = subset
        self.config = config
        self.load_coco()
        
    def get_class_info(self, coco):
        '''
        get image info from raw data. 
        inputs: coco object
        returns: 
        '''
        # build a list of dict to store class info
        class_info = []
        
        # get source class ids
        source_class_ids = sorted(coco.getCatIds())
        
        # get class names
        class_names = []
        for sou_id in source_class_ids:
            class_names.append(coco.loadCats(sou_id)[0]['name'])
            
        # get new class ids. add one for background    
        class_ids = np.array(range(len(source_class_ids) + 1))
        
        # construct class info
        for id in class_ids:
            if id == 0:
                info = {'class_id': 0,
                        'source_class_ids': None,
                        'name': 'BG'}
            else:
                info = {'class_id': id,
                        'source_class_ids': source_class_ids[id - 1],
                        'name': class_names[id - 1]}
            class_info.append(info)
        
        #
        self.map_id_to_source_id = {}
        self.map_source_id_to_id = {}
        self.map_id_to_class_name = {}
        for dictionary in class_info:
            id = dictionary['class_id']
            sou_id = dictionary['source_class_ids']
            name = dictionary['name']
            
            self.map_id_to_source_id[id] = sou_id
            self.map_source_id_to_id[sou_id] = id
            self.map_id_to_class_name[id] = name
            
        self.class_ids = class_ids
        self.source_class_ids = source_class_ids    
        self.class_info = class_info
        self.num_of_classes = len(class_info)
        
        print('num_of_classes: {}'.format(self.num_of_classes))
        #print('source_class_ids: ',source_class_ids)
        #print('class_ids: ',class_ids)
        #print('class_info: ',class_info)
        
    def get_image_info(self, coco):
        '''
        get image info from raw data. 
        inputs: coco object
        returns: 
        '''
        # build a list of dict to store img info
        image_info = []
        
        # get source img ids
        source_image_ids = []
        for sou_id in self.source_class_ids:
            source_image_ids.extend(coco.getImgIds(catIds=[sou_id]))
            
        # new img id
        image_ids = np.array(range(len(source_image_ids)))

        # construct image info
        for id in image_ids:
            info = {}
            sou_id = source_image_ids[id]
            info['image_id'] = id
            info['source_image_id'] = sou_id
            info['path'] = os.path.join(self.path, self.subset, coco.imgs[sou_id]['file_name'])
            info['width']=coco.imgs[sou_id]["width"]
            info['height']=coco.imgs[sou_id]["height"]
            info['annotatioins'] = coco.loadAnns(coco.getAnnIds(imgIds=[sou_id], catIds=self.source_class_ids, iscrowd=None))
            image_info.append(info)
            
        self.image_info = image_info
        self.num_of_images = len(image_info)
        print('Number of images: {}'.format(self.num_of_images))
        #print('image_info[:2]: ',image_info[:2])

    def load_coco(self):
        '''
        get image info from raw data. 
        inputs: coco object
        returns: 
        '''
        coco = COCO("{}/annotations/instances_{}.json".format(self.path, self.subset))
        
        self.get_class_info(coco)
        self.get_image_info(coco)
        
        print('Number of classes: {}'.format(self.num_of_classes))
        
    def load_image(self, image_id):
        '''
        get image info from raw data. 
        inputs: coco object
        returns: 
        '''
        image_path = self.image_info[image_id]['path']
        image = skimage.io.imread(image_path)
        # same images are in 2D format
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate(3*[image], axis=-1)
            
        return image
    
    def load_mask(self, image_id):    
        '''
        get image info from raw data. 
        inputs: coco object
        returns: [height, width, instance count]
        '''
        # get info
        image_info = self.image_info[image_id]
        
        anns = image_info['annotatioins']
        H = image_info['height']
        W = image_info['width']

        # place holders
        masks = np.zeros([H, W, len(anns)], dtype=np.int32)
        mask_class = np.zeros(len(anns), dtype=np.int32)
        
        # fill them
        for i, ann in enumerate(anns):
            masks[:,:,i] = self.annToMask(ann, H, W)
            id = self.map_source_id_to_id[ann['category_id']]
            mask_class[i] = id  
        return masks.astype(np.bool), mask_class

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def load_bbox(self, image_id):
        '''
        get image info from raw data. 
        inputs: coco object
        returns: 
            bboxes: [instance count, (x1, y1, w, h)]
            bbox_class: [instance count,]
        '''
        # get info
        image_info = self.image_info[image_id]
        anns = image_info['annotatioins']

        # place holders
        bboxes = np.zeros([len(anns), 4], dtype=np.float32)
        bbox_class = np.zeros(len(anns), dtype=np.int32)
        
        # fill them
        for i, ann in enumerate(anns):
            bboxes[i] = np.array(ann['bbox'])
            id = self.map_source_id_to_id[ann['category_id']]
            bbox_class[i] = id  
        return bboxes, bbox_class       
    
    def visulize(self, image_id, showmask=True, showbbox=True, showclass=True, show_resized=True):
        image = self.load_image(image_id)
        # resize
        if show_resized:
            print(image.shape, type(image), np.max(image))
            image, window, ratio, padding = self.resize_image(image, [self.config.IMAGE_MAX_DIM]*2)
            print(image.shape, type(image), np.max(image))
            
        _, ax = plt.subplots(1, figsize=(10, 10))
        ax.axis('off')
        
        
        if showmask:
            masks, classes = self.load_mask(image_id)
            # resize
            if show_resized:
                print(masks.shape, type(masks), np.max(masks))
                masks = self.resize_mask(masks, window, padding)
                print(masks.shape, type(masks), np.max(masks))
            num_instance = len(classes)
            colors = self.random_colors(num_instance)
            
            for i in range(num_instance):
                image = self.apply_mask(image, masks[:,:,i], color=colors[i])
         
        if showbbox:
            bboxes, bbox_class = self.load_bbox(image_id)
                # resize
            if show_resized:
                bboxes = self.resize_bbox(bboxes, ratio, padding)
                    
            if not showmask:
                num_instance = len(bbox_class)
                colors = self.random_colors(num_instance)
            
            for i in range(num_instance):
                # Using bboxes generated from masks
                #y1, x1, y2, x2 = bboxes[i]
                #p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                      #linewidth=2,
                                      #alpha=0.7, 
                                      #linestyle="dashed",
                                      #edgecolor=colors[i], 
                                      #facecolor='none')
                # Using bboxes provided in coco
                x1, y1, w, h = bboxes[i]
                p = patches.Rectangle((x1, y1), w, h, 
                                      linewidth=2,
                                      alpha=0.7, 
                                      linestyle="dashed",
                                      edgecolor=colors[i], 
                                      facecolor='none')                
                ax.add_patch(p)
                
                ax.text(x1, y1+8, 
                        self.map_id_to_class_name[bbox_class[i]],
                        color='w', size=11, backgroundcolor="none")
            
        ax.imshow(image)
        plt.show()
        
    def random_colors(self, num=3):
        return np.random.random([num, 3])
    
    def apply_mask(self, image, mask, color, alpha=0.5):
        masked_image = np.zeros(image.shape, dtype=image.dtype)
        for i in range(image.shape[-1]):
            masked_image[:,:,i] = np.where(mask,
                                    alpha*color[i]*255 + (1-alpha)*image[:,:,i],
                                    image[:,:,i])
        return masked_image
        
    # TODO pass in resized image and boxes instead of image_id to speed up
    def generate_gt_for_rpn(self, anchors, bboxes):
        '''
        inputs:
            image_id: the image for which to generat gt rpn data
            anchors: [num_of_anchor, (y1, x1, y2, x2)]
        returns:
            anchors_match: [num_of_anchor, ]
            anchors_delta_bbox: [RPN_TRAIN_ANCHORS_PER_IMAGE, (ct_dy, ct_dx, logdh, logdw)], zeros padded
        '''
        
        # 
        ious = self.IoU_many_to_many(anchors, bboxes)
        
        # place holders
        anchors_match = np.zeros(len(anchors))
        # the number of training anchors is set to, say, 256 with positive ones in the front and padded with zeros to make sure same length for each batch.
        anchors_delta_bbox = np.zeros([self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4])
        
        # chooes negative anchors
        anchors_argmax_ids = np.argmax(ious, axis=1)
        anchors_max = ious[range(len(anchors)), anchors_argmax_ids]
        neg_ids = np.where(anchors_max < 0.3)[0]
        anchors_match[neg_ids] = -1
        
        # positive ones, make sure every gt box has at least one anchor, even when iou < 0.3
        # make the highest scored anchors posivite
        gt_argmax_ids = np.argmax(ious, axis=0)
        anchors_match[gt_argmax_ids] = 1
        # for > 0.7
        pos_ids = np.where(anchors_max > 0.7)[0]
        anchors_match[pos_ids] = 1
        
        # balance pos and neg, pos should not be more than half
        # update pos_ids
        pos_ids = np.where(anchors_match == 1)[0]
        
        extra = len(pos_ids) - self.config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2
        if extra > 0:
            ids = np.random.choice(pos_ids, extra, replace=False)
            anchors_match[ids] = 0
            # update pos_ids
            pos_ids = np.where(anchors_match == 1)[0]
            
        extra = len(neg_ids) - (self.config.RPN_TRAIN_ANCHORS_PER_IMAGE - len(pos_ids))
        if extra > 0:
            ids = np.random.choice(neg_ids, extra, replace=False)
            anchors_match[ids] = 0
        
        
        # fill anchors_delta_bbox
        pos_anchors = anchors[pos_ids]
        gt_bboxes = bboxes[anchors_argmax_ids[pos_ids]]
        
        # change format to (x1, y1, w, h)
        pos_anchors = self.convert_y1x1y2x2_to_x1y1wh(pos_anchors) 
        gt_bboxes = self.convert_y1x1y2x2_to_x1y1wh(gt_bboxes)
        
        an_x1, an_y1, an_w, an_h = np.split(pos_anchors, 4, axis=1)
        gt_x1, gt_y1, gt_w, gt_h = np.split(gt_bboxes, 4, axis=1)
        
        #ct_dx = (gt_x1 + gt_w/2)-(an_x1 + an_w/2)
        #ct_dy = (gt_y1 + gt_h/2)-(an_y1 + an_h/2)
        
        gt_xc = gt_x1 + gt_w/2
        gt_yc = gt_y1 + gt_h/2
        an_xc = an_x1 + an_w/2
        an_yc = an_y1 + an_h/2
        # use relative value
        ct_dx = (gt_xc - an_xc) / an_w
        ct_dy = (gt_yc - an_yc) / an_h
        logdw = np.log(gt_w / an_w)
        logdh = np.log(gt_h / an_h)
        
        #pos_anchors_delta_bbox = np.concatenate([ct_dx, ct_dy, logdw, logdh], axis=1)
        pos_anchors_delta_bbox = np.concatenate([ct_dy, ct_dx, logdh, logdw], axis=1)
        
        pos_anchors_delta_bbox /= self.config.RPN_BBOX_STD_DEV
        
        anchors_delta_bbox[:len(pos_anchors_delta_bbox)] = pos_anchors_delta_bbox
        
        return anchors_match, anchors_delta_bbox
    # deprecated
    #def generate_gt_for_rpn(self, anchors, image_id):
        #'''
        #inputs:
            #image_id: the image for which to generat gt rpn data
            #anchors: [num_of_anchor, (y1, x1, y2, x2)]
        #returns:
            #anchors_match: [num_of_anchor, ]
            #anchors_delta_bbox: [RPN_TRAIN_ANCHORS_PER_IMAGE, (ct_dy, ct_dx, logdh, logdw)], zeros padded
        #'''
        #bboxes, bbox_class = self.load_bbox(image_id)
        ## TODO: in order to get window, padding etc. info, we have to call load_image, try to avoid this
        #image = self.load_image(image_id)
        #image_resized, window, ratio, padding = self.resize_image(image,
                                                        #2*[self.config.IMAGE_MAX_DIM])
        ## bboxes in (x1, y1, w, h)
        #bboxes = self.resize_bbox(bboxes, ratio, padding)
        ## change format to (y1, x1, y2, x2)
        #bboxes = self.convert_x1y1wh_to_y1x1y2x2(bboxes) 
        ##anchors = self.convert_x1y1wh_to_y1x1y2x2(anchors)
        
        ## 
        #ious = self.IoU_many_to_many(anchors, bboxes)
        
        ## place holders
        #anchors_match = np.zeros(len(anchors))
        ## the number of training anchors is set to, say, 256 with positive ones in the front and padded with zeros to make sure same length for each batch.
        #anchors_delta_bbox = np.zeros([self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4])
        
        ## chooes negative anchors
        #anchors_argmax_ids = np.argmax(ious, axis=1)
        #anchors_max = ious[range(len(anchors)), anchors_argmax_ids]
        #neg_ids = np.where(anchors_max < 0.3)[0]
        #anchors_match[neg_ids] = -1
        
        ## positive ones, make sure every gt box has at least one anchor, even when iou < 0.3
        ## make the highest scored anchors posivite
        #gt_argmax_ids = np.argmax(ious, axis=0)
        #anchors_match[gt_argmax_ids] = 1
        ## for > 0.7
        #pos_ids = np.where(anchors_max > 0.7)[0]
        #anchors_match[pos_ids] = 1
        
        ## balance pos and neg, pos should not be more than half
        ## update pos_ids
        #pos_ids = np.where(anchors_match == 1)[0]
        
        #extra = len(pos_ids) - self.config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2
        #if extra > 0:
            #ids = np.random.choice(pos_ids, extra, replace=False)
            #anchors_match[ids] = 0
            ## update pos_ids
            #pos_ids = np.where(anchors_match == 1)[0]
            
        #extra = len(neg_ids) - (self.config.RPN_TRAIN_ANCHORS_PER_IMAGE - len(pos_ids))
        #if extra > 0:
            #ids = np.random.choice(neg_ids, extra, replace=False)
            #anchors_match[ids] = 0
        
        
        ## fill anchors_delta_bbox
        #pos_anchors = anchors[pos_ids]
        #gt_bboxes = bboxes[anchors_argmax_ids[pos_ids]]
        
        ## change format to (x1, y1, w, h)
        #pos_anchors = self.convert_y1x1y2x2_to_x1y1wh(pos_anchors) 
        #gt_bboxes = self.convert_y1x1y2x2_to_x1y1wh(gt_bboxes)
        
        #an_x1, an_y1, an_w, an_h = np.split(pos_anchors, 4, axis=1)
        #gt_x1, gt_y1, gt_w, gt_h = np.split(gt_bboxes, 4, axis=1)
        
        ##ct_dx = (gt_x1 + gt_w/2)-(an_x1 + an_w/2)
        ##ct_dy = (gt_y1 + gt_h/2)-(an_y1 + an_h/2)
        
        #gt_xc = gt_x1 + gt_w/2
        #gt_yc = gt_y1 + gt_h/2
        #an_xc = an_x1 + an_w/2
        #an_yc = an_y1 + an_h/2
        ## use relative value
        #ct_dx = (gt_xc - an_xc) / an_w
        #ct_dy = (gt_yc - an_yc) / an_h
        #logdw = np.log(gt_w / an_w)
        #logdh = np.log(gt_h / an_h)
        
        ##pos_anchors_delta_bbox = np.concatenate([ct_dx, ct_dy, logdw, logdh], axis=1)
        #pos_anchors_delta_bbox = np.concatenate([ct_dy, ct_dx, logdh, logdw], axis=1)
        
        #pos_anchors_delta_bbox /= self.config.RPN_BBOX_STD_DEV
        
        #anchors_delta_bbox[:len(pos_anchors_delta_bbox)] = pos_anchors_delta_bbox
        
        #return anchors_match, anchors_delta_bbox
    
    def IoU_one_to_many(self, box, boxes, area, areas):
        '''
        inputs in (y1, x1, y2, x2) format
        inputs:
            box: (4,)
            boxes: (n, 4)
            area: scalar, area of box
            areas: (n,), areas of boxes
        returns:
            iou: (n,)
        '''
        y1 = np.maximum(box[0], boxes[:,0])
        x1 = np.maximum(box[1], boxes[:,1])
        y2 = np.minimum(box[2], boxes[:,2])
        x2 = np.minimum(box[3], boxes[:,3])
        
        intersections = np.maximum(y2-y1, 0)*np.maximum(x2-x1, 0)
        unions = areas + area - intersections
        iou_one_to_many = intersections / unions
        
        return iou_one_to_many

    def IoU_many_to_many(self, boxes1, boxes2):
        '''
        return the ious of two set of boxes
        inputs in (y1, x1, y2, x2) format
        inputs:
            boxes1: (n1, 4)
            boxes2: (n2, 4)
        returns:
            ious: (n1, n2)
        '''
        ious = np.zeros([len(boxes1), len(boxes2)])
        
        areas1 = (boxes1[:, 2] - boxes1[:, 0])*(boxes1[:, 3] - boxes1[:, 1])
        areas2 = (boxes2[:, 2] - boxes2[:, 0])*(boxes2[:, 3] - boxes2[:, 1])
        
        for i, box in enumerate(boxes1):
            ious[i,:] = self.IoU_one_to_many(box, boxes2, areas1[i], areas2)
            
        return ious

    def convert_x1y1wh_to_y1x1y2x2(self, boxes):
        x1, y1, w, h = np.split(boxes, 4, axis=1)
        x2 = x1 + w
        y2 = y1 + h
        return np.concatenate([y1, x1, y2, x2], axis=1)

    def convert_y1x1y2x2_to_x1y1wh(self, boxes):
        y1, x1, y2, x2 = np.split(boxes, 4, axis=1)
        w = x2 - x1
        h = y2 - y1
        return np.concatenate([x1, y1, w, h], axis=1)

    def resize_image(self, image, shape):
        '''
        inputs:
            image: image to be resized
            shape: target shape
        returns:
            image_resized:
            window: [x1, y1, w, h]
            ratio:
            padding:
        '''
        h, w = image.shape[:2]
        H, W = shape
        image_dtype = image.dtype
        padding = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int)
                    
        # resize image
        if H / h <= W / w:
            # resize
            ratio = H/h
            w_new = np.floor(ratio * w)
            image = skimage.transform.resize(image, [H, w_new], 
                                             preserve_range=True,
                                             mode='constant',
                                             anti_aliasing=False)
            
            # padding
            dw =  W - w_new        
            padding[1][0] = dw // 2 
            padding[1][1] = dw - padding[1][0]
            padding = padding.astype(int)
            
            # get window
            window = np.array([dw//2, 0, w_new, H], dtype=np.int)
            image_resized = np.pad(image, padding, mode='constant', constant_values=0)
        else:
            # resize
            ratio = W / w
            h_new = np.floor(ratio * h)
            image = skimage.transform.resize(image, [h_new, W], 
                                             preserve_range=True,
                                             mode='constant',
                                             anti_aliasing=False)
            
            # padding
            dh = H - h_new
            padding[0][0] = dh // 2 
            padding[0][1] = dh - padding[0][0]
            padding = padding.astype(int)
            
            # get window
            window = np.array([0, dh//2, W, h_new], dtype=np.int)
            
            image_resized = np.pad(image, padding, mode='constant', constant_values=0)

            # note: we have to directly return image_resized.astype(np.uint8) other copies will not work
            #image_resized = image_resized.astype(image_dtype)
            #image_resized = np.copy(image_resized.astype(np.uint8))
        return image_resized.astype(np.uint8), window, ratio, padding

    def resize_mask(self, masks, window, padding):
        '''
        inputs:
            masks: image to be resized
            window: usually for resize_iamge()
            padding: usually for resize_iamge()
        returns:
            masks_resized:
        '''
        # window size for resize
        win_w, win_h = window[2:]
        # build place holder
        H = win_h + padding[0][0] + padding[0][1]
        W = win_w + padding[1][0] + padding[1][1]
        num_masks = masks.shape[-1]
        
        masks_resized = np.zeros([H, W, num_masks], dtype=np.bool)
        
        # resize and pad each mask
        for i in range(num_masks):
            mask = masks[:,:,i]
            mask = skimage.transform.resize(mask, [win_h, win_w])
            masks_resized[:,:,i] = np.pad(mask, padding[:2], mode='constant', constant_values=0)    
            
        return masks_resized.astype(np.bool)

    def resize_bbox(self, bboxes, ratio, padding):
        '''
        inputs:
            masks: image to be resized
            ratio: usually for resize_iamge()
            padding: usually for resize_iamge()
        returns:
            bboxes_resized: [n, (x1, y1, w, h)]
        '''
        num_bboxes = bboxes.shape[0]
        x1, y1, w, h = np.split(bboxes, 4, axis=1)
        
        x1_new = x1 * ratio + padding[1][0]
        y1_new = y1 * ratio + padding[0][0]
        w_new = w * ratio
        h_new = h * ratio
        
        bboxes_resized = np.concatenate([x1_new, y1_new, w_new, h_new], axis=1)
        return bboxes_resized 

    def data_generator(self):
        '''
        this function works as a generator
        input:
            data_path: path to the dataset
        return:
            first list:
                batch_image:                [BATCH_SIZE, h, w, 3]
                batch_anchors_match:        [BATCH_SIZE, anchors, 1]
                batch_anchors_delta_bbox:   [BATCH_SIZE, RPN_TRAIN_ANCHORS_PER_IMAGE, 4]
                batch_mrcnn_classes:        [BATCH_SIZE, MAX_GT_INSTANCES]
                batch_mrcnn_bboxes:         [BATCH_SIZE, MAX_GT_INSTANCES, 4]
                batch_mrcnn_masks:          [BATCH_SIZE, h, w, MAX_GT_INSTANCES]
            second list:
                []
                    
        '''
        # generator 
        # batch_index loops every batch and image_index loops every epoch
        batch_index = 0
        image_id = -1
        # build an id pool for shuffling
        id_pool = np.array(range(self.num_of_images))
        # generate anchors for rpn gt creating
        anchors = generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES, 
                                           self.config.RPN_ANCHOR_RATIOS, 
                                           compute_backbone_shapes(self.config.IMAGE_SHAPE,
                                                                  self.config), 
                                           self.config.BACKBONE_STRIDES, 
                                           self.config.RPN_ANCHOR_STRIDE)
        while True:
            try:
                image_id = (image_id + 1) % self.num_of_images
                if image_id == 0:
                    np.random.shuffle(id_pool)
                image_id = id_pool[image_id]

                # load image and resize 
                image = self.load_image(image_id)
                image, window, ratio, padding = self.resize_image(image, [self.config.IMAGE_MAX_DIM]*2)
                
                ## load bboxes and resize
                # bboxes: [n, (x1, y1, w, h)], classes: [n,]
                bboxes, classes = self.load_bbox(image_id)
                bboxes = self.resize_bbox(bboxes, ratio, padding)
                # convert to (y1, x1, y2, x2) format
                bboxes = self.convert_x1y1wh_to_y1x1y2x2(bboxes)
                
                ## load masks and resize
                # masks: [h, w, n]
                masks, _ = self.load_mask(image_id)
                masks = self.resize_mask(masks, window, padding)
                
                if self.config.USE_MINI_MASK:
                    masks = minimize_mask(masks, self.config.MASK_SHAPE, bboxes)
                    
                # generate rpn gt
                anchors_match, anchors_delta_bbox = self.generate_gt_for_rpn(anchors, bboxes)
                
                # creat batch placeholders
                if batch_index == 0:
                    batch_image = np.zeros((self.config.BATCH_SIZE,) + 
                                           tuple(self.config.IMAGE_SHAPE), dtype=image.dtype)
                    # for rpn
                    batch_anchors_match = np.zeros((self.config.BATCH_SIZE, 
                                                    anchors.shape[0], 1), dtype=np.int32)
                    
                    batch_anchors_delta_bbox = np.zeros((self.config.BATCH_SIZE, 
                                                         self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 
                                                         4), dtype=np.float32)
                    # for mrcnn
                    batch_mrcnn_classes = np.zeros((self.config.BATCH_SIZE,
                                                  self.config.MAX_GT_INSTANCES),
                                                    dtype=np.int32)
                    batch_mrcnn_bboxes = np.zeros((self.config.BATCH_SIZE,
                                                   self.config.MAX_GT_INSTANCES, 4),
                                                    dtype=np.float32)
                    batch_mrcnn_masks = np.zeros((self.config.BATCH_SIZE,
                                                  masks.shape[0],
                                                  masks.shape[1],
                                                  self.config.MAX_GT_INSTANCES), 
                                                    dtype=masks.dtype)
                # add to batch
                # for rpn , same element length
                batch_image[batch_index] = image
                batch_anchors_match[batch_index] = anchors_match[:, np.newaxis]
                batch_anchors_delta_bbox[batch_index] = anchors_delta_bbox
                
                # if the num of instance is larger than MAX_GT_INSTANCES, random choice
                if len(classes) > self.config.MAX_GT_INSTANCES:
                    ids = np.random.choice(range(len(classes)), self.config.MAX_GT_INSTANCES, replace=False)
                    batch_mrcnn_classes = batch_mrcnn_classes[ids]
                    batch_mrcnn_bboxes = batch_mrcnn_bboxes[ids]
                    batch_mrcnn_masks = batch_mrcnn_masks[:,:,ids]
                    
                # for mrcnn, different element lengthes    
                batch_mrcnn_classes[batch_index, :classes.shape[0]] = classes
                batch_mrcnn_bboxes[batch_index, :classes.shape[0]] = bboxes
                batch_mrcnn_masks[batch_index, :, :, :classes.shape[0]] = masks
                
                batch_index += 1
                
                if batch_index >= self.config.BATCH_SIZE:
                    
                    # note: returned varibles need to be correspond to inputs and outputs
                    # here inputs = [batch_image, batch_class], outputs = []
                    yield [batch_image, 
                           batch_anchors_match, batch_anchors_delta_bbox,
                           batch_mrcnn_classes, batch_mrcnn_bboxes, batch_mrcnn_masks], []
                    
                    # start a new batch
                    batch_index = 0
                
            except (GeneratorExit, KeyboardInterrupt):
                raise
############################################################
#  Anchors
############################################################
# modifed base on matterport's code
    
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    note:
        anchors are in [y1, x1, y2, x2]
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    print('Feature map size: {}, number of anchors: {}'.format(shape, len(boxes)))
    return boxes.astype(np.float32)

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    boxes = np.concatenate(anchors, axis=0)
    print('Total number of anchors: {:10}'.format(len(boxes)))
    return boxes

# to be modefied
def plot_image_with_anchors(image, anchors, config):
    '''
    TODO: modifed for all shapes
    '''
    if config.IMAGE_MAX_DIM == 1024:
        backbone_shapes = np.array([[256, 256],[128,128],[64,64],[32,32,],[16,16]])
    else:
        print('This function only deal with [1024, 1024, 3] images')
        exit
    # plot    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # some anchors information  
    levels = backbone_shapes.shape[0]
    colors = np.random.random([levels, 3])
    anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
    
    # get anchors for each level
    anchors_per_level = []
    for i in range(levels):
        anchors_per_level.append(backbone_shapes[i][0] * \
                                 backbone_shapes[i][1] * \
                                 anchors_per_cell //     \
                                 config.RPN_ANCHOR_STRIDE**2)
    # find center anchors for each level
    for i in range(levels):
        start = np.sum(anchors_per_level[:i])

        center_cell = backbone_shapes[i] // 2
        center_anchor_id = start + (center_cell[0] * backbone_shapes[i, 1] / config.RPN_ANCHOR_STRIDE**2 +  center_cell[1] /config.RPN_ANCHOR_STRIDE) * anchors_per_cell
        
        center_anchor_id = int(center_anchor_id)

        for k, rect in enumerate(anchors[center_anchor_id : center_anchor_id + anchors_per_cell]):
            y1, x1, y2, x2 = rect
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, facecolor='none',
                              edgecolor=(k+1)*np.array(colors[i]) / anchors_per_cell)
            ax.add_patch(p)

    ax.imshow(image)
    plt.show()

def apply_deltas_to_boxes(boxes, deltas, config):
    '''
        boxes: [pos_nums, (y1, x1, y2, x2)]
        #deltas: [m, (ct_dx, ct_dy, logdw, logdh)]
        deltas: [m, (ct_dy, ct_dx, logdh, logdw)]
        new_boxes: [pos_nums, (yy1, xx1, yy2, xx2)]
    '''
    pos_nums = len(boxes)
    
    y1, x1, y2, x2 = np.split(boxes, 4, axis=1)
    
    #ct_dx, ct_dy, logdw, logdh = np.split(deltas[:pos_nums] * config.RPN_BBOX_STD_DEV, 4, axis=1)
    ct_dy, ct_dx, logdh, logdw = np.split(deltas[:pos_nums] * config.RPN_BBOX_STD_DEV, 4, axis=1)
    
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w/2
    yc = y1 + h/2
    
    ww = w * np.exp(logdw)
    hh = h * np.exp(logdh)
    # times w, h
    xxc = xc + ct_dx * w
    yyc = yc + ct_dy * h
    
    yy1 = yyc - hh/2
    xx1 = xxc - ww/2
    yy2 = yyc + hh/2
    xx2 = xxc + ww/2  
    
    return np.concatenate([yy1, xx1, yy2, xx2], axis=1)
############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

# from matterport
# TODO 
class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)
    
def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]
############################################################
#  MaskRCNN model
############################################################
class MaskRCNN(object):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config
        self.set_log_dir()
        self.build()

    def load_weights(self, model_path, by_name=True):
        '''
        load weights from trained model for detection
        inputs:
            path to model
        '''        
        import h5py
        from keras.engine import saving
        
        with h5py.File(model_path, 'r') as f:
            saving.load_weights_from_hdf5_group_by_name(f, self.model.layers)
            
    def load_layer_weights(self, model_path, layer_range=[None, None], by_name=True):
        '''
        load a specific range of weights to the model
        None for the very beginning or the very end layer
        TODO: expand to multiple input layers and nultiple output layers
        '''
        start = layer_range[0]
        end = layer_range[1]
        
        # get layer names
        layers = self.model.layers
        layer_names = [l.name for l in layers]
        
        # check input names
        if start == None:
            start = layer_names[0]
        if end == None:
            end = layer_names[-1]

        assert start in layer_names and end in layer_names, "'start' and 'end' must in layer_names"
        layer_names = np.array(layer_names)
        
        # check name order
        start_id = np.where(layer_names == start)[0][0]
        end_id = np.where(layer_names == end)[0][0]
        assert end_id >= start_id, 'start layer should be prior to the end layer'
        
        # build a temperary model for weights loading
        tmp_model = KM.Model(layers[start_id].input, layers[end_id].output)
        tmp_model.load_weights(model_path, by_name=by_name)
        
        # creat a dir to save tmp weights
        tmp_weigths_dir = os.path.join(self.checkpoint_dir, 'tmp_weigths')
        if not os.path.exists(tmp_weigths_dir):
            os.makedirs(tmp_weigths_dir)
            
        # save weights of the tmp_model
        file_name = 'tmp_weigths_from_{}_to_{}.h5'.format(start, end)
        file_dir = os.path.join(tmp_weigths_dir, file_name)
        partial_weights = tmp_model.save_weights(file_dir)
        
        # load tmp_weights to new model
        self.load_weights(file_dir, by_name=by_name)
        print('Layers from "{}" to "{}" are loaded to the model from "{}"'.format(start, end, model_path))
        
    def train(self, train_dataset, val_dataset):
        # add loss
        # clear pervious losses
        self.model._losses = []
        self.model._per_input_losses = {}
        
        # add new losses
        loss_names = ['loss_rpn_class', 'loss_rpn_bbox', 'loss_mrcnn_class', 'loss_mrcnn_bbox', 'loss_mrcnn_mask'] # 
        for name in loss_names:
            layer = self.model.get_layer(name)
            #self.model.add_loss(layer.output)
            loss = (tf.reduce_mean(layer.output, keep_dims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.model.add_loss(loss) 
            
        # add L2 regularization loss
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.model.add_loss(tf.add_n(reg_losses))
        
        # compile the model
        optimizer = keras.optimizers.SGD(lr=self.config.LEARNING_RATE,
                                               momentum=self.config.LEARNING_MOMENTUM)
        # loss should be set to [None] * number of inputs
        self.model.compile(optimizer=optimizer,
                           loss=[None] * len(self.model.outputs))
        # add metrics for losses
        for name in loss_names:
            layer = self.model.get_layer(name)
            loss = (tf.reduce_mean(layer.output, keep_dims=True) * \
                                            self.config.LOSS_WEIGHTS.get(name, 1.))
            self.model.metrics_names.append(name)
            self.model.metrics_tensors.append(loss)
        for name in ['loss_rpn_class', 'loss_rpn_bbox', 'loss_mrcnn_class', 'loss_mrcnn_bbox', 'loss_mrcnn_mask']: # 
            layer = self.model.get_layer(name)
            self.model.metrics_names.append(name)
            self.model.metrics_tensors.append(layer.output)
            
        # create a checkpoint_dir if not exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # callbacks    
        callbacks = [
            #keras.callbacks.TensorBoard(log_dir=self.checkpoint_dir,
                                        #histogram_freq=1, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True)]        
        
        train_generator = train_dataset.data_generator()
        val_generator = val_dataset.data_generator()
        
        #print(next(train_generator))
        #print(next(val_generator))
        
        self.model.fit_generator(
            train_generator,
            epochs = 50, #self.config.EPOCHS,
            steps_per_epoch = self.config.STEPS_PER_EPOCH,
            validation_data = next(val_generator),
            validation_steps = self.config.VALIDATION_STEPS,
            callbacks = callbacks,
            max_queue_size=100,
            workers=0,
            use_multiprocessing=True)
    
    def RPN_graph(self, feature_map):
        '''
        inputs:
            feature_map: feature map from entire image
            anchors_per_location: 
            anchor_stride: usually 1 or 2
        returns:
            #rpn_probs: [batch, anchors, 2]
            #rpn_bbox: [batch, anchors, 4]
            rpn_class_logits: [batch, anchors, 2]
            rpn_class_probs: [batch, anchors, 2]
            rpn_reg_logits: [batch, anchors, 4] (ct_dy, ct_dx, logdh, logdw)
        note: 
            no flatten and dense layer is involved
            softmax is used for 2 classes classification instead of using sigmoid for binary classification
        '''    
        #for each FM, anchors_per_location is len(config.RPN_ANCHOR_RATIOS)
        #instead of len(self.config.RPN_ANCHOR_SCALES) *len(self.config.RPN_ANCHOR_RATIOS)
        
        anchors_per_location = len(self.config.RPN_ANCHOR_RATIOS)
        anchor_stride = self.config.RPN_ANCHOR_STRIDE
        
        shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride, name='rpn_conv_shared')(feature_map)
        
        # classification
        x = KL.Conv2D(anchors_per_location * 2, (1, 1), activation='linear', name='rpn_class_raw')(shared)
        # reshape
        rpn_class_logits = KL.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 2]), name='rpn_cls_reshape')(x)
        rpn_class_probs = KL.Activation('softmax')(rpn_class_logits)
        
        
        # regression
        x = KL.Conv2D(anchors_per_location * 4, (1, 1), activation='linear', name='rpn_bbox_pred')(shared)
        # reshape
        rpn_reg_logits = KL.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 4]), name='rpn_reg_reshape')(x)
        
        return [rpn_class_logits, rpn_class_probs, rpn_reg_logits]

    def build_rpn_model(self):
        # input
        input_feature_map = KL.Input(shape=(None, None, self.config.TOP_DOWN_PYRAMID_SIZE), name="input_rpn_feature_map")
        # output
        outputs = self.RPN_graph(input_feature_map)
        # return model make sure the name is the same as in matterport, so pre-trained weights can be imported for debugging   
        return KM.Model(input_feature_map, outputs, name="rpn_model")        

    # from matterport, TODO why we need to restore anchors
    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(image_shape, self.config)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            #self._anchor_cache[tuple(image_shape)] = norm_boxes(a, image_shape[:2])
            self._anchor_cache[tuple(image_shape)] = a
        return self._anchor_cache[tuple(image_shape)]

    def build(self):
        input_image = KL.Input(shape=(None, None, self.config.IMAGE_SHAPE[-1]), name='input_image')
        # we set train_bn=False, so we can load weights from matterport, see class BatchNorm(KL.BatchNormalization) for detals
        backbone_features = resnet_graph(input_image, self.config.BACKBONE, stage5=True, train_bn=False)
        
        rpn_features, mrcnn_features = self.C_to_P(backbone_features)
        
        RPN = self.build_rpn_model()
        rpn_outputs = []
        for P in rpn_features:
            rpn_outputs.append(RPN(P))
        # [[l2, c2, r2], [l3, c3, r3], [l4, c4, r4], [l5, c5, r5], [l6, c6, r6]] => [(l2, l3, l4, l5, l6), (c2, c3, c4, c5, c6), (r2, r3, r4, r5, r6)]
        [rpn_class_logits, rpn_class_probs, rpn_reg_logits] = list(zip(*rpn_outputs))
        rpn_class_logits = KL.Concatenate(axis = 1, name='rpn_class_logits')(list(rpn_class_logits))
        rpn_class_probs = KL.Concatenate(axis = 1, name='rpn_class')(list(rpn_class_probs))
        rpn_reg_logits = KL.Concatenate(axis = 1, name='rpn_bbox')(list(rpn_reg_logits))        
        
        # proposal layer
        # input_anchors
        if self.mode == 'train':
            # for training, anchors need to be broadcast to batch size
            # mind this part
            #input_anchors = generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES, 
                                   #self.config.RPN_ANCHOR_RATIOS, 
                                   #compute_backbone_shapes(self.config.IMAGE_SHAPE,
                                                          #self.config), 
                                   #self.config.BACKBONE_STRIDES, 
                                   #self.config.RPN_ANCHOR_STRIDE)
            input_anchors = self.get_anchors(self.config.IMAGE_SHAPE[:2])
            input_anchors = tf.Variable(input_anchors)
            input_anchors = tf.broadcast_to(input_anchors,
                                            tf.concat([[self.config.BATCH_SIZE], tf.shape(input_anchors)], axis=0))
            input_anchors = KL.Lambda(lambda x: input_anchors, name='input_anchors')(input_image)

        # for inference, only one set of anchors is required
        # no need to broadcast    
        else:
            # TODO why input_anchors
            input_anchors = KL.Input(shape=(None, 4), name='input_anchors')

            
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if self.mode=='train' \
                                    else self.config.POST_NMS_ROIS_INFERENCE
        nms_threshold = self.config.RPN_NMS_THRESHOLD
        
        proposals = ProposalLayer(proposal_count, 
                                  nms_threshold, 
                                  self.config,           
                                  name='proposal_layer')([rpn_class_probs,
                                                         rpn_reg_logits,
                                                         input_anchors])
        # for training
        if self.mode == 'train':
            # prepare targets for mrcnn
            # input gt class
            input_gt_class_ids = KL.Input(shape=[None], name='input_gt_class_ids', dtype=tf.int32)
            # input gt bbox
            input_gt_boxes = KL.Input(shape=(None, 4), name='input_gt_boxes')
            # normalize
            #input_norm_gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1: 3]),
                                            #name='input_norm_gt_boxes')(input_gt_boxes)
            # input gt mask
            input_gt_masks = KL.Input(shape=[self.config.MASK_SHAPE[0],
                                            self.config.MASK_SHAPE[1], 
                                            None], name='input_gt_mask', dtype=bool)
            
            rois, target_class_ids, target_bbox, target_masks =\
                DetectionTargetLayer(self.config, name="proposal_targets")([proposals, 
                                                            input_gt_class_ids, 
                                                            input_gt_boxes, 
                                                            input_gt_masks])

            # FPN Network Heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_features, 
                                     [self.config.POOL_SIZE, self.config.POOL_SIZE],
                                     self.config.IMAGE_SHAPE[:2],
                                     self.config.NUM_CLASSES,
                                     train_bn=self.config.TRAIN_BN,
                                     fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)
            
            mrcnn_masks = build_fpn_mask_graph(rois, mrcnn_features,
                                              [self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE],
                                              self.config.IMAGE_SHAPE[:2],
                                              self.config.NUM_CLASSES,
                                              train_bn=self.config.TRAIN_BN)
            
            # Loss functions
            # calculate rpn class and bbox losses
            input_rpn_anchor_match = KL.Input(shape=(None, 1), name='input_rpn_anchor_match', dtype=tf.int32)
            input_rpn_delta_bbox = KL.Input(shape=(None, 4), name='input_rpn_delta_bbox')
                                    
            loss_rpn_class = KL.Lambda(lambda x: rpn_class_loss(*x), name='loss_rpn_class') \
                                    ([input_rpn_anchor_match, rpn_class_logits])  
            loss_rpn_bbox = KL.Lambda(lambda x: rpn_bbox_loss(*x, self.config),name='loss_rpn_bbox') \
                                    ([input_rpn_anchor_match, input_rpn_delta_bbox, rpn_reg_logits]) 
                                
            # calculate mrcnn class, bbox loss and mask losses
            loss_mrcnn_class = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name='loss_mrcnn_class') \
                                        ([target_class_ids, mrcnn_class_logits])
            loss_mrcnn_bbox = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name='loss_mrcnn_bbox') \
                                        ([target_bbox, target_class_ids, mrcnn_bbox])
            loss_mrcnn_mask = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name='loss_mrcnn_mask') \
                                        ([target_masks, target_class_ids, mrcnn_masks])

            # construct model
            inputs = [input_image, \
                      input_rpn_anchor_match, input_rpn_delta_bbox, \
                      input_gt_class_ids, input_gt_boxes, input_gt_masks]
            
            outputs = [rpn_class_logits, rpn_class_probs, rpn_reg_logits, \
                        loss_rpn_class, loss_rpn_bbox, proposals, \
                        rois, \
                        mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_masks, \
                        loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask]
            
            self.model = KM.Model(inputs, outputs)
            # valid inputs: ['res', 'bn', 'rpn', 'fpn', 'mrcnn']
            self.set_non_trainable(['res', 'bn', ])
            self.model.summary()
            
        # for evaluating
        else:
            rois = proposals
            # FPN Network Heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_features, 
                                     [self.config.POOL_SIZE, self.config.POOL_SIZE],
                                     self.config.IMAGE_SHAPE[:2],
                                     self.config.NUM_CLASSES,
                                     train_bn=self.config.TRAIN_BN,
                                     fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(self.config, name="mrcnn_detection")(
                [rois, mrcnn_class, mrcnn_bbox])
 
             # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            
            mrcnn_masks = build_fpn_mask_graph(detection_boxes, mrcnn_features,
                                              [self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE],
                                              self.config.IMAGE_SHAPE[:2],
                                              self.config.NUM_CLASSES,
                                              train_bn=self.config.TRAIN_BN)
            
            inputs = [input_image, input_anchors]
            outputs = [proposals, detections, mrcnn_class, mrcnn_bbox, mrcnn_masks]            
            self.model = KM.Model(inputs, outputs)
            #self.model.summary()
            
    def C_to_P(self, backbone_features):
        C1, C2, C3, C4, C5 = backbone_features
        # upsampling and add
        P5 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1,1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name='fpn_p4_add')([
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1,1), name='fpn_c4p4')(C4),
            KL.UpSampling2D((2, 2), name='fpn_p5_upsample')(P5)])
        P3 = KL.Add(name='fpn_p3_add')([
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1,1), name='fpn_c3p3')(C3),
            KL.UpSampling2D((2, 2), name='fpn_p4_upsample')(P4)])
        P2 = KL.Add(name='fpn_p2_add')([
            KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1,1), name='fpn_c2p2')(C2),
            KL.UpSampling2D((2, 2), name='fpn_p3_upsample')(P3)])
        # conv
        P2 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3,3), padding='same', name='fpn_p2')(P2)
        P3 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3,3), padding='same', name='fpn_p3')(P3)
        P4 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3,3), padding='same', name='fpn_p4')(P4)
        P5 = KL.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3,3), padding='same', name='fpn_p5')(P5)
        P6 = KL.MaxPooling2D((1, 1), strides=2, name='fpn_p6')(P5)
        
        rpn_features = [P2, P3, P4, P5, P6]
        mrcnn_features = [P2, P3, P4, P5]
        
        return rpn_features, mrcnn_features
    
    def set_log_dir(self):
        '''
        set the log_dir for the model to store checkpoint files
        '''
        # start a new training with epoch = 0
        #self.epoch = 0
        
        # set log path for all the logs
        self.root_dir = os.path.abspath('.')
        self.log_dir = os.path.join(self.root_dir, 'logs')
        #self.tensorboard = os.path.join(self.log_dir, 'tensorboard')
        # set checkpoint path for each training
        model_name = self.config.NAME
        now = datetime.datetime.now()
        self.checkpoint_dir = os.path.join(self.log_dir, '{}_{:%Y%m%dT%H%M}'.format(model_name.lower(), now))
        
        # set checkpoint file for each epoch, using palceholder *epoch*, keras will fill it with current epoch index                              
        self.checkpoint_path = os.path.join(self.checkpoint_dir, '{}_*epoch*.h5'.format(model_name.lower()))
        self.checkpoint_path = self.checkpoint_path.replace('*epoch*', '{epoch:04d}')

    def rpn_evaluate(self, image):
        '''
        inputs:
            image: a test image [1024, 1024, 3]
        returns:
            rpn_class_logits: [batch, anchors, 2]
            rpn_class_probs: [batch, anchors, 2]
            rpn_reg_logits: [batch, anchors, 4] (ct_dx, ct_dy, logdw, logdh)
             
        '''
        rpn_class_logits, rpn_class_probs, rpn_reg_logits = self.model.predict(image)
        return rpn_class_logits, rpn_class_probs, rpn_reg_logits
        
    # valid inputs: ['res', 'bn', 'rpn', 'fpn', 'mrcnn']
    def set_non_trainable(self, freeze_list): 
        layers = self.model.layers
        for l in layers:
            for freeze in freeze_list:
                if freeze in l.name:
                    l.trainable = False

############################################################
#  Loss functions
############################################################
def rpn_class_loss(rpn_match, rpn_class_logits):
    '''
    rpn_match: [batch, anchors, 1]
    rpn_class_logits: [batch, anchors, 2] FG and BG
    returns:
        loss: scalar
    '''
    # make it [batch, anchors]
    rpn_match = tf.squeeze(rpn_match, axis=-1)
    # get pos and neg ids, cause netural anchor contributes nothing
    pos_neg_ids = tf.where(tf.not_equal(rpn_match, 0))
    # extract the useful part of logits and match
    pos_neg_logits = tf.gather_nd(rpn_class_logits, pos_neg_ids)
    pos_neg_match = tf.gather_nd(rpn_match, pos_neg_ids)
    # for neg anchors, make -1 labels 0
    neg_ids = tf.where(tf.equal(pos_neg_match, -1))
    pos_neg_match_0 = tf.cast(tf.equal(pos_neg_match, 1), tf.int32)
    # without one-hot key we use sparse, make sure from_logits is True, thus a softmax will be applied here in the function
    losses = K.sparse_categorical_crossentropy(pos_neg_match_0,
                                            pos_neg_logits,
                                            from_logits=True)
    #loss = K.mean(loss)
    loss = K.switch(tf.size(losses) > 0, K.mean(losses), tf.constant(0.0, dtype=tf.float32))
    return loss

#def rpn_bbox_loss(rpn_match, target_bbox, rpn_bbox, config):
    #'''
    #inputs:
        #rpn_match: [batch, anchors, 1]
        #anchors_delta_bbox: [batch, pos_anchors, (ct_dx, ct_dy, logdw, logdh)] 
        #rpn_reg_logits: [batch, anchors, 4]
        #config:
    #returns:
        #loss: scalar
    #'''
    ## make it [batch, anchors]
    #rpn_match = tf.squeeze(rpn_match, axis=-1)
    ## get pos anchors
    #pos_ids = tf.where(tf.equal(rpn_match, 1))
    #pos_rpn_box = tf.gather_nd(rpn_bbox, pos_ids) 
    ## get the number of pos in gt
    #num_of_pos_in_batch = K.sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
    ## extract respective gt
    #gt_boxes = []
    #for i in range(config.IMAGES_PER_GPU):
        #gt_boxes.append(target_bbox[i][:num_of_pos_in_batch[i]])
    #gt_boxes = tf.concat(gt_boxes, axis=0)
    
    ## here we use a l2 loss, matterport used a smoothed l1 loss
    ## if l2 is used  the reg loss will be huge, why ?
    ##losses = keras.losses.mean_squared_error(gt_boxes, pos_rpn_box)
    #losses = smooth_l1_loss(gt_boxes, pos_rpn_box)
    ##loss = K.mean(losses)
    
    #loss = K.switch(tf.size(losses) > 0, K.mean(losses), tf.constant(0.0, dtype=tf.float32))
    #return loss

# from matterport    
def rpn_bbox_loss(rpn_match, target_bbox, rpn_bbox, config):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox:    [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
                                Uses 0 padding to fill in unsed bbox deltas.
    rpn_match:      [batch, anchors, 1]. Anchor match type. 1=positive,
                                -1=negative, 0=neutral anchor.
    rpn_bbox:       [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    # TODO
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0, dtype=tf.float32))
    return loss

def mrcnn_class_loss_graph(target_class_ids, pred_class_logits):
    '''
    inputs:
        target_class_ids:   [batch, TRAIN_ROIS_PER_IMAGE]
        pred_class_logits:  [batch, TRAIN_ROIS_PER_IMAGE, 81]
    '''
    target_class_ids = tf.cast(target_class_ids, tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids,
                                                             logits=pred_class_logits)
    loss = tf.reduce_mean(loss)
    return loss

# valid but need further check
#def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    #'''
    #inputs:
        #target_bbox:        [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
        #target_class_ids:   [batch, TRAIN_ROIS_PER_IMAGE]
        #pred_bbox:          [batch, TRAIN_ROIS_PER_IMAGE, 81, (dy, dx, log(dh), log(dw))]
    #note: only the respective prediction of positive boxes contribute to loss
    #'''
    #target_class_ids = tf.cast(target_class_ids, tf.int32)
    
    ## ids of positive boxes
    #pos_ids = tf.cast(tf.where(tf.greater(target_class_ids, 0)), tf.int32)
    
    ## get pos gt class and boxes
    #pos_cls = tf.gather_nd(target_class_ids, pos_ids)
    #pos_gt = tf.gather_nd(target_bbox, pos_ids)
    
    ## prepare prection ids of positive boxes
    #pred_ids = tf.concat([pos_ids, tf.expand_dims(pos_cls, axis=1)], axis=1)
    #pred_pos = tf.gather_nd(pred_bbox, pred_ids)
    
    #loss = smooth_l1_loss(pos_gt, pred_pos)
    
    #loss = K.switch(tf.size(target_bbox) > 0, loss, tf.constant(0.0, dtype=tf.float32))
    #return K.mean(loss)

# from matterport
def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0, dtype=tf.float32))
    loss = K.mean(loss)
    return loss
    
# valid
def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    '''
    inputs:
        target_masks:        [batch, TRAIN_ROIS_PER_IMAGE, H, W]
        target_class_ids:    [batch, TRAIN_ROIS_PER_IMAGE]
        pred_masks:          [batch, TRAIN_ROIS_PER_IMAGE, H, W, 81] float32
    note: only the respective prediction of positive boxes contribute to loss    
    '''
    # TODO  use mini mask to save memory and accelerate

    target_class_ids = tf.cast(target_class_ids, tf.int32)
    
    pos_ids = tf.cast(tf.where(tf.greater(target_class_ids, 0)), tf.int32)
    
    pos_cls = tf.gather_nd(target_class_ids, pos_ids)
    pos_gt = tf.gather_nd(target_masks, pos_ids)
    
    pred_ids = tf.concat([pos_ids, tf.expand_dims(pos_cls, axis=1)], axis=1)
    
    transposed_masks = tf.transpose(pred_masks, [0,1,4,2,3])
    pred_pos = tf.gather_nd(transposed_masks, pred_ids)

    pos_gt = tf.cast(pos_gt, dtype=tf.float32)
    pred_pos = tf.cast(pred_pos, dtype=tf.float32)

    loss = K.binary_crossentropy(target=pos_gt, output=pred_pos, from_logits=False)
    
    loss = K.switch(tf.size(pos_gt) > 0, loss, tf.constant(0., dtype=tf.float32))
        
    return tf.reduce_mean(loss)

def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss. y_true and y_pred are typically: [N, 4], but could be any shape.
    inputs:
        y_true: [N, 4]
        y_pred: [N, 4]
    returns:
        loss: [N,  4]
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss
############################################################
#  Miscellenous
############################################################
def compute_backbone_shapes(image_shape, config):
    '''
    
    '''
    h, w = image_shape[:2]
    return [[int(np.ceil(h/s)), int(np.ceil(w/s))] for s in config.BACKBONE_STRIDES]
    
def convert_x1y1wh_to_y1x1y2x2(boxes):
    x1, y1, w, h = np.split(boxes, 4, axis=1)
    x2 = x1 + w
    y2 = y1 + h
    return np.concatenate([y1, x1, y2, x2], axis=1)

def minimize_mask(original_masks, size, bboxes):
    '''
    get minimask to reduce memory cost
    use unnormalized boxes
    masks in form [H, W, n]
    '''
    bboxes = bboxes.astype(np.int32)
    num = original_masks.shape[-1]
    mini_masks = np.zeros(size + [num], dtype=np.int32)
    for i in range(num):
        mask = original_masks[:,:,i]
        y1, x1, y2, x2 = bboxes[i]
        m = mask[y1:y2, x1:x2]
        m = skimage.transform.resize(m, size)
        mini_masks[:,:,i] = m.astype(np.int32)
    return mini_masks

def expand_mask(mini_masks, size, bboxes):
    '''
    expande mini masks to larger size 
    masks in form [H, W, n]
    '''
    bboxes = bboxes.astype(np.int32)
    num =  mini_masks.shape[-1]
    masks = np.zeros(size + [num], dtype=np.int32)
    for i in range(num):
        mask = mini_masks[:,:,i]
        y1, x1, y2, x2 = bboxes[i]
        h = y2 - y1
        w = x2 - x1
        m = skimage.transform.resize(mask, [h, w], preserve_range=True)
        masks[y1:y2, x1:x2, i] = m.astype(np.int32)
    return masks

############################################################
#  Proposal Layer
############################################################
class ProposalLayer(KE.Layer):
    '''
    this layer corroperate the rpn results with FMs to prepare inputs for mrcnn heads
    inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)]
    returns:
        proposals: [batch, proposal_count, (y1, x1, y2, x2)]
    '''
    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.config = config
    
    def call(self, inputs):
        rpn_probs = inputs[0]
        rpn_bbox = inputs[1]
        anchors = inputs[2]
        
        # trim off low-scored anchors
        pos_scores = rpn_probs[:,:,1]
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(pos_scores)[1])
        pos_ids = tf.nn.top_k(input=pos_scores, 
                                k=pre_nms_limit, 
                                sorted=True).indices
        
        pre_nms_scores = batch_slice([pos_scores, pos_ids], 
                                         lambda x, y: tf.gather(x, y),
                                         self.config.IMAGES_PER_GPU)        
        pre_nms_bbox_delta = batch_slice([rpn_bbox, pos_ids], 
                                         lambda x, y: tf.gather(x, y),
                                         self.config.IMAGES_PER_GPU)
        pre_nms_anchors = batch_slice([anchors, pos_ids], 
                                      lambda x, y: tf.gather(x, y),
                                      self.config.IMAGES_PER_GPU)
        
        # apple deltas to anchors
        new_boxes = batch_slice([pre_nms_anchors, pre_nms_bbox_delta], 
                        lambda x, y: apply_deltas_to_boxes_graph(x, y, self.config), 
                        self.config.IMAGES_PER_GPU)
        
        # clip boxes to fit the image window
        # normalized
        window = [0, 0, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1]]
        #window = [0, 0, 1, 1]

        new_boxes = batch_slice([new_boxes], 
                                lambda x: clip_box_to_window_graph(x, window),
                                self.config.IMAGES_PER_GPU)
        # non-maximum supression
        def nms(boxes, scores):
            boxes = tf.cast(boxes, tf.float32)
            scores = tf.cast(scores, tf.float32)
            ids = tf.image.non_max_suppression(boxes, 
                                               scores, 
                                               self.proposal_count,
                                               self.nms_threshold,
                                               name='rpn_non_max_suppression')
            proposals = tf.gather(boxes, ids)
            # pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
            
        proposals = batch_slice([new_boxes, pre_nms_scores], 
                                lambda x, y: nms(x, y),
                                self.config.IMAGES_PER_GPU)    
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)
############################################################
#  DetectionTargetLayer
############################################################
def detection_target_graph(proposals, input_gt_class_ids, input_gt_boxes, input_gt_masks, config):
    '''
    attach the classes, bboxes and masks to proposals of one image
    inputs:
        proposals:          [proposal_count, (y1, x1, y2, x2)] 
        input_gt_class_ids: [MAX_GT_INSTANCES]
        input_gt_boxes:     [MAX_GT_INSTANCES, (x1, y1, w, h)]
        input_gt_masks:     [H, W, MAX_GT_INSTANCES]
        config: configration info
    returns:
        rois:               [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        target_class_ids:   [TRAIN_ROIS_PER_IMAGE]
        target_bbox:        [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        target_mask:        [TRAIN_ROIS_PER_IMAGE, H, W]
    notes:
        proposals, target_class_ids, target_bbox and target_mask might be zero padded
    '''
    # trim zeros paddings of class, boxes and masks
    proposals, _ = trim_zero_boxes_graph(proposals)
    
    input_gt_boxes, nonzero_ids = trim_zero_boxes_graph(input_gt_boxes)
    input_gt_class_ids = tf.gather(input_gt_class_ids, nonzero_ids, axis=0)
    input_gt_masks = tf.gather(input_gt_masks, nonzero_ids, axis=2)
    
    # calculate overlaps 
    # make sure two parameters are in (y1, x1, y2, x2)
    IoUs = IoU_graph(proposals, input_gt_boxes)
    
    # get pos and neg ids
    roi_iou_max = tf.reduce_max(IoUs, axis=1)
    pos_ids_pool = tf.where(roi_iou_max >=0.5)[:,0]
    neg_ids_pool = tf.where(roi_iou_max < 0.5)[:,0]
    
    # select pos ids, keep it lower than 33%
    pos_count = tf.cast(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO, tf.int32)

    pos_ids = tf.random.shuffle(pos_ids_pool)[:pos_count]
    pos_count = tf.shape(pos_ids)[0]
    # select neg ids
    neg_count = tf.cast(tf.cast(pos_count, tf.float32) * (1./config.ROI_POSITIVE_RATIO - 1), tf.int32)
    neg_ids = tf.random.shuffle(neg_ids_pool)[:neg_count]
    
    # get pos and neg rois and put them together
    pos_rois = tf.gather(proposals, pos_ids)
    neg_rois = tf.gather(proposals, neg_ids)
    rois = tf.concat([pos_rois, neg_rois], axis=0)
    # get gt id for pos rois
    pos_ious = tf.gather(IoUs, pos_ids)
    target_ids = tf.argmax(pos_ious, axis=1)
    ###
    #target_ids = tf.cond(
    #tf.greater(tf.shape(pos_rois)[1], 0),
    #true_fn = lambda: tf.argmax(pos_rois, axis=1),
    #false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    #)
    ###
    # gt class id
    roi_gt_class_ids = tf.gather(input_gt_class_ids, target_ids)
    
    # gt boxes
    roi_gt_boxes = tf.gather(input_gt_boxes, target_ids)
    # compute gt refinement for bboxes
    roi_gt_boxes_deltas = refinement(pos_rois, roi_gt_boxes, config)

    # get gt masks
    transposed_masks = tf.transpose(input_gt_masks, [2, 0, 1])
    roi_gt_masks = tf.gather(transposed_masks, target_ids)
    
    # padding 
    # pad rois
    padding = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0,padding),(0,0)])

    # pad gt
    padding = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(roi_gt_class_ids)[0], 0)
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, padding)])
    roi_gt_boxes_deltas = tf.pad(roi_gt_boxes_deltas, [(0,padding),(0,0)])
    roi_gt_masks = tf.pad(roi_gt_masks, [(0,padding),(0,0),(0,0)])
    
    return rois, roi_gt_class_ids, roi_gt_boxes_deltas, roi_gt_masks

class DetectionTargetLayer(KE.Layer):
    '''
    this layer generate targets for mrcnn heads, targets will be fed into loss functions 
    inputs:
        proposals:          [BATCH_SIZE, rois, (y1, x1, y2, x2)] 
        input_gt_class_ids: [BATCH_SIZE, MAX_GT_INSTANCES]
        input_gt_boxes:     [BATCH_SIZE, MAX_GT_INSTANCES, (x1, y1, w, h)]
        input_gt_masks:     [BATCH_SIZE, H, W, MAX_GT_INSTANCES]
        config: configration info
    returns:
        rois:               [BATCH_SIZE, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        target_class_ids:   [BATCH_SIZE, TRAIN_ROIS_PER_IMAGE]
        target_bbox:        [BATCH_SIZE, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        target_mask:        [BATCH_SIZE, TRAIN_ROIS_PER_IMAGE, H, W]
    notes:
        proposals, target_class_ids, target_bbox and target_mask might be zero padded
    '''
    def __init__(self, config=None, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config
        
    def call(self, inputs):
        proposals = inputs[0]
        input_gt_class_ids = inputs[1]
        input_gt_boxes = inputs[2]
        input_gt_masks = inputs[3]
        
        outputs = batch_slice(
            [proposals, input_gt_class_ids, input_gt_boxes, input_gt_masks], 
            lambda x, y, z, w: detection_target_graph(x, y, z, w, self.config), self.config.IMAGES_PER_GPU)
        return outputs
    # important
    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


def apply_deltas_to_boxes_graph(boxes, deltas, config):
    '''
        boxes: [m, (y1, x1, y2, x2)]
        deltas: [m, (ct_dy, ct_dx, logdh, logdw)]
        new_boxes: [m, (yy1, xx1, yy2, xx2)]
    note: boxes and deltas are equal in lenght
    '''
    # unify data type
    boxes = tf.cast(boxes, deltas.dtype)   
    
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    
    deltas = deltas * np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])
    ct_dy, ct_dx, logdh, logdw = tf.split(deltas, 4, axis=1)
    
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w/2
    yc = y1 + h/2
    
    ww = w * tf.exp(logdw)
    hh = h * tf.exp(logdh)
    # times w, h
    xxc = xc + ct_dx*w
    yyc = yc + ct_dy*h
    
    yy1 = yyc - hh/2
    xx1 = xxc - ww/2
    yy2 = yyc + hh/2
    xx2 = xxc + ww/2  
    
    return tf.concat([yy1, xx1, yy2, xx2], axis=1)

def clip_box_to_window_graph(box, window):
    '''
        box: [N, (y1, x1, y2, x2)]
        window: (y1, x1, y2, x2)
    '''
    # unify data type
    window = tf.cast(window, box.dtype)

    # split
    by1, bx1, by2, bx2 = tf.split(box, 4, axis=1)
    # note: window is [0, 0, 1024, 1024] no axis=1 here
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    
    y1 = tf.maximum(tf.minimum(by1, wy2), wy1)
    y2 = tf.maximum(tf.minimum(by2, wy2), wy1)
    x1 = tf.maximum(tf.minimum(bx1, wx2), wx1)
    x2 = tf.maximum(tf.minimum(bx2, wx2), wx1)
    
    clipped = tf.concat([y1, x1, y2, x2], axis=1)
    clipped.set_shape((clipped.shape[0], 4))
    return clipped
    
def trim_zero_boxes_graph(x):
    '''get positive boxes and trim zero ones'''
    
    reduce_sum = tf.reduce_sum(tf.abs(x), axis=-1)
    # get ids
    ids = tf.where(tf.not_equal(reduce_sum, 0))
    # remove the last dimemsion if needed
    if ids.get_shape()[-1] == 1:
        ids = tf.squeeze(ids, axis=-1)
    # gather posivite boxes
    boxes = tf.gather(x, ids, axis=0)
    
    return boxes, ids         

def IoU_graph(boxes1, boxes2):
    '''
    return the ious of two set of boxes
    inputs:
        boxes1: (N1, (y1, x1, y2, x2))
        boxes2: (N2, (y1, x1, y2, x2))
    returns:
        iou: (N1, N2), 
    '''
    boxes1 = tf.cast(boxes1, boxes2.dtype)
    # num of boxes1 and boxes2
    N1 = tf.shape(boxes1)[0]
    N2 = tf.shape(boxes2)[0]
    
    # tile the entire boxes1 N2 times 
    # [N1, 4] -> [N1, 1, 4] -> [N1, N2, 4] -> [N1*N2, 4]
    boxes1 = tf.expand_dims(boxes1, 1)
    boxes1 = tf.tile(boxes1, [1, N2, 1])
    boxes1 = tf.reshape(boxes1, [-1, 4])
    
    # tile boxes2 N1 times
    # [N2, 4] -> [N2*N1, 4]
    boxes2 = tf.tile(boxes2, [N1, 1])
    
    # split 
    y1_b1, x1_b1, y2_b1, x2_b1 = tf.split(boxes1, 4, axis=1)
    y1_b2, x1_b2, y2_b2, x2_b2 = tf.split(boxes2, 4, axis=1)
    
    # get max for 1 and min for 2
    y1 = tf.maximum(y1_b1, y1_b2)
    x1 = tf.maximum(x1_b1, x1_b2)
    y2 = tf.minimum(y2_b1, y2_b2)
    x2 = tf.minimum(x2_b1, x2_b2)

    intersection = tf.maximum(y2 - y1, 0) * tf.maximum(x2 - x1, 0)  
    union = (x2_b1 - x1_b1) * (y2_b1 - y1_b1) + \
            (x2_b2 - x1_b2) * (y2_b2 - y1_b2) - \
            intersection
    # [N1*N2, 1]    
    iou = intersection / union      
    iou = tf.reshape(iou, [N1, N2])
    return iou

def refinement(rois, gt_boxes, config):
    '''
    calculate refinement of rois
    inputs:
        rois: [n, (y1, x1, y2, x2)]
        gt_boxes: [n, (y1, x1, y2, x2)]
        #gt_boxes: [n, (x1, y1, w, h)]
    returns:
        boxes_deltas: [n, (dy, dx, log(dh), log(dw))]
    '''
    rois = tf.cast(rois, gt_boxes.dtype)

    # for rois
    ro_y1, ro_x1, ro_y2, ro_x2 = tf.split(rois, 4, axis=1)
    
    ro_w = ro_x2 - ro_x1
    ro_h = ro_y2 - ro_y1
    ro_xc = ro_x1 + ro_w/2
    ro_yc = ro_y1 + ro_h/2

    # for gt
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=1)
    
    gt_w = gt_x2 - gt_x1
    gt_h = gt_y2 - gt_y1
    gt_xc = gt_x1 + gt_w/2
    gt_yc = gt_y1 + gt_h/2
    
    # refinement
    ct_dx = (gt_xc - ro_xc) / ro_w
    ct_dy = (gt_yc - ro_yc) / ro_h
    logdw = tf.log(gt_w / ro_w)
    logdh = tf.log(gt_h / ro_h)
    
    boxes_deltas = tf.concat([ct_dy, ct_dx, logdh, logdw], axis=1)
    boxes_deltas /= config.BBOX_STD_DEV

    return boxes_deltas

def apply_refinement_to_boxes_graph(rois, boxes_deltas, config):
    '''
    calculate refinement of rois
    inputs:
        rois: [n, (y1, x1, y2, x2)]
        boxes_deltas: [n, (dy, dx, log(dh), log(dw))]
    returns:
        new_boxes: [n, (y1, x1, y2, x2)]
    '''
    rois = tf.cast(rois, boxes_deltas.dtype)

    # for rois
    ro_y1, ro_x1, ro_y2, ro_x2 = tf.split(rois, 4, axis=1)
    
    ro_w = ro_x2 - ro_x1
    ro_h = ro_y2 - ro_y1
    ro_xc = ro_x1 + ro_w/2
    ro_yc = ro_y1 + ro_h/2

    # for deltas
    ct_dy, ct_dx, logdh, logdw = tf.split(boxes_deltas * config.BBOX_STD_DEV, 4, axis=1)
    
    # new boxes
    ww = ro_w * tf.exp(logdw)
    hh = ro_h * tf.exp(logdh)
    # times w, h
    xxc = ro_xc + ct_dx * ro_w
    yyc = ro_yc + ct_dy * ro_h
    
    yy1 = yyc - hh/2
    xx1 = xxc - ww/2
    yy2 = yyc + hh/2
    xx2 = xxc + ww/2  
    
    return tf.concat([yy1, xx1, yy2, xx2], axis=1)

def apply_refinement_to_boxes(rois, boxes_deltas, config):
    '''
    calculate refinement of rois
    inputs:
        rois: [n, (y1, x1, y2, x2)]
        boxes_deltas: [n, (dy, dx, log(dh), log(dw))]
    returns:
        new_boxes: [n, (y1, x1, y2, x2)]
    '''
    rois = rois.astype(boxes_deltas.dtype)

    # for rois
    ro_y1, ro_x1, ro_y2, ro_x2 = np.split(rois, 4, axis=1)
    
    ro_w = ro_x2 - ro_x1
    ro_h = ro_y2 - ro_y1
    ro_xc = ro_x1 + ro_w/2
    ro_yc = ro_y1 + ro_h/2

    # for deltas
    ct_dy, ct_dx, logdh, logdw = np.split(boxes_deltas * config.BBOX_STD_DEV, 4, axis=1)
    
    # new boxes
    ww = ro_w * np.exp(logdw)
    hh = ro_h * np.exp(logdh)
    # times w, h
    xxc = ro_xc + ct_dx * ro_w
    yyc = ro_yc + ct_dy * ro_h
    
    yy1 = yyc - hh/2
    xx1 = xxc - ww/2
    yy2 = yyc + hh/2
    xx2 = xxc + ww/2  
    
    return np.concatenate([yy1, xx1, yy2, xx2], axis=1)

# from matterport
def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)

# from matterport
def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)

############################################################
#  FPN Heads
############################################################
def fpn_classifier_graph(rois, mrcnn_features, pool_size, image_size, num_classes, train_bn=True, fc_layers_size=1024):
    '''
    this function predict classes and boxes with the rois 
    inputs:
        rois:               [BATCH_SIZE, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        mrcnn_features:     [P2, P3, P4, P5]
        pool_size:          roi pooling size, usually [7, 7]
        image_size;         size of image usually [1024, 1024]
        num_classes:        NUM_CLASSES, for coco we have 81 (80 + 1)
        train_bn:           boolean, train the BN layers
        fc_layers_size:     usually FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    returns:
        mrcnn_class_logits: [BATCH_SIZE, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES]
        mrcnn_class:        [BATCH_SIZE, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES]
        mrcnn_bbox:         [BATCH_SIZE, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw)]
    notes:
    '''    
    # apply roialign, the out put shape has 5 dimensions, say [batch, ROIS_PER_IMAGE, 7, 7, 256] so we
    # used TimeDistributed layers to each roi later
    # [batch, ROIS_PER_IMAGE, 7, 7, 256]
    x = PyramidROIAlign(pool_size, image_size, name='roi_align_classifier')([rois] + mrcnn_features)
    # [batch, ROIS_PER_IMAGE, 1, 1, 1024]
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, pool_size), name='mrcnn_class_conv1')(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # [batch, ROIS_PER_IMAGE, 1, 1, 1024]
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, [1, 1]), name='mrcnn_class_conv2')(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # [batch, ROIS_PER_IMAGE, 1, 1024]
    x = KL.Lambda(lambda x: K.squeeze(x, 2), name='pool_squeeze1')(x)
    # [batch, ROIS_PER_IMAGE, 1024]
    shared = KL.Lambda(lambda x: K.squeeze(x, 2),name='pool_squeeze2')(x)
    # classification
    # [batch, ROIS_PER_IMAGE, 81]
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    # [batch, ROIS_PER_IMAGE, 81]
    mrcnn_class = KL.Activation('softmax', name='mrcnn_class')(mrcnn_class_logits)
    
    # regression
    # [batch, ROIS_PER_IMAGE, 81 * 4]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)
    # [batch, ROIS_PER_IMAGE, 81, 4]
    shape = K.int_shape(x) 
    mrcnn_bbox = KL.Reshape((shape[1], num_classes, 4), name='mrcnn_bbox')(x)
    
    return mrcnn_class_logits, mrcnn_class, mrcnn_bbox

def build_fpn_mask_graph(rois, mrcnn_features, pool_size, image_size, num_classes, train_bn):
    # TODO use mini mask
    #[batch, ROIS_PER_IMAGE, 14, 14, 256]
    x = PyramidROIAlign(pool_size, image_size, name='roi_align_mask')([rois] + mrcnn_features)
    
    # 4 blocks of conv2D [batch, ROIS_PER_IMAGE, 7, 7, 256]
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding='same'), name='mrcnn_mask_conv1')(x)    
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding='same'), name='mrcnn_mask_conv2')(x)    
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding='same'), name='mrcnn_mask_conv3')(x)    
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding='same'), name='mrcnn_mask_conv4')(x)    
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # deconv s = 2, (2 ,2) [batch, ROIS_PER_IMAGE, 1024, 1024, 256]
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation='relu'), name='mrcnn_mask_deconv')(x)
    # (1, 1) conv2d [batch, ROIS_PER_IMAGE, 1024, 1024, 81]
    
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation='sigmoid'), name='mrcnn_mask')(x)
    
    return x
        
    ## use original mask
    ## [batch, ROIS_PER_IMAGE, 32, 32, 256]
    #x = PyramidROIAlign(pool_size, image_size, name='roi_align_mask')([rois] + mrcnn_features)
    #N = 256
    ## 4 blocks of Conv2DTranspose [batch, ROIS_PER_IMAGE, 64, 64, 256]
    #x = KL.TimeDistributed(KL.Conv2DTranspose(N, (2, 2), strides=2, activation='relu'), name='mrcnn_mask_deconv1')(x)    
    #x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn1')(x, training=train_bn)
    #x = KL.Activation('relu')(x)

    ## [batch, ROIS_PER_IMAGE, 128, 128, 256]
    #x = KL.TimeDistributed(KL.Conv2DTranspose(N, (2, 2), strides=2, activation='relu'), name='mrcnn_mask_deconv2')(x) 
    #x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn2')(x, training=train_bn)
    #x = KL.Activation('relu')(x)

    ## [batch, ROIS_PER_IMAGE, 256, 256, 256]
    #x = KL.TimeDistributed(KL.Conv2DTranspose(N, (2, 2), strides=2, activation='relu'), name='mrcnn_mask_deconv3')(x)
    #x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(x, training=train_bn)
    #x = KL.Activation('relu')(x)

    ## [batch, ROIS_PER_IMAGE, 512, 512, 256]
    #x = KL.TimeDistributed(KL.Conv2DTranspose(N, (2, 2), strides=2, activation='relu'), name='mrcnn_mask_deconv4')(x)
    #x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn4')(x, training=train_bn)
    #x = KL.Activation('relu')(x)

    ## deconv s = 2, (2 ,2) [batch, ROIS_PER_IMAGE, 1024, 1024, 256]
    #x = KL.TimeDistributed(KL.Conv2DTranspose(N, (2, 2), strides=2, activation='relu'), name='mrcnn_mask_deconv5')(x)

    ## (1, 1) conv2d [batch, ROIS_PER_IMAGE, 1024, 1024, 81]
    #x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation='sigmoid'), name='mrcnn_mask')(x)
    
    #return x
############################################################
#  ROIAlign Layer
############################################################
def norm_boxes(boxes, shape):
    '''
    inputs:
        boxes: [m, (y1, x1, y2, x2)]
        shape: [h, w], the target shape according to which to normalize
    returns:
        norm_boxes: [m, (ny1, nx1, ny2, nx2)]
    '''
    if type(shape) != tf.float32:
        shape = tf.cast(shape, tf.float32)
    if type(boxes) != tf.float32:
        boxes = tf.cast(boxes, tf.float32)

    shift = tf.constant([0., 0., 1., 1.], dtype=tf.float32)
    scale = tf.tile(shape, [2]) - 1
    
    norm_boxes = (boxes - shift)/scale
    return norm_boxes

def denorm_boxes(norm_boxes, shape):
    '''
    inputs:
        norm_boxes: [m, (ny1, nx1, ny2, nx2)]
        shape: [h, w], the target shape according to which to denormalize
    returns:
        boxes: [m, (y1, x1, y2, x2)]
    '''
    if type(shape) != tf.float32:
        shape = tf.cast(shape, tf.float32)
    if type(norm_boxes) != tf.float32:
        norm_boxes = tf.cast(norm_boxes, tf.float32)

    shift = tf.constant([0., 0., 1., 1.], dtype=tf.float32)
    scale = tf.tile(shape, [2]) - 1
    
    boxes = norm_boxes * scale + shift
    return boxes

def log2_graph(x):
    x = tf.cast(x, tf.float32)

    return tf.log(x) / tf.log(tf.constant(2., dtype=tf.float32))
    
# TODO stop grad?
class PyramidROIAlign(KE.Layer):
    '''
    This layer takes rois and pyramid feature maps to crop FM for each roi
    params:
        pool_size:      [7, 7] size of pool filter 
        image_size:     [1024, 1024] size of image        
    inputs:
        boxes:          [batch, ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        feature_maps:   [P2, P3, P4, P5] in shape [batch, h, w, deep], usually
                        P2 ~ [batch, 256, 256, 256]
                        P3 ~ [batch, 128, 128, 256]
                        P4 ~ [batch, 64, 64, 256]
                        P5 ~ [batch, 32, 32, 256]
    returns:
        pooled_sorted:  [batch, ROIS_PER_IMAGE, H, W, C], usually 
                        [batch, ROIS_PER_IMAGE, 7, 7 256]
    '''
    def __init__(self, pool_size, image_size, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_size = tuple(pool_size)
        self.image_size = tuple(image_size)
        
    def call(self, inputs):
        boxes = inputs[0]
        feature_maps = inputs[1:]
        return self.get_pooled(boxes, feature_maps)
        
    def get_pooled(self, boxes, feature_maps):

        # get level for each box, P2 ~ P5
        h = boxes[:,:,2] - boxes[:,:,0]
        w = boxes[:,:,3] - boxes[:,:,1]
        level_ids = self.get_level(h, w)

        # normalize boxse for crop_and_resize 
        nboxes = norm_boxes(boxes, shape=self.image_size)
        
        # get pooled fm for each level
        total_ids = []
        pooled = []
        for i, level in enumerate(range(2, 6)):
            # extract ids of the same level
            ids = tf.where(tf.equal(level_ids, level))
            
            # roi alignment
            level_boxes = tf.gather_nd(nboxes, ids)
            crop_ids = tf.cast(ids[:,0], tf.int32)
            
            # work around float16
            #level_boxes = tf.cast(level_boxes, tf.float32)
            level_pooled = tf.image.crop_and_resize(feature_maps[i], level_boxes,
                                                     crop_ids, self.pool_size,
                                                     method="bilinear")
            #level_pooled = tf.cast(level_pooled, tf.float16)
            
            # TODO Stop gradient propogation to ROI proposals ???
            level_boxes = tf.stop_gradient(level_boxes)
            crop_ids = tf.stop_gradient(crop_ids)
            
            # gather all
            total_ids.append(ids)
            pooled.append(level_pooled)
        
        
        total_ids = tf.concat(total_ids, axis=0)
        pooled = tf.concat(pooled, axis=0)
        
        # sorting
        sort_ids = 100000 * total_ids[:, 0] + total_ids[:, 1]
        indices = tf.nn.top_k(sort_ids, tf.shape(sort_ids)[0], sorted=True).indices
        indices = indices[::-1]
        pooled_sorted = tf.gather(pooled, indices)
        
        # reshape to original shape
        shape = tf.concat([tf.shape(boxes)[:2],  tf.shape(pooled)[1:]], axis = 0)
        pooled_sorted = tf.reshape(pooled_sorted, shape)
        
        return pooled_sorted
    
    # TODO different from matterport
    # for boxes without normalization, deprecated
    def get_level(self, h, w):
        '''
        note:
            2 refers to P3
            h, w are non-normalized
        '''
        level = log2_graph(tf.sqrt(h * w) / 1024.0 )
        level = tf.cast(tf.round(level), tf.int32)
        level = tf.minimum(5, tf.maximum(2, 5 + level))
        return level
    
    # for normalized boxes without normalization
    #def get_level(self, h, w):    
        
        ## Equation 1 in the Feature Pyramid Networks paper. Account for
        ## the fact that our coordinates are normalized here.
        ## e.g. a 224x224 ROI (in pixels) maps to P4
        #image_area = tf.cast(1024.0 * 1024.0, tf.float32)
        #roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        #roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        ##roi_level = tf.squeeze(roi_level, 0)
        #return roi_level
    
    # important
    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_size + (input_shape[1][-1], )        

############################################################
#  Detection Layer from matterport
############################################################
def refine_detections_graph(rois, probs, deltas, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_refinement_to_boxes_graph(
        rois, deltas_specific, config)
    # Clip boxes to image window
    window = [0, 0, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]]
    # normalized boxes, so use unit window
    #window = [0, 0, 1, 1]
    refined_rois = clip_box_to_window_graph(refined_rois, window)
    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections

class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        #image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        
        #m = parse_image_meta_graph(image_meta)
        #image_shape = m['image_shape'][0]
        #window = norm_boxes_graph(m['window'], image_shape[:2])
        # [0,0,1024,1024]
        
        # Run detection refinement graph on each item in the batch
        detections_batch = batch_slice(
            [rois, mrcnn_class, mrcnn_bbox],
            lambda x, y, w: refine_detections_graph(x, y, w, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        #[batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)

############################################################
#  Graphs
############################################################
#def batch_slice(inputs, graph, num_of_batch):
    '''
    TODO: think about why this function fials sometimes
    '''
    #if not isinstance(inputs, list):
        #inputs = [inputs]
        
    #outputs = []
    #for i in range(num_of_batch):
        #input_slice = [x[i] for x in inputs]
        #output_slice = graph(*input_slice)
        
        #if not isinstance(output_slice, list):
            #output_slice = [output_slice]
        #outputs.append(output_slice)
            
    #result = tf.concat(outputs, axis=0)
    #return result
    
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result













