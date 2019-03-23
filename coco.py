

###############################################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
print('############## GPU PREPARATION DONE ##############')
###############################################################

from config import Config
import argparse
import os
import model as modellib

PWD_dir = os.path.abspath('.')

print(PWD_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mask rcnn')
    parser.add_argument('mode',
                       help='train or evaluate')
    parser.add_argument('--dataset',
                        required=True,
                        metavar='</path/to/data>',
                        default=r'/home/yang/Downloads/COCO-dataset',
                       help='set the path of dataset')
    parser.add_argument('--model_path',
                        metavar='</path/to/model>',
                        default=PWD_dir,
                       help='set the path of model')    
    args = parser.parse_args()
    
    data_path = args.dataset
    mode = args.mode
    model_path = args.model_path
    assert mode in ['train', 'evaluate']
    
    print('\n')
    print('{:30}{}'.format('mode: ', mode))
    print('{:30}{}'.format('dataset: ', data_path))
    print('{:30}{}'.format('model_path: ', model_path))
    
    assert mode in ['train', 'evaluate']
    
    # for traning
    if mode == 'train':
        config = Config()
        
        # training data
        train_dataset = modellib.Dataset(data_path, 'train2017', config)
        
        # validation data
        val_dataset = modellib.Dataset(data_path, 'val2017', config)

        # creat model
        model = modellib.MaskRCNN(mode, config)

        model.load_weights(os.path.join(model.root_dir, 'pre_trained_model', 'mask_rcnn_coco.h5'))     
        #model.load_layer_weights(os.path.join(model.root_dir, 'pre_trained_model', 'mask_rcnn_coco.h5'),
                                              #layer_range=[None, 'res5c_out'])
        # train
        model.train(train_dataset, val_dataset)
    
    # for evaluating   
    else:
        config = Config()
        model = modellib.MaskRCNN(mode, config)
        model.load_weights(os.path.join(model.root_dir, 'logs/coco_20190316T1248', 'coco_0001.h5'))
        # TODO
        model.evaluate(image)
    
    
    
    
    
    
    
    
    
    
    
    
    
    

