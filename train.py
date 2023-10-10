from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

from PIL import Image
import numpy as np

# Fix error: module 'numpy' has no attribute 'bool' when using numpy 1.24.0+
# Or insyall numpy==1.23.1
np.bool = np.bool_

# Fix error: module 'PIL.Image' has no attribute 'ANTIALIAS'
Image.ANTIALIAS = Image.LANCZOS


def main():
    config = Cfg.load_config_from_name('vgg_transformer')

    dataset_params = {
        'name': 'hw',
        'data_root': './datasets/',
        'train_annotation': 'train_line_annotation.txt',
        'valid_annotation': 'test_line_annotation.txt'
    }

    # data_root: the folder save your all images
    # train_annotation: path to train annotation
    # valid_annotation: path to valid annotation
    # print_every*: show train loss at every n steps
    # valid_every: show validation loss at every n steps
    # iters: number of iteration to train your model
    # export: export weights to folder that you can use for inference
    # metrics: number of sample in validation annotation you use for computing
    #          full_sequence_accuracy, for large dataset it will take too long,
    #          then you can reuduce this number
    params = {
        'print_every': 200,
        'valid_every': 15*200,
        # 'iters': 20000,
        'iters': 6000,
        'checkpoint': './checkpoint/transformerocr_checkpoint.pth',
        'export': './weights/transformerocr.pth',
        'metrics': 10000,
        # 'weights': './models/pretrained/vgg_transformer.pth',
    }

    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cpu'
    # Fix TypeError: cannot pickle 'Environment' object when PC have not GPU -> set num_workers = 0 in Dataloader
    config['dataloader']['num_workers'] = 0

    trainer = Trainer(config, pretrained=True)

    # Save model configuration for inference, load_config_from_file
    trainer.config.save('config.yml')

    # Visualize your dataset to check data augmentation is appropriate
    # trainer.visualize_dataset()

    # Train
    trainer.train()

    # Visualize prediction from our trained model
    # trainer.visualize_prediction()

    # Compute full seq accuracy for full valid dataset
    # trainer.precision()


if __name__ == '__main__':
    main()