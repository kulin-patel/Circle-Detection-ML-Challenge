''' Circle Detection ML Challenge'''

import os
import sys
import logging
import json
import torch
from torch.utils.data import DataLoader
from dataset import CircleDataset
from model import CircleDetector2
from trainer import Trainer

def main(config_file):
    ''' Main function for the project.
    Args:
        config_file: str, path to the config file.
    '''
    # 0. Load configs
    try:
        configs = load_config(config_file)
    except FileNotFoundError:
        logging.error("Config file not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error("Config file is not a valid JSON file.")
        sys.exit(1)
    except AssertionError as e:
        logging.error(e)
        sys.exit(1)

    # 1. Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(configs)

    # 2. Build a model to predict the parameters of the circle
    model = CircleDetector2()
    print(model)
    print('Model Size: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 3. Build a trainer to train the model
    trainer = Trainer(configs, model, train_loader, val_loader, test_loader)
    # 3.1  Train the model
    trainer.train()

    # # 4. Evaluate the model on the test set
    # Pretrained model from config will be used if provided.
    # trainer.evaluate()

def load_config(config_file):
    ''' Load and validate configuration from a JSON file.
    And add some additional parameters: device, cwd
    '''

    with open(config_file, 'r', encoding='utf-8') as file:
        configs = json.load(file)

    assert configs['img_size'] > 0, "Image size should be greater than 0."
    assert configs['min_radius'] > 0, "Min radius should be greater than 0."
    assert configs['max_radius'] > 0, "Max radius should be greater than 0."
    assert configs['noise_level'] >= 0 and configs['noise_level'] <= 1, "Noise level should be between 0 and 1."
    assert configs['num_train_examples'] > 0, "Number of examples should be greater than 0."
    assert configs['num_val_examples'] > 0, "Number of examples should be greater than 0."
    assert configs['num_test_examples'] > 0, "Number of examples should be greater than 0."
    assert configs['min_radius'] <= configs['max_radius'], "Min radius should be less than max radius."
    assert configs['max_radius'] <= configs['img_size']/2, "Max radius should be less than half of the image size."
    assert configs['batch_size'] > 0, "Batch size should be greater than 0."
    assert configs['num_epochs'] > 0, "Number of epochs should be greater than 0."
    # assert configs['thresholds'] is a list of 4 floats between 0 and 1
    assert 0 < configs['thresholds'][0] < configs['thresholds'][1] <\
            configs['thresholds'][2] < configs['thresholds'][3] < 1, "Thresholds should be a list of 4 floats between 0 and 1."



    configs['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs['cwd'] = os.getcwd()

    #check if pretrained model is provided
    if configs['pretrained_model'] != 'None' and configs['pretrained_model'] != '':
        assert os.path.isfile(configs['pretrained_model']),  "Please provide a valid path to the pretrained model\
                                                                OR set it to 'None' or '' to train a new model."

    return configs


def get_dataloaders(configs: dict)-> tuple[DataLoader, DataLoader, DataLoader]:
    ''' Get dataloaders for train, val and test sets.'''
    train_loader = DataLoader(CircleDataset(configs, mode='train'), batch_size=configs['batch_size'])
    val_loader = DataLoader(CircleDataset(configs, mode='val'), batch_size=configs['batch_size'])
    test_loader = DataLoader(CircleDataset(configs, mode='test'), batch_size=configs['batch_size'])
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main("config.json")
    