''' Trainer class for training the model.'''

# Imports
import time
import os
from typing import Tuple, Dict
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from circle_utils import torch_circle_iou

logger = logging.getLogger(__name__)

# Trainer class
class Trainer:
    ''' Trainer class for training the model.
    Also has functions to evaluate the model, save the model and plot the training and validation loss.

    loss = nn.MSELoss()
    Additional metrics for validation and evaluation:
    mean_iou = average of iou for all the images in dataset
    thresholded_iou = % of images with iou >= threshold. for example, thresholded_iou for 0.5 = % of images with iou >= 0.5
    thresholded_ious = dict of thresholded_iou for all the thresholds(0.5, 0.75, 0.9, 0.95))
                for  example, thresholded_ious = {0.5: 0.8, 0.75: 0.6, 0.9: 0.4, 0.95: 0.2}

    '''
    def __init__(self, configs, model, train_loader, val_loader, test_loader, ):
        self.configs = configs
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = configs['device']
        # self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.SmoothL1Loss()

        self.optimizer = self.create_optimizer(configs['optimizer_config'], self.model)
        self.scheduler = self.create_scheduler(configs['scheduler_config'], self.optimizer)

        # # self.optimizer = optim.Adam(self.model.parameters(), lr=configs['lr'])
        # self.optimizer = optim.SGD(self.model.parameters(),
        #                            lr = configs['lr'],
        #                            momentum= configs['momentum'])
        # # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
        #                                            step_size = configs['steplr_step_size'],
        #                                            gamma = configs['steplr_gamma'])

        self.num_epochs = configs['num_epochs']

        self.thresholds = configs['thresholds']

        # Train and validation loss
        self.min_val_loss = float('inf')
        self.train_loss = []
        self.val_loss = []
        self.max_mean_iou = 0.0

        # Load pretrained model if provided
        if configs['pretrained_model'] != 'None' and configs['pretrained_model'] != '':
            self.load_checkpoint(configs['pretrained_model'])


    def create_optimizer(self, optimizer_config:Dict, model:nn.Module) -> optim.Optimizer:
        ''' Create and return an optimizer based on the provided configuration.
        Parameters:
        - optimizer_config (Dict): A dictionary containing the configuration for the optimizer.
        - model (torch.nn.Module): The model for which the optimizer will be created.

        Returns:
        - Optimizer: The created optimizer.
        '''
        opt_type = optimizer_config['type']
        opt_params = optimizer_config['params']

        if opt_type.lower() == 'adam':
            return optim.Adam(model.parameters(), **opt_params)
        elif opt_type.lower() == 'sgd':
            return optim.SGD(model.parameters(), **opt_params)
        else:
            raise ValueError("Unsupported optimizer type")

    def create_scheduler(self, scheduler_config:Dict, optimizer:optim.Optimizer
                         ) -> optim.lr_scheduler._LRScheduler:
        ''' Create scheduler from the scheduler config.
        Parameters:
        - scheduler_config (Dict): A dictionary containing the configuration for the scheduler.
        - optimizer (Optimizer): The optimizer for which the scheduler will be created.

        Returns:
        - _LRScheduler: The created learning rate scheduler.

        '''
        scheduler_type = scheduler_config['type']
        scheduler_params = scheduler_config['params']

        if scheduler_type.lower() == 'steplr':
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_type.lower() == 'reducelronplateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        else:
            raise ValueError("Unsupported scheduler type")



    def train(self) -> None:
        ''' Train the model for a number of epochs specified in the configuration.

        This method does following training process, including:
            - running training epochs,
            - validation,
            - updating learning rates, and
            - saving checkpoints.

        '''
        logger.info('Training the model...')
        self.model.to(self.device)

        for epoch in range(self.num_epochs):

            # Train the model for one epoch
            running_loss = self.train_epoch()
            self.train_loss.append(running_loss)

            # Validate the model
            val_loss, mean_iou, thresholded_ious = self.validate()
            self.val_loss.append(val_loss)

            # Print the metrics
            print(f'Epoch: {epoch+1}, LR: {self._get_lr()}', end=',  ')
            print(f'Loss: {round(running_loss, 4)}, Val Loss: {round(val_loss, 4)}', end=',  ')
            print(f'Mean IOU: {round(mean_iou, 2)} Thresholded IOUs: {thresholded_ious}')

            # Update the scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()


            # Save the model if validation loss is minimum
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.save_checkpoint(epoch)


    def train_epoch(self) -> float:
        ''' Train the model for one epoch. '''
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            loss = self.train_step(inputs, labels)
            running_loss += loss
        return running_loss

    def train_step(self, inputs, labels) -> float:
        ''' Train the model for one step. '''

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self) -> Tuple[float, float, Dict[float, float]]:
        ''' Validate the model on the validation set. '''
        self.model.eval()
        return self.get_metrics(self.val_loader) # val_loss, mean_iou, thresholded_ious


    def evaluate(self, data_loader: torch.utils.data.DataLoader = None
                 ) -> Tuple[float, float, Dict[float, float]]:
        ''' Evaluate the model on the given data loader.'''

        print('Evaluating the model...')
        if data_loader is None:
            data_loader = self.test_loader
        self.model.to(self.device)
        self.model.eval ()
        loss, mean_iou, thresholded_ious = self.get_metrics(data_loader)
        print(f'Loss: {loss}, Mean IOU: {mean_iou}, Thresholded IOUs: {thresholded_ious}')
        return loss, mean_iou, thresholded_ious

    def get_metrics(self, data_loader: torch.utils.data.DataLoader
                    ) -> tuple[float, float, dict[float, float]]:
        ''' Get metrics for the given data loader.

        Returns:
        - Tuple[float, float, Dict[float, float]]: A tuple containing:
                loss: float (loss for all the images in dataset),
                mean_iou: float (average of iou for all the images in dataset),
                thresholded_ious: dict (thresholded_iou for all the thresholds(0.5, 0.75, 0.9, 0.95))
        '''
        loss = 0.0
        mean_iou = 0.0
        thresholded_ious = {threshold: 0.0 for threshold in self.thresholds}
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                curr_loss = self.criterion(outputs, labels)
                loss += curr_loss.item()

                for i, output in enumerate(outputs):
                    curr_iou = torch_circle_iou(output, labels[i]).item()
                    mean_iou += curr_iou
                    for threshold in self.thresholds:
                        if curr_iou >= threshold:
                            thresholded_ious[threshold] += 1

        mean_iou /= len(data_loader)
        for threshold in self.thresholds:
            # get percentage of images with iou >= threshold, from 0 to 100. Not 0 to 1.
            thresholded_ious[threshold] = thresholded_ious[threshold] * 100 / len(data_loader.dataset)


        return loss, mean_iou, thresholded_ious

    def predict(self, inputs):
        ''' Predict the parameters of the circle for the given inputs. '''
        self.model.eval()
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        return outputs

    def save_checkpoint(self, epoch):
        ''' Save checkpoint of the model. '''
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'configs': self.configs,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }

        # Save the checkpoint to model name in saved/checkpoints folder
        checkpoint_dir = os.path.join(self.configs['cwd'], self.configs['checkpoint_dir'])
        checkpoint_path = os.path.join(checkpoint_dir, self.configs['model_name'] + \
                            f'_{int(self.max_mean_iou*100)}_{time.strftime("%Y%m%d_%H%M%S")}.pth')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, checkpoint_path)



    def load_checkpoint(self, checkpoint_path):
        ''' Load the checkpoint of the model. '''
        try:
            checkpoint_path = os.path.join(self.configs['cwd'], checkpoint_path)
            if not os.path.exists(checkpoint_path):
                raise ValueError(f'Checkpoint {checkpoint_path} does not exist.')

            if self.device == torch.device('cpu'):
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(checkpoint_path)

            # print(f'Loading checkpoint {checkpoint_path}...')
            logger.info('Loading checkpoint %s...', checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.num_epochs = checkpoint['epoch']
            # self.configs = checkpoint['configs']
            self.train_loss = checkpoint['train_loss']
            self.val_loss = checkpoint['val_loss']
            logger.info('Loaded checkpoint %s successfully.', checkpoint_path)
        except Exception as e:
            logger.error('Error loading checkpoint %s. %s', checkpoint_path, e)
            raise e

    def _get_lr(self):
        ''' Get the learning rate of the optimizer. '''
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _set_lr(self, lr):
        ''' Set the learning rate of the optimizer. '''
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def plot(self):
        ''' Plot the training and validation loss. '''
        plt.plot(self.train_loss, label='Training Loss')
        plt.plot(self.val_loss, label='Validation Loss')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    pass
