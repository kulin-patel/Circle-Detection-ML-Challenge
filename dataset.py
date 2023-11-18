''' Dataset class for generating a dataset of images with circles and corresponding parameters. '''

# Imports
from typing import Generator, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from circle_utils import noisy_circle

# Dataset class
# torch supported dataset class.
class CircleDataset(Dataset):
    ''' Dataset class for generating a dataset of images with circles and corresponding parameters.
    Args:
        config (Dict[str, Any]): Configuration parameters with keys:
            - img_size (int): The size of the images.
            - min_radius (int): Minimum radius of circles.
            - max_radius (int): Maximum radius of circles.
            - noise_level (float): Noise level for the image.
            - num_train_examples (int): Number of training examples.
            - num_val_examples (int): Number of validation examples.
            - num_test_examples (int): Number of test examples.
            - device (str): Device to use for tensor ('cpu' or 'cuda').
        mode (str): Mode of the dataset, one of 'train', 'val', or 'test'.
    '''
    def __init__(self, config: Dict[str, any], mode: str = 'train') -> None:
        self.config = config
        self.mode = mode
        self.img_size = config['img_size']
        self.min_radius = config['min_radius']
        self.max_radius = config['max_radius']
        self.noise_level = config['noise_level']
        self.device = config['device']

        assert self.mode in ['train', 'val', 'test'], "Invalid mode. Choose from 'train', 'val' or 'test'."

        self.num_examples = config['num_train_examples'] if mode == 'train' \
                            else config['num_val_examples'] if mode == 'val' \
                            else config['num_test_examples']
        self.data = self.generate_examples()

    def __len__(self) -> int:
        ''' Get the number of examples in the dataset.'''
        return self.num_examples

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Get an example from the dataset.
        Args:
            idx: int, index of the example. Unused. Added for compatibility with torch dataloader.
        Returns:
            image: torch tensor of shape (1, img_size, img_size)
            parameters: torch tensor of shape ()
        '''
        img, params = next(self.data)
        img = img.unsqueeze(0)

        return img, params

    def generate_examples(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        ''' Generate a dataset of images with circles and corresponding parameters.
            Torch supported generator function.

        Returns:
            image: torch tensor of shape (img_size, img_size)
            parameters: torch tensor of shape (3,). (x,y,r)
        '''

        while True:
            img, params = noisy_circle(self.img_size, self.min_radius, self.max_radius, self.noise_level)
            params = torch.tensor(params, dtype=torch.float32, device=self.device)
            img = torch.tensor(img, dtype=torch.float32, device=self.device)
            yield img, params



if __name__ == '__main__':
    import json
    configs = json.load(open("config.json", encoding='utf-8'))
    configs['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_data_loader = DataLoader(CircleDataset(configs, mode='val'), batch_size=8)

    print(f'Number of batches: {len(val_data_loader)}')
    # To check if the dataset is working properly
    for batch in val_data_loader:
        print(batch[0].shape)
        break
