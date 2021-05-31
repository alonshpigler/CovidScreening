import pytorch_lightning as pl
import pandas as pd
from torchvision import transforms


class DataModuleMNIST(pl.LightningDataModule):

    def __init__(self,
                 data_dir='',
                 mean = 0,
                 std = 0,
                 plates_split = [[1],[25]],

                 ):
        super().__init__()

        # Directory to store MNIST Data
        if len(data_dir) == 0:
            self.data_dir = ''

        # Defining batch size of our data
        self.batch_size = 32

        mean, std = self._get_mean_and_std()
        # Defining transforms to be applied on the data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomCrop(64,64,4)
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Cr(64,64,4)
        ])

    def prepare_data(self):
        """

        :param args:
            metadata_path: path to image filenames
            plates_split: dict containing:
                train: plates numbers used for training
                test: plates numbers used for test
            split_ratio (float in [0,1]): train-val split param
            target_channel (int): channel to predict

        :return:
        """
        plates = [1, 25]
        train_plates = [25]
        test_plates = [1]
        experiment = 1
        # TODO - implement drawing plates random split
        # modes = ['train', 'test']

        # mock_train =
        df = pd.read_csv(args.metadata_path)

        partitions = split_by_plates(df, experiment, args.plates_split[0], args.plates_split[1])
        partitions['train'], partitions['val'] = train_test_split(np.asarray(partitions['train']),
                                                                  train_size=args.split_ratio,
                                                                  shuffle=True)

        datasets = create_datasets(partitions, args.target_channel)
        dataloaders = create_dataloaders(datasets, partitions, args.batch_size)

        return dataloaders

    def setup(self, stage=None):
        # Loading our data after applying the transforms
        data = datasets.MNIST(self.download_dir,
                              train=True,
                              transform=self.transform)

        self.train_data, self.valid_data = random_split(data,
                                                        [55000, 5000])

        self.test_data = datasets.MNIST(self.download_dir,
                                        train=False,
                                        transform=self.transform)

    def train_dataloader(self):
        # Generating train_dataloader
        return DataLoader(self.train_data,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        # Generating val_dataloader
        return DataLoader(self.valid_data,
                          batch_size=self.batch_size)

    def test_dataloader(self):
        # Generating test_dataloader
        return DataLoader(self.test_data,
                          batch_size=self.batch_size)