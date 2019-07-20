import numpy as np
import random
import keras


class Generator(keras.utils.Sequence):
    """ Abstract generator class.
    """

    def __init__(
            self,
            batch_size=1,
            group_method='random',  # one of 'none', 'random'
            shuffle_groups=True,
            sample_num=40000,
    ):
        """ Initialize Generator object.

        Args
            batch_size             : The size of the batches to generate.
            shuffle_groups         : If True, shuffles the groups each epoch.
            sample_num             : sample rate
        """

        self.sample_num = int(sample_num)
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        # Define groups
        self.group_eegs()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_eegnet_classes(self):
        raise NotImplementedError('num_eegnet_classes method not implemented')

    def load_eeg(self, eeg_index):
        """ （eeg）Load an eeg at the eeg_index.
        """
        raise NotImplementedError('load_eeg method not implemented')

    def load_eegnet_classes(self, eeg_index):
        """Load eegnet_classes for an eeg_index"""
        raise NotImplementedError('load_eegnet_classes methood not implemented')

    def load_eeg_group(self, group):
        """ Load eegs for all eegs in a group."""
        return np.array([self.load_eeg(eeg_index) for eeg_index in group])

    def group_eegs(self):
        """ Order the eegs according to self.order and makes groups of self.batch_size."""
        # determine the order of the eegs
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def compute_inputs(self, eeg_group):
        """ Compute inputs for the network using an eeg_group."""
        sample_data_group = []
        for j in eeg_group:
            sample_data = []
            total_samples = len(j)
            for i in np.linspace(0, total_samples, self.sample_num + 2)[1:self.sample_num + 1]:
                frame_data = j[int(i)]
                sample_data.append(frame_data)
            sample_data_group.append(sample_data)
        return np.array(sample_data_group)

    def compute_eegnet_targets(self, group):
        eegnet_labels_batch = np.zeros((len(group), self.num_eegnet_classes()), dtype=keras.backend.floatx())
        for index, eeg_index in enumerate(group):
            eegnet_labels_batch[index][int(self.load_eegnet_classes(eeg_index))] = 1
        return eegnet_labels_batch

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
            Returns:
                inputs:ndarray(batch_size,sample_num,electrode_num)
                targets:eegnet_labels_batch:ndarray(batch_size,num_eegnet_classes)
        """
        # load eegs and annotations
        eeg_group = self.load_eeg_group(group)

        # compute network inputs
        inputs = self.compute_inputs(eeg_group)

        # compute network targets
        eegnet_targets = self.compute_eegnet_targets(group)
        targets = eegnet_targets

        return inputs, targets

    def __len__(self):
        """Number of batches for generator.
        """
        return len(self.groups)

    def __getitem__(self, index):
        """Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)
        return inputs, targets
