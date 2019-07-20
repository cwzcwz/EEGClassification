from preprocess.generator import Generator

import os
import numpy as np

dissaOrsa_class = {
    'satisfaction': 0,
    'dissatisfaction': 1
}


class EEGDataGenerator(Generator):
    """ Generate data for a Pascal VOC dataset."""

    def __init__(
            self,
            data_dir,
            set_name,
            eegnet_classes=dissaOrsa_class,
            eeg_extension='.txt',
            **kwargs
    ):
        self.data_dir = data_dir
        self.set_name = set_name
        self.eegnet_classes = eegnet_classes
        self.eeg_extension = eeg_extension
        self.eeg_names = [l.strip().split(' ')[0] for l in
                          open(os.path.join(data_dir, set_name + '.txt')).readlines()]
        self.eegnet_eeg_classes = [l.strip().split(' ')[1] for l in
                                   open(
                                       os.path.join(data_dir, set_name + '.txt')).readlines()]

        super(EEGDataGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.eeg_names)

    def num_eegnet_classes(self):
        return len(self.eegnet_classes)

    def load_eegnet_classes(self, eeg_index):
        return self.eegnet_eeg_classes[eeg_index]

    def load_eeg(self, eeg_index):
        path = os.path.join(self.data_dir, 'all', self.eeg_names[eeg_index] + self.eeg_extension)
        eeg_file = open(path)
        eeg_datas = []
        for i, line in enumerate(eeg_file.readlines()):
            if i != 0:
                a = line.split(',')[1:]
                eeg_datas.append([float(x.strip()) for x in a])
        return eeg_datas
