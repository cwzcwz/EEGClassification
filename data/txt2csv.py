import numpy as np
import pandas as pd
import os
from tqdm import tqdm

file_dir = "./DissatisfactionOrSatisfaction"
file_names = os.listdir(file_dir)
for filename in tqdm(file_names):
    a = file_dir + '/' + filename
    file_items = os.listdir(a)
    for file_item in file_items:
        if os.path.splitext(file_item)[1] == '.txt' and os.path.splitext(file_item)[0].split('_')[-1] == 'data':
            b = a + '/' + file_item
            data_txt = np.loadtxt(b, dtype=str)
            data_txtDF = pd.DataFrame(data_txt)
            c = a + '/' + file_item.split('.')[0] + '.csv'
            data_txtDF.to_csv(c, index=False)
