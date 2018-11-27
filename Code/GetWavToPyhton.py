from pathlib import Path

import ijson
from scipy.io import wavfile
import pandas as pd
import numpy as np

len_files = 0
k = 0
df_meta = pd.DataFrame()
df_meta_list = []

p = Path('C:\\Users\\srech\\PycharmProjects\\cnn_mnist.py')
filename = "C:\\Users\\srech\\PycharmProjects\\cnn_mnist.py\\Databases\\database.json"
filename_meta = "C:\\Users\\srech\\PycharmProjects\\cnn_mnist.py\\Databases\\meta.json"
audio_file_name_str = "C:\\Users\\srech\\PycharmProjects\\cnn_mnist.py\\Databases\\audio_files\\audio_file{0}.json"

#p = Path('C:\\Users\\jonny\\PycharmProjects\\timit_16kHz_wav\\train')
#filename = "C:\\Users\\jonny\\desktop\\log\\Databases\\database.json"
#filename_meta = "C:\\Users\\jonny\\desktop\\log\\Databases\\meta.json"
#audio_file_name_str = "C:\\Users\\Jonny\\Desktop\\log\\Databases\\audio_files_right\\audio_file{0}.json"

with open(filename, 'r') as f:
    objects = ijson.items(f, 'data.item')
    list_names = list(objects)
    list_wavPath = list(p.glob('**/*.wav'))
    for i in range(len(list_wavPath)):
        wav_str = str(list_wavPath[i])
        rate, audio_data = wavfile.read(wav_str)

        audio_data_proc_mean = audio_data - audio_data.mean()
        audio_data_proc = audio_data_proc_mean/audio_data_proc_mean.std()

        print(i)

        df = pd.DataFrame({'name': list_names[i][1], 'gender': list_names[i][2], 'sample': list_names[i][4], 'audiosignal{0}'.format(i): audio_data_proc})
        df_meta_list.append([list_names[i][1], list_names[i][4], np.mean(audio_data_proc), np.var(audio_data_proc), audio_data_proc.max(), audio_data_proc.size, list_names[i][2]])

        df_meta = pd.DataFrame(df_meta_list, columns=['name', 'sample', 'mean', 'var', 'max', 'sample_length', 'gender'])
        audio_file_name_str = "C:\\Users\\srech\\PycharmProjects\\cnn_mnist.py\\Databases\\audio_files\\audio_file{0}.json".format(i)
        df_json = df.to_json(orient='split')
        with open(audio_file_name_str, 'w') as fs:
            fs.write(df_json)


with open(filename_meta, 'w') as fs:
    meta_json = df_meta.to_json(orient='split')
    fs.write(meta_json)

