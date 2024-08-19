import os
import numpy as np
from predictions import get_embeddings
from utils.preprocessing import extract_fbanks

os.environ['KMP_DUPLICATE_LIB_OK']='True'
 
async def load_data_speaker(labId):
    speaker_path =f'./modelDir/{labId}/speaker/'
    if os.path.exists(speaker_path):
        data_dict = {}
        for dir_name in os.listdir(speaker_path):
            dir_path = os.path.join(speaker_path, dir_name)
            if os.path.isdir(dir_path):
                sub_data = {}
                for file_name in os.listdir(dir_path):
                    if file_name.endswith('.npy'):
                        file_path = os.path.join(dir_path, file_name)
                        key = file_name.replace('.npy', '')  # Sử dụng tên file làm key
                        value = np.load(file_path)  # Load file .npy
                        sub_data[key] = value
                
                data_dict[dir_name] = sub_data
                
        return data_dict
    else:
        return "folder do not exist"


async def show_all_speaker(labId):
    speaker_path =f'./modelDir/{labId}/speaker/'
    if not os.path.exists(speaker_path):
        os.makedirs(speaker_path)
    list_user=os.listdir(speaker_path)
    return {
        "result": list_user
    }
async def add_more_speaker(speech_file_path, speaker_name, labId):
    speaker_path =f'./modelDir/{labId}/speaker/'
    print(speaker_path)
    dir_ = speaker_path + speaker_name
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    fbanks = extract_fbanks(speech_file_path)
    # embeddings = get_embeddings(fbanks)
    # print('shape of embeddings: {}'.format(embeddings.shape), flush=True)
    # mean_embeddings = np.mean(embeddings, axis=0)
    # np.save(speaker_path+speaker_name+'/embeddings.npy',mean_embeddings)
    np.save(speaker_path + speaker_name + "/fbanks.npy", fbanks)
    list_user=os.listdir(speaker_path)
    return {
        "result": list_user
    }


if __name__ == '__main__':
    speaker_list = load_data_speaker("")
    print(speaker_list)
    result=add_more_speaker("modelDir/speaker/DuyTa/sample.wav", "DuyTa", "")
    print(result)



