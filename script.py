import sys
import numpy as np
import librosa
from scipy.io.wavfile import write


def get_nonzero_index(audio: np.ndarray) -> list:
    result = []
    length = audio.shape[0]
    i = 0
    while i < length:
        zero_count = 0
        
        while i < length and abs(audio[i]) == 0:
            i += 1
            zero_count += 1
            
        if zero_count > 44100:
            result.append(i)
        
        i += 1
    
    return result


def trim_list(lst:list[np.ndarray]) -> list[np.ndarray]:
    result = []
    for item in lst:
        result.append(np.trim_zeros(item))
    
    return result


def write_wav(data:list[np.ndarray], filename='sound_', sr=44100):
    for i in range(len(data)):
        write(f'{filename}{i + 1}.wav', sr, data[i])


if __name__ == '__main__':
    filename = 'common_voice_v2_1151-1200.m4a'
    if len(sys.argv) == 2:
        filename = sys.argv[1]

    print('-------- SCRIPT START --------')
    
    data, sr = librosa.load(filename, sr=44100)
    
    print('''
    LOADING...
    Please wait
          ''')

    trimmed_data = np.trim_zeros(data) # delete zeros from front and end
    print('''
    Data trimmed      
          ''')
    indices_or_sections = get_nonzero_index(trimmed_data) # get nonzero indices for split
    print('''
    Spliting data...      
          ''')
    splited_data = np.array_split(trimmed_data, indices_or_sections)
    splited_data = trim_list(splited_data) # trim the list again
        
    print('''
    Writing data to files...      
          ''')
    filtered_data = list(filter(lambda x: x.shape[0] > 50000, splited_data)) # remove items that ~0sec
    write_wav(filtered_data)

    print('-------- SCRIPT END --------')
