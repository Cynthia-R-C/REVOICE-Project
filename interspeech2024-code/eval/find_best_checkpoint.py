# Program for finding the StutterNet model checkpoint with the lowest cv loss
# Cynthia Chen 11/11/2025

import os
import yaml

def find_best_checkpoint(model_dir):
    best_loss = float('inf')  # initialize best loss to infinity
    best_file = None

    # For each .yaml file in the model directory
    for fname in os.listdir(model_dir):
        if fname.endswith('.yaml'):
            path = os.path.join(model_dir, fname)

            # read file contents
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if 'cv_loss' in data:
                    loss = float(data['cv_loss'])
                    # update best loss
                    if loss < best_loss:
                        best_loss = loss
                        best_file = fname
                    
            except Exception as e:
                print(f'Skipping {fname}: {e}')

    if best_file:
        print(f'Best checkpoint: {best_file}')
        print(f'Lowest cv_loss: {best_loss:.6f}')
    else:
        print('No valid .yaml files found or none contained cv_loss.')

if __name__ == '__main__':
    model_dir = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\stutternet_en'  # adjust path as needed
    find_best_checkpoint(model_dir)
