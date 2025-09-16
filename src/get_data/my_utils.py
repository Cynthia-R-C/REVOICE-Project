import os
import shutil
import stat

def coalesce_data(origin_paths_list, target_path):
    '''Input: a list of paths to extract the data from and the target directory path;
    Copies all data from the list of paths into one place at the target dir path.
    Assumes there are only folders in each path.'''

    for origin_path in origin_paths_list:

        if not os.path.isdir(target_path):  # checks if the folder exists
            os.makedirs(target_path)   # if the folder doesn't exist, create a path to it

        folders = os.listdir(origin_path)  # gives a list of all the folders that exist in the directory

        for folder in folders:

            folder_path = os.path.join(origin_path, folder)
            # gives us the path to each folder - concatenate the name of each folder with the origin_path

            new_path = os.path.join(target_path, folder)

            if os.path.exists(new_path):  # check to see if there's a duplicate path
                raise FileExistsError(f"Duplicate folder detected: '{folder}' already exists in '{target_path}'")

            shutil.copytree(folder_path, new_path, dirs_exist_ok=True)  # copies each folder inside origin path into the new path; dirs_exist_ok means if the folder already exists just merge into it instead of crashing


def filter_data(larger_path, smaller_path):
    '''Checks to see if every thing in larger_path exists in smaller_path; 
    if it doesn't, deletes it; 
    if it does, keeps it.
    Assumes there are only folders in each path.'''

    folders1 = os.listdir(larger_path)  # list of folders in larger path
    folders2 = os.listdir(smaller_path) # list of folders in smaller path

    for folder in folders1:
        if folder not in folders2:  # if the folder doesn't also exist in folder 2

            folder_path = os.path.join(larger_path, folder)  # gives us the path to the folder

            shutil.rmtree(folder_path, onerror=force_remove_readonly)  # deletes the folder and all its contents from the larger path 
            # The second parameter allows shutil to bypass read-only permission to delete the necessary flac files


def force_remove_readonly(func, path, _):
    '''Error handler for shutil.rmtree to handle read-only files.'''

    os.chmod(path, stat.S_IWRITE)
    func(path)         