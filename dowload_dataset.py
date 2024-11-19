import os
import zipfile
from tqdm import tqdm

def dowload_and_unzip(zip_file_path, out_path, dataset_name = '3Dircadb1'):
    os.makedirs(os.path.join(os.getcwd(), 'Datset'), exist_ok=True)
    if not os.path.exists(zip_file_path):
        raise Exception(f"Zip file {zip_file_path} not exists")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist() 
        for file in tqdm(file_list, desc="Extracting files", unit="file"):
            zip_ref.extract(file, out_path)
    
    dataset_path = os.path.join(out_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise Exception("Undefined")
    sub_folder_names = ['PATIENT_DICOM', 'LABELLED_DICOM', 'MASKS_DICOM']
    for folder in tqdm(os.listdir(dataset_path), desc = "Extracting sub-folders", unit="folder"):
        if folder in ['.DS_Store']:
            continue
        folder_path = os.path.join(dataset_path, folder)
        for sub_folder in sub_folder_names:
            if sub_folder in ['MESHES_VTK']:
                continue
            zip_subfolder_path = os.path.join(folder_path, sub_folder+'.zip')
            with zipfile.ZipFile(zip_subfolder_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                for file in file_list:
                    zip_ref.extract(file, folder_path)
            os.remove(zip_subfolder_path)

if __name__ == '__main__':
    dowload_and_unzip('./Dataset/3Dircadb1.zip', './Dataset')