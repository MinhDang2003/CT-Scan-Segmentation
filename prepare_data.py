import os
import numpy as np
import pydicom
from glob import glob
import shutil
import dicom2nifti

def create_dicom_groups(dicom_dataset_dir, image_out_path='', label_out_path='', num_slices = 32):
    for index ,patient in enumerate(sorted(glob(dicom_dataset_dir + '/*'), key = lambda x: float(x.split('.')[-1]))):
        
        sub_folders = ['PATIENT_DICOM', 'LABELLED_DICOM']
        
        num_objects = len(glob(patient + f'/{sub_folders[0]}' + '/*'))
        num_folders = int(num_objects / 32)
        folder_name = f'patient_{index}'
        for i in range(num_folders):
            output_path_name = os.path.join(image_out_path, folder_name + '_' + str(i))
            os.mkdir(output_path_name)
            for j, file in enumerate(sorted(glob(patient + f'/{sub_folders[0]}' + '/*'), key = lambda x: float(x.split('_')[-1]))):
                if j == num_slices + 1:
                    break
                shutil.move(file, output_path_name)
                
            output_path_name = os.path.join(label_out_path, folder_name + '_' + str(i))
            os.mkdir(output_path_name)
            for j, file in enumerate(sorted(glob(patient + f'/{sub_folders[1]}' + '/*'), key = lambda x: float(x.split('_')[-1]))):
                if j == num_slices + 1:
                    break
                shutil.move(file, output_path_name)

# create_dicom_groups('./Dataset/3Dircadb1', './Dataset/dicom_groups/images', './Dataset/dicom_groups/labels')
os.makedirs("./Dataset/nifti_files",exist_ok=True)
os.makedirs("./Dataset/nifti_files/images",exist_ok=True)
os.makedirs("./Dataset/nifti_files/labels",exist_ok=True)
in_path_images = './Dataset/dicom_groups/images/*'
in_path_labels = './Dataset/dicom_groups/labels/*'
out_path_images = './Dataset/nifti_files/images'
out_path_labels = './Dataset/nifti_files/labels'
def convert_2_nifty(in_path,out_path):
    images_list = glob(in_path)
    images_list = sorted(
        images_list, 
        key=lambda x: tuple(map(int, x.split("\\")[-1].replace("patient_", "").split("_")))
    )

    for patient in images_list:
        patient_name = os.path.basename(os.path.normpath(patient))
        dicom2nifti.dicom_series_to_nifti(patient, os.path.join(out_path, patient_name + '.nii.gz'))
        
convert_2_nifty(in_path_images, out_path_images)
convert_2_nifty(in_path_labels, out_path_labels)