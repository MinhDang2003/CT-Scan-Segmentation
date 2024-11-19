# CT-Scan-Segmentation

Steps:
1 - create venv
python -m venv segment 
2 - activate venv
source ./segment/bin/activate
3 - install package
pip3 install requirement.txt
4 - extract dataset
python3 ./dowload_dataset.py
5 - prepare data
python3 ./prepare_data.py
6 - train
python3 ./train.py