# CT-Scan-Segmentation

## Steps

### 1. Create Virtual Environment
Run the following command to create a virtual environment:
```bash
python -m venv segment
```

### 2. Activate Virtual Environment
Activate the virtual environment using:
```bash
source ./segment/bin/activate
```

### 3. Install Required Packages
Install all the required dependencies from the `requirements.txt` file:
```bash
pip3 install -r requirements.txt
```

### 4. Extract Dataset
Run the following script to download and extract the dataset:
```bash
python3 ./download_dataset.py
```

### 5. Prepare Data
Prepare the data for training by running:
```bash
python3 ./prepare_data.py
```

### 6. Train the Model
Start training the model with:
```bash
python3 ./train.py
```

---

This README outlines the steps to set up the project, prepare data, and train the CT-Scan segmentation model. If you encounter any issues or have questions, feel free to reach out.
