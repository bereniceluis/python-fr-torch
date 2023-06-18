# python-fr-torch

## Usage
Clone repository
```
git clone https://github.com/bereniceluis/python-fr-torch.git
cd python-fr-torch
```  

Requirements
```
pip install -r requirements.txt
```

Download weights
| Pre-trained               | Backbone            | **Description**                                                                | Download link                                                                                | Size     |
|:--------------------------|:--------------------|:-------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:---------|
| backbone.pth              | iresnet100          | for feature extraction with arcface loss function. save in backbones folder    | [File](https://drive.google.com/file/d/1TVfnDTCYa1bS9Yat-h2SAos0qjAwN3vI/view?usp=drive_link)| 249.1 MB |
| yolov5m-face.pt           | YOLO5-CSPNet        | used for face detection. save in weights folder                                | [File](https://drive.google.com/file/d/1bP86MtZNFQ-c8dgYf_-UuGAthnzSdmFr/view?usp=drive_link)| 161.3 MB |
| yolov5s-face.pt           | YOLO5-CSPNet        | used for face detection, but much smaller in size. save in weights folder      | [File](https://drive.google.com/file/d/11oKjCKTVVTXqX5T9GJ9mdPAxCu9eZS2S/view?usp=drive_link)| 54.4 MB  |
| features.npz              |                     | extracted face feature embeddings. create a static folder and put the npz file | [File](https://drive.google.com/file/d/1IDO2YmtWMhgUdhfczsEIKGDTRN0N3auA/view?usp=drive_link)| 1.2 MB   |  

Overall directory structure

```
├── dataset                      # For adding training dataset
│   ├── add-train-data           # add photos
│   ├── face-datasets            # all datasets are saved here
│   └── trained-data             # all trasined datasets are saved here
├── model_arch                   # where models are saved
│   ├── backbones # Model Weights
│   │   ├── __init__.py
│   │   ├── backbone.pth         # save the weights here
│   │   └── iresnet.py           # resnet architecture
├── static
│   ├── features.npz             # facial embeddings. will be used for similarity matching.
├── yolov5_face                  # cloned from yolov5face repository (unnecessary files are removed)
│   ├── models
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── yolo.py
│   │   ├── experimental.py
│   │   └── other yaml files..
│   ├── utils
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── autoanchor.py
│   │   ├── datasets.py
│   │   └── other util files..
│   └── detect_face.py
├── fr.py                        # facial recognition
├── train.py                     # train added dataset
```

## Training

Create folders inside the add-train-data folder
```
├── dataset                      
│   ├── add-train-data           
│   │   ├── lastname_firstname
│   │   │   ├── pic1.jpg
│   │   │   ├── pic2.jpg
│   │   │   └── pic3.jpg
│   │   └── lastname_firstname
│   │       ├── pic1.jpg
│   │       ├── pic2.jpg
│   │       └── pic3.jpg
│   ├── face-datasets
│   └── trained-data   
```
Run in terminal
```
python train.py --is-add-user=True
```
Note: After training an .npz file will be saved inside the static folder.  

Do face recognition
```
python fr.py
```

