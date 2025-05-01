# AIChickenCoop

Due to the large size of the trained model files, they have been uploaded to an external location: https://drive.google.com/drive/folders/17OQg4lPGYvOzETDl56jKib-Igw2VSIbZ?usp=sharing  
Similarly, because of the size of the training dataset, only a test subset is available at: https://drive.google.com/drive/folders/1fTgv6XhMIQDeJ1x3GQX_SdcbVhqbidak?usp=sharing  


### Create Virtual Environment
Before using the code, create and activate a Python virtual environment:  
bash  
python -m venv venv  
.venv\Scripts\activate  

### Install Dependencies
pip install -r requirements.txt  

### Training the model
Download the test dataset, ensure it is in ImgLabelling/TestVids  
Launch GroundedSAM.ipynb and run all cells.  

### Check annotations  
run AnnotationChecking.py in AnnotationCleaners  

### Detect Chickens  
If you are downoading the trained models, ensure they are in Training/trained_models  
run YOLOModelDetecting.py or YOLOModelDetectingWithTracking.py  
