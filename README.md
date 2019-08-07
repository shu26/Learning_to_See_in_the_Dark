# Learning_to_See_in_the_Dark_PyTorch  
  
Learning to See in the Dark ([paper](http://cchen156.web.engr.illinois.edu/paper/18CVPR_SID.pdf)) implementation in PyTorch, in development.  
    
Auther's codes is [here](https://github.com/cchen156/Learning-to-See-in-the-Dark).  
    
## Requierment  
Required python (3.7) libralies: Pytorch (1.1.0) + Numpy(1.15.0) + Rawpy.  
Tested in Ubuntu + Intel(R) Xeon(R) CPU + NVIDIA GeForce GTX Titan X(Pascal).  
  
## Download Dataset  
Download Dataset `download_dataset.py` from the [original code](https://github.com/cchen156/Learning-to-See-in-the-Dark) and put it in the top level directory of this project and execute: `python download_dataset.py'.  
Need the dataset of Sony images only.  
  
## Training  
To train the Sony model, clone this repository and run `python train.py`.  
The result will be saved in "result_Sony" folder, and the trained model will be saved in "saved_model" by default.  
  
## Testing  
Download the trained model from the [original_code](https://github.com/cchen156/Learning-to-See-in-the-Dark) and put it folder "saved_model".
To test the model, clone this repository and run `python test.py`.  
 
