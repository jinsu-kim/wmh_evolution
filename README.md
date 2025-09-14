# WMH Evolution Project

- Training code for U-Net and GAN models is located under the `./unet` folder, and training code for diffusion models is in the `./diffusion` folder.  
- To train the diffusion models, run `train_diffusion_unet.py` first and then `train_controlnet.py` in sequence.  
- Store the 2D slice files (`.npy`) under the `./dataset` directory. 
- Within each folder, save baseline slices with the suffix `_base.npy` and follow-up slices with the suffix `_foll.npy`.
