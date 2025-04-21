from script.ts_unet1 import main as run_unet1
from script.ts_unet2 import main as run_unet2
from script.ts_unet3 import main as run_unet3
from script.training import main as run_training
from script.sampling import main as run_sampling

if __name__ == "__main__":

    """训练采样一体入口"""
    # run_unet1()  # U-Net + Resblock
    # run_unet2()  # U-Net
    # run_unet3()  # 简单网络
    
    """正常训练采样入口"""
    # run_training("unet1")
    # run_sampling("unet1", 3)