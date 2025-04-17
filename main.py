from script.ts_unet1 import main as run_unet1
from script.ts_unet2 import main as run_unet2
from script.ts_unet3 import main as run_unet3
from script.training import main as run_training
from script.sampling import main as run_sampling

if __name__ == "__main__":
    run_unet1()
    # run_unet2()
    # run_unet3()
    # run_training()
    # run_sampling()