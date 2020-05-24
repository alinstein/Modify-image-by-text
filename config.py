import argparse

# ----------------------------------------
# Some nice macros to be used for arparse(Refered from UVIC Assignements)
def str2bool(v):
    return v.lower() in ("true", "1")

# ----------------------------------------
# Arguments for the main program
def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    # ----------------------------------------
    # Arguments for the TEXT AND IMAGE COMPOSITION model
    parser.add_argument('--model', type=str,
                        default='tirg',
                        help="TIRG model(not applicable for this project)")
    parser.add_argument('--embed_dim', type=int, 
                        default=512,
                        help="Dimesion of embedded space")
    parser.add_argument('--loss', type=str, 
                        default='soft_triplet')
    parser.add_argument('--learning_rate', type=float, 
                        default=1e-2)
    parser.add_argument('--learning_rate_decay_frequency', type=int,
                        default=9999999)
    parser.add_argument('--domain_trans', type=str2bool,
                        default=False,
                        help='Domain trasfer from 3D to 2D')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-6)
    parser.add_argument('--num_iters', type=int, 
                        default=210000)
    parser.add_argument("--resume_TIRG", type=str2bool,
                        default=True,
                        help="Whether to resume training of TIRG from existing checkpoint")  
    # ----------------------------------------
    # Arguments for the STACKGAN model (STAGE)
    parser.add_argument("--stage", type=int,
                        default=1,
                        help="Training Stage GAN (start with trainging Stage 1)")
        # ----------------------------------------
    # Arguments for the STACKGAN model 
    parser.add_argument("--rep_intv", type=int,
                        default=2,
                        help="Report interval")
    parser.add_argument("--nc", type=int,
                        default=3,
                        help="Number of channles in the training images. \
                        For coloured images this is 3")
    parser.add_argument("--nz", type=int,
                        default=100,
                        help=" Size of the Z latent vector (the input to the generator)")
    parser.add_argument("--ngf", type=int,
                        default=64,
                        help="# Size of feature maps in the generator. \
                        The depth will be multiples of this.")
    parser.add_argument("--ndf", type=int,
                        default=64,
                        help="Size of features maps in the discriminator.\
                        The depth will be multiples of this.")
    parser.add_argument("--beta1", type=int,
                        default= 0.5,
                        help="Beta1 hyperparam for Adam optimizer")
    parser.add_argument("--save_epoch", type=int,
                        default=10,
                        help="Save step.")
    parser.add_argument("--LR_DECAY_EPOCH", type=int,
                        default=600,
                        help="Learning Decay step")
    parser.add_argument("--DISCRIMINATOR_LR", type=int,
                        default=2e-4,
                        help="Learning rate for discriminator")
    parser.add_argument("--GENERATOR_LR", type=int,
                        default=2e-4,
                        help="Learnining rate for generator")

    # ----------------------------------------
    # Arguments for the Training
    parser.add_argument("--nepochs", type=int,
                        default=125,
                        help="Number of epoches")
    parser.add_argument('--batch_size', type=int,
                        default=64,
                        help="Default Batchsize 8 stage II and 64 for stage 1")
    # ----------------------------------------
    # Arguments for the Datasets
    parser.add_argument('-f', type=str,
                        default='')
    parser.add_argument('--comment', type=str, 
                        default='test_notebook')
    parser.add_argument('--dataset', type=str,
                        default='css3d')
    parser.add_argument('--dataset_path', type=str,
                        default='./',
                        help="Location of CSS IMAGE")
    parser.add_argument("--location", type=str,
                        default="home/alinsteinjose/alin/project/TIRG_STACK_GAN",
                        help="Current Dictonary")

 
    # ----------------------------------------
    # Arguments for the logs   
    parser.add_argument("--log_dir", type=str,
                        default="./logs",
                        help="Directory to save logs for TIRG model")                                            
    parser.add_argument("--log_dir1", type=str,
                        default="./logsSGAN",
                        help="Directory to save logs for GAN STAGE 1 model")
    parser.add_argument("--log_dir2", type=str,
                        default="./logsSGAN2",
                        help="Directory to save logs for GAN STAGE 2 model")
    # ----------------------------------------
    # Arguments for the saving models                      
    parser.add_argument("--save_dir", type=str,
                        default="./save",
                        help="Directory to save the best TIRG model")
    parser.add_argument("--save_dir_GAN", type=str,
                        default="./saveSGAN",
                        help="Directory to save the best model (StackGAN stage 1)")
    parser.add_argument("--save_dir_GAN2", type=str,
                        default="./saveSGAN2",
                        help="Directory to save the best model (StackGAN stage 2)")
    parser.add_argument("--bestmodel_gan", type=str,
                        default="model_epoch_120.pth",
                        help="Trained model of stage 1 GAN, used for training stage 2")   
    # ----------------------------------------
    # Arguments for the saving images                      
    parser.add_argument("--image_GAN2", type=str,
                        default="./image_GAN2",
                        help="Directory to save the images (GAN Stage 2)")
    parser.add_argument("--image_GAN", type=str,
                        default="./image_GAN",
                        help="Directory to save the images (GAN)")

                 

    parser.add_argument('--loader_num_workers', type=int, 
                        default=4)
  
    args = parser.parse_args()
    return args