import torch
import os
import numpy as np
import pickle
import argparse
from core.models.model_factory import Model
from core.data_provider.datasets_factory import data_provider
from core.utils import preprocess
from core.trainer import train_eval, test_eval

# ---------- ARGUMENTS ----------
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--train_data_paths', type=str, default='moving-mnist-data/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths', type=str, default='moving-mnist-data/moving-mnist-valid.npz')
#parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn_all')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='predrnn_v2')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='128,128,128,128')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=0)
parser.add_argument('--decouple_beta', type=float, default=0.1)
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

# reverse scheduled sampling
parser.add_argument('--reverse_scheduled_sampling', type=int, default=1)
parser.add_argument('--r_sampling_step_1', type=float, default=25000)
parser.add_argument('--r_sampling_step_2', type=int, default=50000)
parser.add_argument('--r_exp_alpha', type=int, default=2500)

# optimization
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()

# ---------- DEVICE ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- DATA ----------
train_input_handle, test_input_handle = data_provider(
    dataset_name=args.dataset_name,
    train_data_paths=args.train_data_paths,
    valid_data_paths=args.valid_data_paths,
    batch_size=args.batch_size,
    img_width=args.img_width,
    seq_length=args.total_length,
    injection_action="concat",
    is_training=True
)

# ---------- EVALUATION ----------
def gen_real_input_flag():
    shape = (args.batch_size, args.total_length - 2,
             args.img_width // args.patch_size,
             args.img_width // args.patch_size,
             args.patch_size**2 * args.img_channel)
    return np.ones(shape)

ORIGINAL_MODEL_PATH = "checkpoints/mnist_model.ckpt"

#print(f"[Loading] Model from: {args.pretrained_model}")
model = Model(args)
model.load(ORIGINAL_MODEL_PATH)

if train_input_handle.no_batch_left():
    train_input_handle.begin(do_shuffle=True)
ims = train_input_handle.get_batch()
ims = preprocess.reshape_patch(ims, args.patch_size)
real_input_flag = gen_real_input_flag()

print("[Evaluating]...")
train_loss = train_eval(model, ims, real_input_flag, itr=0)
mse, ssim_val, psnr_val, lp = test_eval(model, test_input_handle, args, itr=0)

results = {
    "train_loss": train_loss.item(),
    "mse_frame": mse,
    "ssim": ssim_val,
    "psnr": psnr_val,
    "lpips": lp
}

with open("original_results.pkl", "wb") as f:
    pickle.dump(results, f)


