import torch
import os
import numpy as np
import pickle
import argparse
from core.models.model_factory import Model
from core.data_provider.datasets_factory import data_provider
from core.utils import preprocess
from core.trainer import train_eval, test_eval
import matplotlib.pyplot as plt

# ---------- CONFIGURATION ----------
CHECKPOINT_DIR_1 = "checkpoints/mnist_predrnn_gru_final"
CHECKPOINT_DIR_2 = "checkpoints/mnist_predrnn_gru_final_2"
CHECKPOINT_INTERVAL = 1000
MAX_ITERATION = 80000
EVAL_BATCH_SIZE = 8
PICKLE_SAVE_PATH = "eval_results.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
parser.add_argument('--model_name', type=str, default='predrnn_gru')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='128,128,128,128')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=0)
parser.add_argument('--decouple_beta', type=float, default=0.1)

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

# ---------- SAMPLING FUNCTION ----------
def reserve_schedule_sampling_exp(itr):
    step_1, step_2, alpha = 2000, 5000, 2500
    r_eta = 0.5 if itr < step_1 else 1.0 - 0.5 * np.exp(-(itr - step_1) / alpha) if itr < step_2 else 1.0
    eta = 0.5 if itr < step_1 else 0.5 - 0.5 * (itr - step_1) / (step_2 - step_1) if itr < step_2 else 0.0

    def gen_flags(prob, length):
        return np.random.rand(EVAL_BATCH_SIZE, length) < prob

    def frame_mask(token):
        ones = np.ones((args.img_width // args.patch_size,
                        args.img_width // args.patch_size,
                        args.patch_size**2 * args.img_channel))
        zeros = np.zeros_like(ones)
        return np.where(token, ones, zeros)

    r_true_token = gen_flags(r_eta, args.input_length - 1)
    true_token = gen_flags(eta, args.total_length - args.input_length - 1)

    flags = []
    for i in range(EVAL_BATCH_SIZE):
        seq = []
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                seq.append(frame_mask(r_true_token[i, j]))
            else:
                seq.append(frame_mask(true_token[i, j - (args.input_length - 1)]))
        flags.append(seq)

    return np.array(flags).reshape(EVAL_BATCH_SIZE, args.total_length - 2,
                                   args.img_width // args.patch_size,
                                   args.img_width // args.patch_size,
                                   args.patch_size**2 * args.img_channel)

# ---------- MAIN LOOP ----------
if os.path.exists(PICKLE_SAVE_PATH):
    with open(PICKLE_SAVE_PATH, "rb") as f:
        results = pickle.load(f)
else:
    results = {
        "iterations": [],
        "losses": [],
        "mse_frame": [],
        "ssim": [],
        "psnr": [],
        "lpips": []
    }

# Pre-load data once
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

# Evaluate each checkpoint
for itr in range(CHECKPOINT_INTERVAL, MAX_ITERATION + 1, CHECKPOINT_INTERVAL):
    if itr in results["iterations"]:
        continue  # Skip already-evaluated

    print(f"[Eval] Iteration {itr}")
    ckpt_path = os.path.join(CHECKPOINT_DIR_1 if itr <= 25000 else CHECKPOINT_DIR_2,
                             f"model.ckpt-{itr if itr <= 25000 else itr - 25000}")

    model = Model(args)
    model.load(ckpt_path)

    if train_input_handle.no_batch_left():
        train_input_handle.begin(do_shuffle=True)

    ims = train_input_handle.get_batch()
    ims = preprocess.reshape_patch(ims, args.patch_size)
    real_input_flag = reserve_schedule_sampling_exp(itr)

    loss = train_eval(model, ims, real_input_flag, itr)
    mse, ssim_val, psnr_val, lp = test_eval(model, test_input_handle, args, itr)

    results["iterations"].append(itr)
    results["losses"].append(loss.item())
    results["mse_frame"].append(mse)
    results["ssim"].append(ssim_val)
    results["psnr"].append(psnr_val)
    results["lpips"].append(lp)

    with open(PICKLE_SAVE_PATH, "wb") as f:
        pickle.dump(results, f)

# ---------- PLOT ----------
# plt.plot(results["iterations"], results["losses"], marker='o')
# plt.xlabel("Training Iterations")
# plt.ylabel("Training Loss")
# plt.title("Loss Curve from Saved Checkpoints")
# plt.grid(True)
# plt.show()
