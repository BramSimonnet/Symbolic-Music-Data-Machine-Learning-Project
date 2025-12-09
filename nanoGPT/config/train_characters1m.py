out_dir = 'music-characters-output1m'
device = 'cpu'
compile = False
eval_interval = 250
eval_iters = 200
log_interval = 50
always_save_checkpoint = False
wandb_log = False
wandb_project = 'music'
wandb_run_name = 'music-characters1m'

dataset = 'music_char_nanogpt'
gradient_accumulation_steps = 2
batch_size = 8
block_size = 128
eval_only = False
init_from = 'scratch'
learning_rate = 3e-4
max_iters = 2500
weight_decay = 1e-4
decay_lr = True

warmup_iters = 200
lr_decay_iters = max_iters
min_lr = 1e-5

dropout = 0.1

n_layer = 4
n_head = 4
n_embd = 128