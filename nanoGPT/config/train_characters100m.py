out_dir = 'music-characters-output100m'
device = 'cpu'
compile = False
eval_interval = 1000
eval_iters = 200
log_interval = 100
always_save_checkpoint = False

wandb_log = False
wandb_project = 'music'
wandb_run_name = 'music-characters100m'

dataset = 'music_characters'
gradient_accumulation_steps = 2
batch_size = 8
block_size = 512

learning_rate = 3e-4
max_iters = 200000
weight_decay = 1e-4
decay_lr = True

warmup_iters = 200
lr_decay_iters = max_iters
min_lr = 1e-5

dropout = 0.1

n_layer = 16
n_head = 16
n_embd = 768
