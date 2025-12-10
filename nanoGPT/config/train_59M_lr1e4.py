# Quick tuning run: lower learning rate (1e-4)

out_dir = 'transformer-59M-lr1e4'
device = 'cpu'
compile = False

eval_interval = 200
eval_iters = 100
log_interval = 50

always_save_checkpoint = False
wandb_log = False

dataset = 'music_char_nanogpt'
gradient_accumulation_steps = 2
batch_size = 8
block_size = 128

eval_only = False
init_from = 'scratch'

#tuning
learning_rate = 1e-4

max_iters = 400

weight_decay = 1e-4
decay_lr = True

warmup_iters = 50
lr_decay_iters = max_iters
min_lr = 1e-5

dropout = 0.1

n_layer = 12
n_head = 8
n_embd = 640
