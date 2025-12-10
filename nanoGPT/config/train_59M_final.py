out_dir = 'transformer-59M-finalpart4'
device = 'cpu'
compile = False

eval_interval = 250
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
wandb_log = False

dataset = 'music_char_nanogpt'
gradient_accumulation_steps = 2
batch_size = 8
block_size = 128

eval_only = False
init_from = 'scratch'

learning_rate = 3e-4
weight_decay = 1e-4
dropout = 0.1
max_iters = 8000 

decay_lr = True
warmup_iters = 200
lr_decay_iters = max_iters
min_lr = 1e-5

n_layer = 12
n_head = 8
n_embd = 640
