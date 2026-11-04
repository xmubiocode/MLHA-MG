import argparse
import os
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset.concat_dataset import MyConcatDataset
from model.model_stage2 import LatentDiffusionModel
# from dsagent import DeepSpeedAgent

from utils.lr import LR
from utils.optim import build_optimizer
from utils.seed import set_seed
# from utils.io import load_config
from torch.cuda.amp import GradScaler
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from exp_utils import initialize_exp, get_dump_path, describe_model, set_seed, filter_logging
# from load_config import load_config
from load_config_cross import load_config
from pathlib import Path
from transformers import BertTokenizer, WordpieceTokenizer

logger = logging.getLogger(__name__)

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def main(args, config):
    import os, psutil
    p = psutil.Process(os.getpid())
    print("RSS(GB)=", p.memory_info().rss/1024**3)

    ngpu=2
    # data
    print("Creating dataset")
    train_dataset_list=[]
    list_name=7
    for i in range(list_name):
        train_dataset_list.append(f"smiles_to_smiles_train_{i}")
    # train_dataset_list=["smiles_to_smiles_train","smiles_to_smiles_train_1"]
    dataset=MyConcatDataset(train_dataset_list, args.prop_name)
    print('#data:', len(dataset), torch.cuda.is_available())
    data_loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    # sampler=train_sampler,
                    collate_fn=lambda x: dataset.collate(x),
                    num_workers=8,
                    pin_memory=True,
                    shuffle=False,
                    drop_last=False)
    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    # model
    model = LatentDiffusionModel(args,config, tokenizer=tokenizer, loader_len=len(data_loader) // torch.cuda.device_count())   
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        _ = model.load_state_dict(checkpoint['state_dict'], strict=False)

    # training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, filename='checkpoint_{step}',
                                                       #save_top_k=2, 
                                                       #monitor='epoch',
                                                       #every_n_epochs=1,
                                                       every_n_train_steps=1000,
                                                       )
    trainer = pl.Trainer(accelerator='gpu', devices=ngpu, precision='16-mixed', max_epochs=config['schedular']['epochs'],
                         callbacks=[checkpoint_callback], strategy=DDPStrategy(find_unused_parameters=False), limit_val_batches=0.)
    trainer.fit(model, data_loader, None, ckpt_path=args.checkpoint if args.checkpoint else None)






from pathlib import Path
def parser_args():
    current_dir = Path(__file__).resolve().parent

    # 上一级目录
    parent_dir = current_dir.parent
    print("parent_dir:", parent_dir)
    parser = argparse.ArgumentParser(description='train parameters')
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--mode', type=str, default='train', help='train or test or validation')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--exp_name', default='run', type=str)
    parser.add_argument('--exp_id', default='test', type=str)
    parser.add_argument('--dump_path', default='dump', type=str)
    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logging_step', default=None, type=int)
    parser.add_argument('--max_length', default=None, type=int)


    parser.add_argument('--eval_step', default=None, type=int)
    parser.add_argument('--warmup_step', default=None, type=int)
    parser.add_argument('--total_epochs', default=None, type=int)
    parser.add_argument('--total_steps', default=None, type=int)
    parser.add_argument('--bs_per_gpu', default=None, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=None, type=int)
    parser.add_argument('--lr', default=None, type=float)

    parser.add_argument('--llama_ckpt_path', default=None, type=str)
    parser.add_argument('--dataset_name_list', default=None, nargs='+')
    parser.add_argument('--val_dataset_name_list', default=None, nargs='+')
    parser.add_argument('--dataset_selected_prob', default=None, nargs='+')
    parser.add_argument('--load_ckpt_path_list', default=None, nargs='+')
    parser.add_argument('--restore_ds', action='store_true')
    parser.add_argument('--restore_lr', action='store_true')

    parser.add_argument('--batch_size', type=int, default=96, help='input batch size for training (default: 64)')   
    parser.add_argument('--prop_name', type=str, default='property_name.txt', help='path to the property name file')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default=f"{parent_dir}/pretrained_ckpt", type=str, help='Directory to save the model checkpoints')
    parser.add_argument('--vocab_filename', default='vocab/vocab_bpe_300.txt')
    parser.add_argument(
    '--gen_pretrain_ckpt',
    type=str,
    default='pretrained_ckpt/checkpoint_step.ckpt',
    help='Path to pretrained Generation_model checkpoint (for pre_encoder & property_encoder)'
)


    
    return parser.parse_args()

if __name__ == '__main__':

    args = parser_args()

    initialize_exp(args)
    dump_path = get_dump_path(args)
    set_seed(args.random_seed)
    config = load_config(vars(args)) 

    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    print(args)

    pretrain_config = {
        'property_width': 768,
        'embed_dim': 256,
        'batch_size': 128,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'queue_size':2560,
        'momentum': 0.995,
        'bert_config_text': 'config/config_bert.json',
        'bert_config_property': 'config/config_bert_pp.json',
        'schedular': {'sched': 'cosine', 'lr': 5e-5, 'epochs': 10, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 5e-5, 'weight_decay': 0.02}
    }
    main(args, pretrain_config)
    
