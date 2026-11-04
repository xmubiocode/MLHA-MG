import argparse
import os
import torch
from tqdm import tqdm
from utils.seed import set_seed
from exp_utils import initialize_exp, get_dump_path, set_seed


from model.model_stage1 import Generation_model
from model.model_stage2 import LatentDiffusionModel
from load_config_cross import load_config


from rdkit import Chem
from collections import OrderedDict
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import json
from torch.distributions.categorical import Categorical
from rdkit import Chem
import random
import numpy as np
import pickle
import warnings
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
from rdkit import RDLogger
from collections import OrderedDict
from rdkit.Contrib.SA_Score import sascorer
from typing import List, Tuple
import csv
warnings.filterwarnings(action='ignore')




def generate(model, image_embeds, text, input_config, stochastic=False, 
             prop_att_mask=None, k=None):
    text_atts = torch.where(text == 0, 0, 1)
    if prop_att_mask is None:   
        prop_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
    
    token_output = model(text,
                        attention_mask=text_atts,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=prop_att_mask,
                        return_dict=True,
                        is_decoder=True,
                        return_logits=True,
                        )[:, -1, :] 
    
    temperature = args.temperature
    token_output = token_output / temperature
    
    if k:
        p = torch.softmax(token_output, dim=-1)
        if stochastic and hasattr(model, 'use_top_p') and model.use_top_p:

            sorted_probs, sorted_indices = torch.sort(p, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            top_p = 0.95
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            for i in range(p.size(0)):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                p[i, indices_to_remove] = 0
            p = p / p.sum(dim=-1, keepdim=True)
        
        if stochastic:
            output = torch.multinomial(p, num_samples=k, replacement=False)
            return torch.log(torch.stack([p[i][output[i]] for i in range(output.size(0))])), output
        else:
            output = torch.topk(p, k=k)
            return torch.log(output.values), output.indices
    
    if stochastic:
        p = torch.softmax(token_output, dim=-1)
        m = Categorical(p)
        token_output = m.sample()
    else:
        token_output = torch.argmax(token_output, dim=-1)
    
    return token_output.unsqueeze(1)

@torch.no_grad()
def generate_pv(model_gen,model_rdm, input_config, num_gen=100, batch=1000, stochastic=True, 
                prop_input=None, prop_mask=None, k=2, noise_scale=0.6):
    
    print("prop_input:", prop_input)
    print("noise_scale:", noise_scale)
    device = model_gen.device
    model_gen.eval()
    tokenizer = model_gen.tokenizer
    model_rdm.eval()
    
    print(f"PV-to-SMILES generation in {'stochastic' if stochastic else 'deterministic'} manner with k={k}...")
    
    with open('normalize_all(0-6).pkl', 'rb') as w:
        norm = pickle.load(w)
    property_mean, property_std = norm
    prop_input_normalized = ((prop_input - property_mean[-32:-1]) / property_std[-32:-1])
    prop_input_normalized = prop_input_normalized.unsqueeze(0).repeat(batch, 1)
    prop_input_normalized = prop_input_normalized.float()
    print("prop_input_normalized:", prop_input_normalized.shape)

    property_feature = model_gen.prop_embed(prop_input_normalized.unsqueeze(2).to(device))
    property_unk = model_gen.property_mask.expand(property_feature.size(0), property_feature.size(1), -1)
    mpm_mask_expand = prop_mask.unsqueeze(0).unsqueeze(2).repeat(property_unk.size(0), 1, property_unk.size(2)).to(device)
    property_masked = property_feature * (1 - mpm_mask_expand) + property_unk * mpm_mask_expand
    properties = torch.cat([model_gen.property_cls.expand(property_masked.size(0), -1, -1), property_masked], dim=1)
    
    prop_embeds = model_gen.property_encoder(inputs_embeds=properties, return_dict=True).last_hidden_state
    prop_embeds = prop_embeds.permute(0, 2, 1).contiguous()
    prop_atts = torch.ones(prop_embeds.size()[:-1], dtype=torch.long).to(device)
    print("prop_embeds:", prop_embeds)  ##[batch,32,256]
    sample_shape = ((batch) , *input_config.sample_shape)
    confation_list=[]
    
    count = 0
    while count < num_gen:
        prop_perturb = prop_embeds + 0 * torch.randn_like(prop_embeds) 
        count += batch
        mol_z = model_rdm.p_sample_loop(
            model_rdm,
            sample_shape,
            condition=prop_perturb,
            clip_denoised=True,
            progress=True,
        )
        mol_z = mol_z.permute(0, 2, 1).contiguous()
        print("mol_z stats:", mol_z.shape)
        print("mol_z stats:", mol_z.mean(), mol_z.std(), mol_z.min(), mol_z.max())
        if noise_scale > 0:
            noise = torch.normal(
                mean=torch.zeros_like(mol_z),
                std=noise_scale * torch.ones_like(mol_z)
            )
            mol_z_noisy = mol_z + noise
        else:
            mol_z_noisy = mol_z

        conditon_z = torch.cat([prop_embeds.permute(0, 2, 1).contiguous(), mol_z_noisy], dim=1)
        confation_list.append(conditon_z)
    condation_all = torch.cat(confation_list).to(device)
    print("condation:", condation_all.shape)

    candidate = []
    for i in tqdm(range(num_gen)):
        condation = condation_all[i, :, :].unsqueeze(0)
        for _ in range(1):

            product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(device)

            values, indices = generate(model_gen.SMILES_decoder, condation, product_input,input_config, stochastic=stochastic, k=k)

            product_input = torch.cat([torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(device),indices.view(k, 1)], dim=-1)
            current_p = values.squeeze(0)
            final_output = []
            for _ in range(130):
                values, indices = generate(model_gen.SMILES_decoder, condation, product_input,input_config, stochastic=stochastic, k=k)
                k2_p = current_p[:, None] + values
                product_input_k2 = torch.cat([product_input.unsqueeze(1).repeat(1, k, 1), indices.unsqueeze(-1)], dim=-1)
                if tokenizer.sep_token_id in indices:
                    ends = (indices == tokenizer.sep_token_id).nonzero(as_tuple=False)
                    for e in ends:
                        p = k2_p[e[0], e[1]].cpu().item()
                        final_output.append((p, product_input_k2[e[0], e[1]]))
                        k2_p[e[0], e[1]] = -1e5
                    if len(final_output) >= k ** 2:
                        break
                current_p, i = torch.topk(k2_p.flatten(), k)
                next_indices = torch.from_numpy(np.array(np.unravel_index(i.cpu().numpy(), k2_p.shape))).T
                product_input = torch.stack([product_input_k2[idx[0], idx[1]] for idx in next_indices], dim=0)
            candidate_k = []
            final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:k]
            for p, sentence in final_output:

                cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence[:-1])).replace('[CLS]', '')
                candidate_k.append(cdd)

            if not stochastic:
                if candidate_k:
                    candidate.append(candidate_k[0])
                else:
                    candidate.append("")  
            else:
                if candidate_k:
                    candidate.append(random.choice(candidate_k))
                else:
                    candidate.append("")

    return candidate


def load_training_smiles(json_dir):

    all_smiles = set()
    json_files = []
    for i in range(7):
        json_files.append(os.path.join(json_dir, f"mols_{i}.json"))
    for file in json_files:
        if os.path.exists(file):
            with open(file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    smi = data.get("canonical_smiles")
                    if smi:
                        all_smiles.add(smi)
    return all_smiles


def calculate_property(smiles):
    with open('property_name.txt', 'r') as f:
        names = [n.strip() for n in f.readlines()]

    descriptor_dict = OrderedDict()
    for n in names:
        if n == 'QED':
            descriptor_dict[n] = lambda x: QED.qed(x)
        elif n == "SAS":
            descriptor_dict[n] = lambda x: sascorer.calculateScore(x)
        else:
            descriptor_dict[n] = getattr(Descriptors, n)
            
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if smiles == '' or mol is None:
        return torch.zeros(len(descriptor_dict), dtype=torch.float)
    output = []
    for i, descriptor in enumerate(descriptor_dict):
        output.append(descriptor_dict[descriptor](mol))
    return torch.tensor(output, dtype=torch.float)


@torch.no_grad()
def evaluate_molecules(gen_smiles, train_smiles=None, prop_input=None, mask=None):
    results = {}


    valid_smiles = []
    for smi in gen_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
    results["Validity"] = len(valid_smiles) / len(gen_smiles) if gen_smiles else 0


    unique_smiles = set(valid_smiles)
    results["Uniqueness"] = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0


    if train_smiles is None:
        novel_smiles=-1
        results["Novelty"]=-1
    else:
        novel_smiles = unique_smiles - train_smiles
        results["Novelty"] = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0
    

    with open('normalize_all(0-6).pkl', 'rb') as w:
        norm = pickle.load(w)

    if prop_input is not None:
        prop_input=torch.tensor(prop_input)
        mse=[]
        for i in range(len(gen_smiles)):
            try:
                prop_cdd=calculate_property(gen_smiles[i])
                n_ref = (prop_input - norm[0][-32:-1]) / norm[1][-32:-1]
                n_cdd = (prop_cdd - norm[0][-32:-1]) / norm[1][-32:-1]
                mse.append((n_ref - n_cdd) ** 2)

            except:
                continue
            
        mse = torch.stack(mse, dim=0)
        rmse = torch.sqrt(torch.mean(mse, dim=0))
        results["RMSE"] =  rmse[(1 - mask).long().bool()].mean().item()


    return results, valid_smiles, novel_smiles   





from pathlib import Path 

def main(args, config):
    device = torch.device(args.DEVICE)
    
    seed = 512#1024
    print('seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)


    model_gen = Generation_model(args, config, tokenizer=tokenizer, no_train=True)
    if args.gen_pretrain_ckpt:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.gen_pretrain_ckpt, map_location='cpu')
        state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'word_embeddings' in key and 'property_encoder' in key:
                del state_dict[key]
            if 'queue' in key:
                del state_dict[key]

        msg = model_gen.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.gen_pretrain_ckpt)
        print(msg)
    model_gen = model_gen.to(device)
    
    
    #####Model_RDM####
    model_rdm = LatentDiffusionModel(args,config,tokenizer=tokenizer,no_train=True)
    if args.checkpoint_RDM:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint_RDM, map_location='cpu')
        state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'word_embeddings' in key and 'property_encoder' in key:
                del state_dict[key]
            if 'queue' in key:
                del state_dict[key]

        msg = model_rdm.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint_RDM)
        print(msg)
    model_rdm = model_rdm.to(device)
   
    ####single property generation
    # prop_mask = torch.ones(31)
    # prop_mask[[30]] = 0
    # prop_input = torch.zeros(31)
    # prop_input[30] = 0.6
    
    ###uncondation generation
    prop_mask = torch.ones(31)
    prop_input = torch.zeros(31)
    
    ##multi-property generation
    # prop_mask = torch.ones(31)
    # prop_mask[[28, 18, 29, 30]] = 0
    # prop_input = torch.zeros(31)
    # prop_input[28] = 2
    # prop_input[18] = 1
    # prop_input[29] = 30
    # prop_input[30] = 0.8
        

    train_json_dir = "data/train_data/json_1"
    train_smiles = load_training_smiles(train_json_dir)

    samples, = generate_pv(model_gen, model_rdm, args, num_gen=args.n_generate, prop_input=prop_input, prop_mask=prop_mask, k=args.k, stochastic=args.stochastic)

    os.makedirs(f"{args.output_dir}", exist_ok=True)
    with open(f"{args.output_dir}/smiles_generation.txt", "w") as f:
        for smi in samples:
            f.write(smi + "\n")
        results, valid_smiles, novel_smiles = evaluate_molecules(samples, train_smiles=train_smiles, prop_input=prop_input, mask=prop_mask)
        print("=== Evaluation Results ===")
        print(f"Validity:   {results['Validity']:.2%}")
        print(f"Uniqueness: {results['Uniqueness']:.2%}")
        print(f"Novelty:    {results['Novelty']:.2%}")
        print(f"RMSE:      {results['RMSE']:.4f}")
        print("=" * 50)


def parser_args():
    parser = argparse.ArgumentParser(description='inference parameters')
    parser.add_argument('--mode', type=str, default='inference', help='train or test or validation')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--exp_name', default='run', type=str)
    parser.add_argument('--exp_id', default='test', type=str)
    parser.add_argument('--dump_path', default='dump', type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    
    parser.add_argument('--gen_pretrain_ckpt', type=str, 
                        default='stage1/checkpoint_step.ckpt',
                        help='path to generation model checkpoint')
    parser.add_argument('--checkpoint_RDM', type=str, 
                        default='stage2/checkpoint_step.ckpt',
                        help='path to generation model checkpoint')
    
    parser.add_argument('--output_dir', type=str, default='', help='path to the output directory')
    
    parser.add_argument('--DEVICE', type=str, default='cuda:1', help='device to use')
    parser.add_argument('--config', type=str, default='config/config_bert_pp.json', help='path to the config file')
    parser.add_argument('--base_config', type=str, default='config/base.yaml', help='path to the base config file')
    parser.add_argument('--vocab_filename', default='vocab/vocab_bpe_300.txt',
                        help='vocab file')
    
    # Generation parameters
    parser.add_argument('--n_generate', type=int, default=1000, help='number of molecules to generate')
    parser.add_argument('--k', type=int, default=2, help='beam search width')
    parser.add_argument('--temperature', type=float, default=1, help='temperature for sampling')
    parser.add_argument('--stochastic', default=True,type=bool, help='use stochastic sampling')


    parser.add_argument('--sample_shape', type=List[int], default=(768, 32), help="the number of atoms")
    
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
    

    if not hasattr(args, 'GENERATE'):
        args.GENERATE = {'N_REPEAT': 1}
    
    configs = {
        'embed_dim': 256,
        'bert_config_text': 'config/config_bert.json',
        'bert_config_property': 'config/config_bert_pp.json',
    }
    
    main(args, configs)