import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Molecule1dEncoder, Molecule3dEncoder



class EncoderProjector(nn.Module):
    def __init__(self,
                 args,
                 input_modality,
                 hidden_size,
                 encoder_layer,
                 decoder_layer,
                 output_size,
                 num_query_tokens,
                 num_fp):
        super(EncoderProjector, self).__init__()
        self.args = args
        self.input_modality = input_modality

        self.input_encoder_1, self.input_encoder_2, self.input_dim_1, self.input_dim_2 = \
            self.init_input_modality_encoder(input_modality)

        self.encoder_linear_1 = nn.Sequential(
            nn.LayerNorm(self.input_dim_1),
            nn.Linear(self.input_dim_1, hidden_size)
        )
        self.encoder_linear_2 = nn.Sequential(
            nn.LayerNorm(self.input_dim_2),
            nn.Linear(self.input_dim_2, hidden_size)
        )
        self.out_linear = nn.Linear(hidden_size, output_size)

        self.transformer = nn.Transformer(
            d_model=hidden_size,
            num_encoder_layers=encoder_layer,
            num_decoder_layers=decoder_layer,
            batch_first=True
        )

        self.fp_embedding = nn.Linear(num_fp, hidden_size, bias=False)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens - 1, hidden_size))

    @classmethod
    def init_input_modality_encoder(cls, input_modality):
        if input_modality == 'molecule':
            encoder1 = Molecule1dEncoder()
            encoder2 = Molecule3dEncoder()
        else:
            raise NotImplementedError
        return encoder1, encoder2, encoder1.get_dim(), encoder2.get_dim()

    def create_mask(self, src: torch.Tensor, tgt: torch.Tensor):

        device = src.device  
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device, dtype=torch.bool)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device).bool()


        return src_mask, tgt_mask

    def mol_2d_encoder(self,input_batch1,dtype):
        device = next(self.parameters()).device 
        input_emb1, input_mask1 = self.input_encoder_1(input_batch1, device, dtype)
        input_emb1 = self.encoder_linear_1(input_emb1)

        return input_emb1[:,0,:],input_emb1, input_mask1

    def mol_3d_encoder(self,input_batch2,dtype):
        device = next(self.parameters()).device  
        input_emb2, input_mask2 = self.input_encoder_2(input_batch2, device, dtype)
        input_emb2 = self.encoder_linear_2(input_emb2)

        mask = input_mask2.unsqueeze(-1).to(input_emb2.dtype)
        geo_global = (input_emb2 * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        geo_emb_with_global = torch.cat([geo_global.unsqueeze(1), input_emb2],dim=1)
        geo_mask_with_global = torch.cat([torch.ones(input_mask2.shape[0], 1, dtype=torch.bool, device=input_mask2.device),input_mask2],dim=1)
        return geo_global,geo_emb_with_global, geo_mask_with_global


    def forward_from_encoded(self, input_emb1, input_mask1, input_emb2, input_mask2):

        device = input_emb1.device
 
        input_modality_emb = torch.cat([input_emb1, input_emb2], dim=1)
        input_modality_mask = torch.cat([input_mask1, input_mask2], dim=1)
 
        src_key_padding_mask = (~input_modality_mask).to(device)
 
        query_input = self.query_tokens.repeat(input_modality_emb.size(0), 1, 1).to(device)
 
        src_mask, tgt_mask = self.create_mask(input_modality_emb, query_input)
 
        transformer_output = self.transformer(
            src=input_modality_emb,
            tgt=query_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask
        )
 
        transformer_output = self.out_linear(transformer_output)
        return transformer_output

    def forward(self, input_batch, dtype):
        input_batch1, input_batch2 = input_batch
        _, input_emb1, input_mask1 = self.mol_2d_encoder(input_batch1, dtype)
        _, input_emb2, input_mask2 = self.mol_3d_encoder(input_batch2, dtype)
        return self.forward_from_encoded(input_emb1, input_mask1, input_emb2, input_mask2)





