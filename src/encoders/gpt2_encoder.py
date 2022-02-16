from typing import Dict, Any

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from .utils.bert_self_attention import BertConfig, BertModel
from .masked_seq_encoder import MaskedSeqEncoder
from utils.tfutils import pool_sequence_embedding
from transformers import TFGPT2Model


class GPT2Encoder(MaskedSeqEncoder):
    model = TFGPT2Model.from_pretrained('gpt2', cache_dir = './cache/')
    
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {'self_attention_activation': 'gelu',
                          'self_attention_hidden_size': 768,
                          'self_attention_intermediate_size': 512,
                          'self_attention_num_layers': 3,
                          'self_attention_num_heads': 8,
                          'self_attention_pool_mode': 'weighted_mean',
                          }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)
        

    @property
    def output_representation_size(self):
        # return self.get_hyper('self_attention_hidden_size')
        return 768

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope("gpt2_encoder"):
            self._make_placeholders()

            output_pool_mode = self.get_hyper('self_attention_pool_mode').lower()
            if output_pool_mode == 'gpt2':
                outputs =  GPT2Encoder.model(self.placeholders['tokens'], attention_mask=self.placeholders['tokens_mask'], training = is_train, return_dict=True)
                return outputs.last_hidden_state

            else:
                seq_token_embeddings = GPT2Encoder.model(self.placeholders['tokens'], attention_mask=self.placeholders['tokens_mask'], training = is_train, return_dict=True).last_hidden_state
                print("After")
                seq_token_masks = self.placeholders['tokens_mask']
                seq_token_lengths = tf.reduce_sum(seq_token_masks, axis=1)  # B
                return pool_sequence_embedding(output_pool_mode,
                                               sequence_token_embeddings=seq_token_embeddings,
                                               sequence_lengths=seq_token_lengths,
                                               sequence_token_masks=seq_token_masks)
