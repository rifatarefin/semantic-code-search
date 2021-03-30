import os
from typing import Dict, Any
from transformers import GPT2Tokenizer, TFGPT2Model, GPT2Config
import tensorflow as tf
# from .utils.bert_self_attention import BertConfig, BertModel
from .masked_seq_encoder import MaskedSeqEncoder
from utils.tfutils import pool_sequence_embedding


# from transformers import GPT2Config

class GPT2Encoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {'token_vocab_size': 50257,
                          'token_vocab_count_threshold': 10,
                          'token_embedding_size': 768,
                          'use_subtokens': False,
                          'mark_subtoken_end': False,

                          'max_num_tokens': 200,
                          'self_attention_pool_mode': 'weighted_mean',
                          'self_attention_hidden_size': 768,
                          'batch_size': 2,
                          'use_bpe': True,
                          'pct_bpe': 0.5
                          }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        return self.get_hyper('self_attention_hidden_size')

    def make_model(self, is_train: bool = False):
        # with tf.compat.v1.variable_scope("gpt2_encoder_" + name):
        self._make_placeholders()
        """
        GPT-2 uses Transformer's decoder as a building block, excluding the encoder-decoder attention module.
        Thus, the only difference with Bert's building blocks(Transformer's encoder) is the masked attention.
        However, in this implementation the masked attention is used for the BertEncoder.
        Therefore the BertModel will be used and adjust the hyper-parameters to be the same of those of the
        pretrained GPT-2 models.
        """
        cache_dir = "../resources/hugging_face/gpt2/"
        model = TFGPT2Model.from_pretrained('gpt2', cache_dir=cache_dir, return_dict=True)

        output = model(self.placeholders['tokens'], training=is_train)

        seq_token_embeddings = output.last_hidden_state

        seq_token_masks = self.placeholders['tokens_mask']
        seq_token_lengths = tf.reduce_sum(input_tensor=seq_token_masks, axis=1)  # B
        return pool_sequence_embedding("weighted_mean",
                                       sequence_token_embeddings=seq_token_embeddings,
                                       sequence_lengths=seq_token_lengths,
                                       sequence_token_masks=seq_token_masks)
