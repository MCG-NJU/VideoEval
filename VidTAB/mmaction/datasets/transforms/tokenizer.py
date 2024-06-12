from typing import Sequence
import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS

import clip

from transformers import BertTokenizer

@TRANSFORMS.register_module()
class CLIPTokenize(BaseTransform):
    """Using CLIP's tokenizer to tokenize txt/
    """

    def __init__(self, text_list=(,)):
        self.text_list = text_list


    def transform(self, results):
        """Perform the text tokenization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        for k in results.keys():
            if k in self.text_list:
                results[k.replace('_txt', '_CLIP_token')] = clip.tokenize(results[k])
        
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(text_list={self.text_list})'
        return repr_str

@TRANSFORMS.register_module()
class BertTokenize(BaseTransform):
    """Using Bert's tokenizer to tokenize txt/
    """

    def __init__(self, text_list=(,), tokenizer_type='bert-base-uncased', add_special_tokens=True, padding="longest", max_length=20, truncation=True):
        self.text_list = text_list
        self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_type)
        self.tokenizer_type = tokenizer_type
        self.add_special_tokens=add_special_tokens
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation

    def transform(self, results):
        """Perform the text tokenization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        for k in results.keys():
            if k in self.text_list:
                results[k.replace('_txt', '_Bert_token')] = torch.tensor(
                        self.bert_tokenizer.encode(
                            results[k],
                            add_special_tokens=self.add_special_tokens,
                            padding=self.padding,
                            max_length=self.max_length,
                            truncation=self.truncation,
                        ),
                        dtype=torch.long,
                    )
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(text_list={self.text_list}'
        repr_str += f', tokenizer_type={self.tokenizer_type}'
        repr_str += f', add_special_tokens={self.add_special_tokens}'
        repr_str += f', padding={self.padding}'
        repr_str += f', max_length={self.max_length}' 
        repr_str += f', truncation={self.truncation}'
        repr_str += ')'
        return repr_str