"""
Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""

from io import open
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm

from model.vision_encoders.vision_builder import builder as vision_builder
from model.txt_encoders.txt_builder import builder as txt_builder
from model.txt_encoders.multi_builder import builder as multi_builder
from utils.logger import LOGGER
from utils.misc import OPTConfig


class VLABPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, OPTConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `OPTConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    @classmethod
    def from_pretrained(cls, config, state_dict, *inputs, **kwargs):
        """
        Instantiate a OPTPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific OPT class
        """
        # Load config
        config = vars(config)
        config = OPTConfig(config)
        LOGGER.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        if len(state_dict.keys()) == 0:
            return model
        missing_keys,unexpected_keys = model.load_state_dict(state_dict,strict=False)
        if state_dict != {}:
            # print(state_dict)
            LOGGER.info(f"Builded Model Unexpected keys {unexpected_keys}")
            LOGGER.info(f"Builded Model missing_keys  {missing_keys}")
        return model




class VLABModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        vision_cfg=config.vision_encoder
        txt_cfg=config.txt_encoder
        multi_cfg=config.multi_encoder

        vision_cfg['resolution'] = config.video_cfg['resolution']
        builded_models = vision_builder(vision_cfg) ## check
        if isinstance(builded_models, tuple):
            self.vision_text_model=builded_models[0]
            self.vision_encoder = self.vision_text_model.encode_image
            self.txt_encoder = self.vision_text_model.encode_text

        else:
            self.vision_encoder=builded_models
        
        if txt_cfg.get("text_type", None) is not None:
            self.txt_encoder = txt_builder(txt_cfg)
        
        self.multimodal_encoder, self.multimodal_head=multi_builder(multi_cfg)


    def forward_txt_encoder(self, txt_tokens):

        txt_output = self.txt_encoder(txt_tokens)        
        return txt_output


    def forward_video_encoder(self, video_pixels):

        video_output = self.vision_encoder(video_pixels)
        return video_output
    
        
    def forward_multimodal_encoder(self, *args,**kwargs):
 
        return self.multimodal_encoder(*args,**kwargs)




