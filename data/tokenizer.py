from utils.logger import LOGGER

def build_tokenizer(tokenizer_dict):

    res = {}
    LOGGER.info(tokenizer_dict)

    for k, tokenizer_type in tokenizer_dict.items():
        if tokenizer_type == "bert_tokenizer":
            from model.txt_encoders.bert_tokenizer import BertTokenizer
            tokenizer = BertTokenizer("./output/pretrained_weights/bert/bert-base-uncased-vocab.txt")
            tokenizer_func = MetaFunc(tokenizer)
            bos_token = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
            eos_token = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]

        if tokenizer_type == "clip_tokenizer":
            from model.txt_encoders.clip_tokenizer import SimpleTokenizer
            tokenizer = SimpleTokenizer()
            tokenizer_func = tokenizer.encode
            bos_token = tokenizer.encoder["<|startoftext|>"]
            eos_token = tokenizer.encoder["<|endoftext|>"]


        if tokenizer_type == "gpt2_tokenizer":
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
            tokenizer_func = MetaFunc(tokenizer)
            bos_token = tokenizer.convert_tokens_to_ids(['</s>'])[0]
            eos_token = tokenizer.convert_tokens_to_ids(['</s>'])[0]

        if tokenizer_type == "deberta_tokenizer":
            from transformers import DebertaV2Tokenizer
            tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
            tokenizer_func = MetaFunc(tokenizer)
            bos_token = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
            eos_token = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

        res[k]={"tokenizer": tokenizer_func,
            "bos":bos_token,
            "eos":eos_token}

    return res


class MetaFunc(object):

    def __init__(self, tokenizer) -> None:
        super().__init__()
        
        self.tokenizer = tokenizer
    def __call__(self, text):
        
        tokens = self.tokenizer.tokenize(text)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens
