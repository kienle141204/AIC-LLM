import torch 
from torch import nn
from modelscope.models import Model
from swift import LoRAConfig, Swift
from modelscope import AutoTokenizer 

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, input_ids, attention_mask=None, labels=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def gettokenizer(self):
        return NotImplementedError("Subclasses should implement this method.")

    def getembedding(self, input_ids):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_month_embedding(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_week_embedding(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
class GPT2(BaseModel):
    def __init__(self, lora, ln_grad, layers=None):
        super(GPT2, self).__init__()

        self.llm = Model.from_pretrained('AI-ModelScope/gpt2', trust_remote_code=True)
        self.dim = 768

        if not layers is None:
            self.llm.transformer.h = self.llm.transformer.h[:layers]
        
        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)
        
        if lora:
            lora_config = LoRAConfig(
                r=16,
                lora_alpha=32,
                target_modules=['q_attn','c_attn'],
                lora_dropout=0.,
            )
            self.llm = Swift.prepare_model(self.llm, lora_config,trust_remote_code=True).model

        if ln_grad:
            for name, param in self.llm.named_parameters():
                if 'ln_' in name or 'wpe' in name:
                    param.requires_grad_(True)
        
        self.tokenizer = AutoTokenizer.from_pretrained('AI-ModelScope/gpt2', trust_remote_code=True)
    
    def forward(self, input: torch.FloatTensor, attention_mask=None):
        output = self.llm(inputs_embeds=input, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]

        return output
    
    def gettokenizer(self):
        return self.tokenizer

    def getembedding(self, input_ids):
        return self.llm.transformer.wte(input_ids)


