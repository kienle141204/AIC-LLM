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

        try:
            print("Loading local model")
            local_model_path = '/home/user03/VARDiff-test/newtest1/AIC-LLM/gpt2_modelscope/AI-ModelScope/gpt2'
            self.llm = Model.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True)
        except:
            print("Loading remote model")
            self.llm = Model.from_pretrained('AI-ModelScope/gpt2', trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained('AI-ModelScope/gpt2', trust_remote_code=True)
        
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
    
    def forward(self, input: torch.FloatTensor, attention_mask=None):
        output = self.llm(inputs_embeds=input, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]

        return output
    
    def gettokenizer(self):
        return self.tokenizer

    def getembedding(self, input_ids):
        return self.llm.transformer.wte(input_ids)

class LLaMA7B(BaseModel):
    def __init__(self, lora, ln_grad, layers=None): 
        super(LLaMA7B, self).__init__()
        from transformers import AutoModel, AutoTokenizer as TransformerTokenizer

        print("Loading LLaMA-7B model")
        model_path = "meta-llama/Llama-2-7b-hf" 
        try:
             self.llm = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
             self.tokenizer = TransformerTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
             print(f"Could not load {model_path}, trying open_llama_7b")
             self.llm = AutoModel.from_pretrained("openlm-research/open_llama_7b", trust_remote_code=True, device_map='auto')
             self.tokenizer = TransformerTokenizer.from_pretrained("openlm-research/open_llama_7b", trust_remote_code=True)
        
        self.dim = 4096

        if not layers is None:
            if hasattr(self.llm, 'layers'):
                 self.llm.layers = self.llm.layers[:layers]
        
        # Freeze
        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)
            
    def forward(self, input: torch.FloatTensor, attention_mask=None):
        output = self.llm(inputs_embeds=input, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        return output
    
    def gettokenizer(self):
        return self.tokenizer

    def getembedding(self, input_ids):
        return self.llm.get_input_embeddings()(input_ids)


