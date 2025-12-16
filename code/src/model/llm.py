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


class Qwen2_5(BaseModel):
    """
    Qwen 2.5 model implementation with LoRA support.
    Supports multiple sizes: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
    """
    def __init__(self, lora, ln_grad, layers=None, model_size='7B'): 
        super(Qwen2_5, self).__init__()
        
        # Model size mapping
        model_mapping = {
            '0.5B': 'Qwen/Qwen2.5-0.5B',
            '1.5B': 'Qwen/Qwen2.5-1.5B',
            '3B': 'Qwen/Qwen2.5-3B',
            '7B': 'Qwen/Qwen2.5-7B',
            '14B': 'Qwen/Qwen2.5-14B',
            '32B': 'Qwen/Qwen2.5-32B',
            '72B': 'Qwen/Qwen2.5-72B',
        }
        
        # Dimension mapping for different model sizes
        dim_mapping = {
            '0.5B': 896,
            '1.5B': 1536,
            '3B': 2048,
            '7B': 3584,
            '14B': 5120,
            '32B': 5120,
            '72B': 8192,
        }
        
        model_name = model_mapping.get(model_size, model_mapping['7B'])
        self.dim = dim_mapping.get(model_size, dim_mapping['7B'])
        
        try:
            print(f"Loading local Qwen 2.5 {model_size} model")
            local_model_path = f'/home/user03/VARDiff-test/newtest1/AIC-LLM/qwen2.5_{model_size.lower()}'
            self.llm = Model.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True)
        except:
            print(f"Loading remote Qwen 2.5 {model_size} model from {model_name}")
            self.llm = Model.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Limit number of layers if specified
        if layers is not None:
            self.llm.model.layers = self.llm.model.layers[:layers]
        
        # Freeze all parameters by default
        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)
        
        # Apply LoRA if enabled
        if lora:
            lora_config = LoRAConfig(
                r=16,
                lora_alpha=32,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                lora_dropout=0.05,
            )
            self.llm = Swift.prepare_model(self.llm, lora_config, trust_remote_code=True).model
        
        # Enable gradient for layer normalization if specified
        if ln_grad:
            for name, param in self.llm.named_parameters():
                if 'norm' in name.lower() or 'ln' in name.lower():
                    param.requires_grad_(True)
    
    def forward(self, input: torch.FloatTensor, attention_mask=None):
        output = self.llm(inputs_embeds=input, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        return output
    
    def gettokenizer(self):
        return self.tokenizer
    
    def getembedding(self, input_ids):
        return self.llm.model.embed_tokens(input_ids)


