import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class VQAHead(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, aligned_tokens, text_query):
        # aligned_tokens: (B, num_queries, 32)
        # For demo, flatten and use as prompt embedding (real use: fuse with text)
        batch_size = aligned_tokens.size(0)
        answers = []
        for i in range(batch_size):
            prompt = text_query[i]
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output = self.generator.generate(input_ids, max_length=20)
            answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answers.append(answer)
        return answers
