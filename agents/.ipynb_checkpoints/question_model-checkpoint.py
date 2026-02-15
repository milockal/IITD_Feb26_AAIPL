import torch
from transformers import pipeline
import json
import time
import re
import random

class QAgent:
    def __init__(self):
        self.model_path = "/workspace/AAIPL/hf_models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
        print("Loading Q-Agent model...")
        self.pipe = pipeline("text-generation", model=self.model_path, device=0, torch_dtype=torch.float16)
        
        # Expose tokenizer for the Judge
        self.tokenizer = self.pipe.tokenizer
        
        print("Q-Agent Loaded!")

    def generate_response(self, prompt, system_prompt, **kwargs):
        start_time = time.time()
        
        # Ensure input is a list
        topics = prompt if isinstance(prompt, list) else [prompt]
        
        batch_prompts = []
        for topic in topics:
            # --- TOURNAMENT OPTIMIZATION ---
            # 1. Attack: Use "Negative Logic" (NOT, EXCEPT) to confuse opponent AIs.
            # 2. Compliance: Explicitly ask for List format ["A)..."] and <100 words explanation.
            fake_thought = f"""<|im_start|>user
Create a high-difficulty {topic} puzzle. 
Use negative logic (e.g., 'which is NOT true') or complex conditions.
Output valid JSON. 
Constraints:
- 'choices' must be a list ["A) ...", "B) ..."].
- 'explanation' must be under 100 words.
<|im_end|>
<|im_start|>assistant
<think>
I need to generate a deceptive {topic} question. I will ensure the choices are a list and the explanation is short.
</think>
```json
{{"""
            batch_prompts.append(fake_thought)

        # Generate
        outputs = self.pipe(batch_prompts, max_new_tokens=600, temperature=0.7, return_full_text=False)
        
        final_responses = []
        token_counts = []
        
        for i, out in enumerate(outputs):
            raw_text = out[0]['generated_text']
            
            # Reconstruct JSON
            full_json_str = "{" + raw_text
            
            # Robust Closer
            end = full_json_str.rfind("}")
            
            clean_json = ""
            if end != -1:
                clean_json = full_json_str[:end+1]
            else:
                clean_json = full_json_str + '"}'
            
            if len(clean_json) < 10:
                clean_json = "{}"

            final_responses.append(clean_json)
            token_counts.append(len(clean_json.split()))

        end_time = time.time()
        generation_time = end_time - start_time
        
        return final_responses, sum(token_counts), generation_time