import torch
from transformers import pipeline
import json
import time
import re

class AAgent:
    def __init__(self):
        self.model_path = "/workspace/AAIPL/hf_models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
        print("Loading A-Agent model...")
        self.pipe = pipeline("text-generation", model=self.model_path, device=0, torch_dtype=torch.float16)
        self.tokenizer = self.pipe.tokenizer
        
        # --- WARMUP ---
        print("Warming up GPU...")
        self._warmup()
        print("A-Agent Loaded & Warmed Up!")

    def _warmup(self):
        dummy = [{"role": "user", "content": "1+1="}]
        self.pipe(dummy, max_new_tokens=10, min_new_tokens=1)

    def generate_response(self, prompt, system_prompt=None, **kwargs):
        start_time = time.time()
        
        questions = prompt if isinstance(prompt, list) else [prompt]
        
        batch_prompts = []
        for q_text in questions:
            # --- THE STRATEGY ---
            # Capacity: 450 tokens (High)
            # Control: "1 short sentence" (Strict)
            # We are betting the model obeys this instruction and stops early.
            fake_thought = f"""<|im_start|>user
You are a logic expert. Solve this puzzle. 
Output valid JSON with keys: "answer" (letter only) and "reasoning" (MAXIMUM 1 short sentence).
Puzzle:
{q_text}<|im_end|>
<|im_start|>assistant
<think>
I will analyze the constraints. I will pick the best option (A, B, C, or D) and provide a concise reasoning.
</think>
```json
{{"""
            batch_prompts.append(fake_thought)

        # --- SETTING: 450 TOKENS ---
        # If the model ignores the prompt and writes a paragraph, YOU WILL TIMEOUT (>10s).
        # If the model obeys and writes 1 sentence, you will be fast (<3s).
        outputs = self.pipe(
            batch_prompts, 
            max_new_tokens=450, 
            temperature=0.1,  # Low temp to force it to be boring and concise
            repetition_penalty=1.2,
            return_full_text=False
        )
        
        final_responses = []
        token_counts = []
        
        for out in outputs:
            raw_text = out[0]['generated_text']
            
            full_json_str = "{" + raw_text
            
            end = full_json_str.rfind("}")
            if end != -1:
                clean_json = full_json_str[:end+1]
            else:
                if full_json_str.count('"') % 2 != 0:
                    clean_json = full_json_str + '"}' 
                else:
                    clean_json = full_json_str + "}"

            if len(clean_json) < 10:
                clean_json = "{}"

            final_responses.append(clean_json)
            token_counts.append(len(clean_json.split()))
            
        end_time = time.time()
        generation_time = end_time - start_time
        
        return final_responses, sum(token_counts), generation_time