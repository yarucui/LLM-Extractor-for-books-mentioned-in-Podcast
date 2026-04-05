import json
import re
from typing import Any, Optional

class TokenTracker:
    def __init__(self, default_model: str):
        self.default_model = default_model
        self.usage_by_model = {}
        
        # OpenRouter pricing for google/gemini-3-flash-preview (approximate)
        # Prices are per 1M tokens
        self.prices = {
            "google/gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
            "google/gemini-3.1-pro-preview": {"input": 1.25, "output": 5.00},
            "default": {"input": 0.1, "output": 0.4} # Fallback
        }

    def add_usage(self, prompt_tokens: int, completion_tokens: int, model_name: Optional[str] = None):
        model = model_name or self.default_model
        if model not in self.usage_by_model:
            self.usage_by_model[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0
            }
        
        self.usage_by_model[model]["prompt_tokens"] += prompt_tokens
        self.usage_by_model[model]["completion_tokens"] += completion_tokens
        self.usage_by_model[model]["calls"] += 1

    def get_report(self) -> str:
        report = f"\n{'='*40}\n"
        report += f"📊 MULTI-MODEL USAGE REPORT\n"
        report += f"{'='*40}\n"
        
        total_cost_all = 0
        total_calls_all = 0
        
        for model, usage in self.usage_by_model.items():
            # Determine pricing
            price_key = "default"
            for key in self.prices:
                if key in model:
                    price_key = key
                    break
            
            p = self.prices[price_key]
            input_cost = (usage["prompt_tokens"] / 1_000_000) * p["input"]
            output_cost = (usage["completion_tokens"] / 1_000_000) * p["output"]
            total_cost = input_cost + output_cost
            
            total_cost_all += total_cost
            total_calls_all += usage["calls"]
            
            report += f"Model: {model}\n"
            report += f"  Calls:             {usage['calls']}\n"
            report += f"  Prompt Tokens:     {usage['prompt_tokens']:,}\n"
            report += f"  Completion Tokens: {usage['completion_tokens']:,}\n"
            report += f"  Estimated Cost:    ${total_cost:.4f}\n"
            report += f"{'-'*40}\n"
            
        report += f"TOTAL CALLS:         {total_calls_all}\n"
        report += f"TOTAL ESTIMATED COST: ${total_cost_all:.4f}\n"
        report += f"{'='*40}\n"
        return report

def count_words(text: str) -> int:
    """
    Counts the number of words in a string.
    """
    if not text:
        return 0
    return len(text.split())

def safe_json_loads(text: str) -> Optional[Any]:
    """
    Robustly parse JSON from a string, handling markdown blocks and common errors.
    """
    if not text:
        return None
    
    text = text.strip()
    
    # 1. Remove markdown code blocks if present
    if text.startswith("```"):
        # Try to find the content between ```json and ``` or just ``` and ```
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        else:
            # Fallback: just strip the markers
            text = re.sub(r'^```(?:json)?|```$', '', text).strip()
    
    # 2. Try direct parse
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    
    # 3. Try to extract the first { ... } or [ ... ] block
    # This helps if the model added conversational text around the JSON
    json_block_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if json_block_match:
        candidate = json_block_match.group(1).strip()
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            # 4. Last ditch effort: fix common JSON syntax errors
            # - Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',\s*([\}\]])', r'\1', candidate)
            # - Try to fix single quotes used as property names (risky but common)
            # fixed = re.sub(r"\'(\w+)\'\s*:", r'"\1":', fixed) 
            try:
                return json.loads(fixed, strict=False)
            except:
                return None
    
    return None
