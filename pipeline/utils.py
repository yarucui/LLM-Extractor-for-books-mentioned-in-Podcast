import json
import re
from typing import Any, Optional

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
