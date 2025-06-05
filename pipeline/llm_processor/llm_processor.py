from typing import Dict, Any, Optional
from .llm_utils import ask_llm, ask_llm_multi, cut_off
from .config import PROMPTS, MODEL_CONFIG, KEYWORDS

def process_with_llm(content: str, model: str = None, temperature: float = None, args: Any = None) -> Dict[str, Any]:
    """
    Process text with LLM
    
    Args:
        content: text to process
        model: model name (optional, defaults to config)
        temperature: temperature parameter (optional, defaults to config)
        args: arguments object containing mode and cut_off_th
    
    Returns:
        Dict[str, Any]: processing result
    """
    # Use default values from config if not provided
    model = model or MODEL_CONFIG["default_model"]
    temperature = temperature or MODEL_CONFIG["temperature"]
    
    # Get prompt template based on mode
    if args is not None and hasattr(args, 'mode'):
        if args.mode == "find_public_trigger":
            prompt_template = PROMPTS.get("find_public_trigger")
        else:
            prompt_template = PROMPTS.get("find_public_trigger_general")  # 或者你定义的其他 key
            if not prompt_template:
                raise ValueError(f"Unknown mode: {args.mode}")
    else:
        raise ValueError("Mode must be specified in args")

    
    # Apply cut_off if args is provided and mode is find_public_trigger
    if args.mode == 'find_public_trigger':
        content = cut_off(content, args)
        return ask_llm(prompt_template, content, model, temperature)
    
    # Process in chunks if content is too long (for find_public_trigger_general)
    else:
        the_json = {}
        total_json = {}
        length = 0
        chunk = ""
        
        for line in content.split('\n'):
            chunk += line + '\n'
            length += len(line.split())
            
            if length > MODEL_CONFIG["chunk_size"]:
                # Try different temperatures
                new_json = ask_llm_multi(prompt_template, chunk, the_json, model, temperature)
                if not new_json:
                    for temp in MODEL_CONFIG["fallback_temperatures"]:
                        new_json = ask_llm_multi(prompt_template, chunk, the_json, model, temp)
                        if new_json:
                            break
                
                if new_json:
                    the_json = new_json
                    for key in new_json:
                        if key not in total_json:
                            total_json[key] = new_json[key]
                
                chunk = ""
                length = 0
        
        # Process remaining content
        if chunk:
            new_json = ask_llm_multi(prompt_template, chunk, the_json, model, temperature)
            if not new_json:
                for temp in MODEL_CONFIG["fallback_temperatures"]:
                    new_json = ask_llm_multi(prompt_template, chunk, the_json, model, temp)
                    if new_json:
                        break
            
            if new_json:
                the_json = new_json
                for key in new_json:
                    if key not in total_json:
                        total_json[key] = new_json[key]
        
        return total_json