import os
import json
import openai
from typing import Dict, Any, Optional, List

# 关键词列表，用于cut_off函数
keyword_list = ["citizen", "resident", "audience", "crowd", 
                "citizens", "residents", "audiences", 
                "communities", "comment", "comments", 
               "hearing", "hearings"]

def cut_off(content: str, args: Any) -> str:
    """
    Cut off content based on length and keywords
    
    Args:
        content: input content
        args: arguments object containing cut_off_th
    
    Returns:
        str: processed content
    """
    total_list = content.split("\n")
    temp_list = []
    for i in total_list:
        if len(i.split()) >= args.cut_off_th:
            sentences = i.split(".")
            chunks = []
            chunk_len = 0
            chunk = ""
            for j in sentences:
                chunk += j + " "
                chunk_len += len(j.split())
                if chunk_len > args.cut_off_th:
                    chunks.append(chunk)
                    chunk_len = 0
                    chunk = ""
            
            out_str = ""
            for c in chunks:
                status = 0
                for kw in keyword_list:
                    if kw in c:
                        status = 1
                if status:
                    out_str += c + " "
            if out_str:
                temp_list.append(out_str)
        else:
            temp_list.append(i) 
    return "\n".join(temp_list)

def ask_llm(initial_prompt: str, content: str, model: str = "gpt-4", temperature: float = 0) -> Dict[str, Any]:
    """
    Ask GPT model for processing
    
    Args:
        initial_prompt: initial prompt
        content: content to process
        model: model name
        temperature: temperature parameter
    
    Returns:
        Dict[str, Any]: processing result
    """
    print(f"Using model: {model}")
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    msg = initial_prompt + content
    messages = [
        {"role": "user", "content": msg}
    ]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    try:
        return eval(response.choices[0].message.content)
    except:
        return {}

def ask_llm_multi(initial_prompt: str, content: str, the_json: Dict[str, Any], model: str = "gpt-4", temperature: float = 0) -> Dict[str, Any]:
    """
    Ask GPT model for multi-turn processing
    
    Args:
        initial_prompt: initial prompt
        content: content to process
        the_json: existing JSON result
        model: model name
        temperature: temperature parameter
    
    Returns:
        Dict[str, Any]: processing result
    """
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    msg = "The JSON file: \n" + json.dumps(the_json) + "\n Transcript segment: \n" + content
    
    messages = [
        {"role": "system", "content": initial_prompt},
        {"role": "user", "content": msg}
    ]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    try:
        return eval(response.choices[0].message.content)
    except:
        return {}

def ask_llm_new(initial_prompt: str, content: str, model: str = "gpt-4", temperature: float = 0) -> Dict[str, Any]:
    """
    Ask GPT model with new format
    
    Args:
        initial_prompt: initial prompt
        content: content to process
        model: model name
        temperature: temperature parameter
    
    Returns:
        Dict[str, Any]: processing result
    """
    print(f"Using model: {model}")
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    msg = "\n Transcript segment: \n" + content
    
    messages = [
        {"role": "system", "content": initial_prompt},
        {"role": "user", "content": msg}
    ]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    content = response.choices[0].message.content
    print(content[7:-4].strip())
    print("```json" in content)
    
    if "```json" in content:
        c = content[7:-4].strip()
        try:
            return eval(c)
        except:
            return {}
    return {} 