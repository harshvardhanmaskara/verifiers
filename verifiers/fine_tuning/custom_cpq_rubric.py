import json
import re
from typing import List, Dict, Callable

from verifiers import Parser, XMLParser, Rubric

class CustomCPQRubric(Rubric):
    def __init__(self,
                 parser: Parser = XMLParser(fields=["think", ("tool", "answer")]),
                 env_parser: Parser = XMLParser(fields=["result"]),
                 tools: List[Callable] = []):
        super().__init__(parser=parser)
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {
            tool.__name__ if hasattr(tool, '__name__') else str(tool): tool
            for tool in tools
        }
        
        # Define reward functions and weights
        self.reward_funcs = [
            self.structural_format_reward_func,
            self.tool_usage_reward_func,
            self.tool_formatting_reward_func,
            self.length_penalty_reward_func,
        ]
        self.reward_weights = [
            0.4,  # Structural format
            0.3,  # Tool usage
            0.2,  # Tool formatting
            0.1,  # Length penalty (negative)
        ]

    def structural_format_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Reward function that evaluates structural format compliance.
        Assigns partial rewards for each aspect of the format.
        """
        if not completion:
            return 0.0
        
        total_reward = 0.0
        max_possible = 0.0
        
        for msg in completion:
            if msg['role'] == 'assistant':
                content = msg['content']
                
                # Check for think tags (0.25 points)
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    total_reward += 0.25
                max_possible += 0.25
                
                # Check for either tool or answer tags (0.25 points)
                tool_match = re.search(r'<tool>(.*?)</tool>', content, re.DOTALL)
                answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if tool_match or answer_match:
                    total_reward += 0.25
                max_possible += 0.25
                
                # Check for proper XML structure (0.25 points)
                if '<think>' in content and '</think>' in content:
                    if '<tool>' in content and '</tool>' in content:
                        # Has both think and tool
                        total_reward += 0.25
                    elif '<answer>' in content and '</answer>' in content:
                        # Has both think and answer
                        total_reward += 0.25
                max_possible += 0.25
                
                # Check for no mixed tags (0.25 points)
                if not (tool_match and answer_match):
                    total_reward += 0.25
                max_possible += 0.25
        
        return total_reward / max_possible if max_possible > 0 else 0.0

    def tool_usage_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Reward function that checks if one of the two tools has been called.
        """
        tool_calls = 0
        
        for msg in completion:
            if msg['role'] == 'assistant':
                content = msg['content']
                tool_match = re.search(r'<tool>(.*?)</tool>', content, re.DOTALL)
                if tool_match:
                    tool_content = tool_match.group(1).strip()
                    try:
                        command = json.loads(tool_content)
                        if isinstance(command, dict) and command.get("name") in self.tools:
                            tool_calls += 1
                    except json.JSONDecodeError:
                        pass
        
        # Reward if at least one tool was called
        return 1.0 if tool_calls > 0 else 0.0

    def tool_formatting_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Reward function that verifies tools are called with correct JSON formatting.
        """
        correct_formats = 0
        total_attempts = 0
        
        for msg in completion:
            if msg['role'] == 'assistant':
                content = msg['content']
                tool_match = re.search(r'<tool>(.*?)</tool>', content, re.DOTALL)
                if tool_match:
                    tool_content = tool_match.group(1).strip()
                    total_attempts += 1
                    
                    try:
                        command = json.loads(tool_content)
                        if isinstance(command, dict):
                            # Check for required fields
                            if "name" in command and "args" in command:
                                tool_name = command["name"]
                                args = command["args"]
                                
                                # Check if it's one of our tools
                                if tool_name in self.tools:
                                    # Check if args is a dictionary
                                    if isinstance(args, dict):
                                        correct_formats += 1
                    except json.JSONDecodeError:
                        pass
        
        return correct_formats / total_attempts if total_attempts > 0 else 0.0

    def length_penalty_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Penalty function to avoid reward hacking through excessive length or repeated characters.
        Returns negative values for penalties.
        """
        total_length = 0
        repeated_char_penalty = 0
        
        for msg in completion:
            if msg['role'] == 'assistant':
                content = msg['content']
                total_length += len(content)
                
                # Check for repeated characters (reward hacking)
                for char in content:
                    if char * 5 in content:  # 5 or more repeated characters
                        repeated_char_penalty += 0.1
                        break
        
        # Length penalty: penalize if total length > 2000 characters
        length_penalty = 0.0
        if total_length > 2000:
            length_penalty = (total_length - 2000) / 1000  # 0.1 penalty per 100 chars over limit
        
        # Combine penalties (negative values)
        total_penalty = -(length_penalty + repeated_char_penalty)
        
        # Clamp to reasonable range
        return max(-1.0, total_penalty)