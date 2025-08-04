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
            self.answer_summary_reward_func,
            self.length_penalty_reward_func,
        ]
        self.reward_weights = [
            0.25,  # Structural format
            0.35,  # Tool usage (most important)
            0.25,  # Tool formatting
            0.10,  # Answer summary
            0.05,  # Length penalty (negative)
        ]

    def structural_format_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Reward function that evaluates structural format compliance for the simplified task.
        Checks for: think tags, tool tags, answer tags, and proper structure.
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
                
                # Check for tool tags (0.25 points)
                tool_match = re.search(r'<tool>(.*?)</tool>', content, re.DOTALL)
                if tool_match:
                    total_reward += 0.25
                max_possible += 0.25
                
                # Check for answer tags (0.25 points)
                answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if answer_match:
                    total_reward += 0.25
                max_possible += 0.25
                
                # Check for all three tags present (0.25 points)
                if think_match and tool_match and answer_match:
                    total_reward += 0.25
                max_possible += 0.25
        
        return total_reward / max_possible if max_possible > 0 else 0.0

    def tool_usage_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Reward function that checks if exactly one search_product tool call was made.
        This is the most important aspect of the task.
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
                        if isinstance(command, dict) and command.get("name") == "search_product":
                            tool_calls += 1
                    except json.JSONDecodeError:
                        pass
        
        # Reward for exactly one tool call (not multiple, not zero)
        if tool_calls == 1:
            return 1.0
        elif tool_calls == 0:
            return 0.0
        else:
            return 0.5  # Partial reward for multiple calls

    def tool_formatting_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Reward function that verifies search_product is called with correct JSON formatting and arguments.
        Specifically checks for the correct tool name and query argument.
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
                                
                                # Check if it's search_product tool
                                if tool_name == "search_product":
                                    # Check if args is a dictionary with query field
                                    if isinstance(args, dict) and "query" in args:
                                        # Check if query is a string (natural language)
                                        if isinstance(args["query"], str) and len(args["query"].strip()) > 0:
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

    def answer_summary_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Reward function that checks if the model provided a summary in the answer tags.
        """
        for msg in completion:
            if msg['role'] == 'assistant':
                content = msg['content']
                answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # Check if the answer contains a meaningful summary
                    if len(answer_content) > 10 and len(answer_content) < 500:
                        # Check for summary-like content (mentions reasoning, tool call, etc.)
                        summary_keywords = ['reason', 'think', 'search', 'tool', 'call', 'found', 'retrieve', 'product']
                        if any(keyword in answer_content.lower() for keyword in summary_keywords):
                            return 1.0
                        else:
                            return 0.5  # Partial reward for having answer tags but not great content
                    else:
                        return 0.0  # Too short or too long
        
        return 0.0  # No answer tags found