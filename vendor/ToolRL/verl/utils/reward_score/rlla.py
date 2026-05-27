# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import os
from collections import Counter


def match_score(list1, list2):
    """Compute multiset F1 considering element frequency, ignoring order."""
    if list1 == list2:
        return 1.0
    if os.getenv("REFINEDREWARD", 0) == "1":
        print("REFINEDREWARD is set to 1, so strict match is used")
        if list1 != list2:
            return 0.0
    
    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)  # Frequency count for list1
    count2 = Counter(list2)  # Frequency count for list2

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())

    return 2 * intersection / (len(list1) + len(list2)) if len(list1) + len(list2) > 0 else 0.0


def token_f1(value1, value2):
    tokens1 = re.findall(r"\w+", str(value1).lower())
    tokens2 = re.findall(r"\w+", str(value2).lower())
    if tokens1 == tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    count1 = Counter(tokens1)
    count2 = Counter(tokens2)
    overlap = sum(min(count1[token], count2[token]) for token in count1.keys() & count2.keys())
    return 2 * overlap / (len(tokens1) + len(tokens2)) if overlap else 0.0


def extract_xml_block(text, tag):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_multi_attempts(text, count):
    attempts = []
    for i in range(1, count + 1):
        attempt = extract_xml_block(text, f"response_{i}")
        attempts.append(attempt)
    if any(attempts):
        return attempts
    return [text]


def extract_assistant_text(solution_str):
    if "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        return solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    if "<|im_start|>assistant" in solution_str:
        return solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    return solution_str.strip()


def parse_tool_calls(text):
    tool_call = extract_xml_block(text, "tool_call")
    if not tool_call:
        return []
    return [json.loads(tool) for tool in tool_call.split("\n") if tool.strip()]


# custoimzed reward functions: format
def customize_format_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step >= 30:
            max_possible_reward = max_possible_reward / 2
            min_possible_reward = min_possible_reward / 2
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    # schedule reward
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = 2 - (2 - max_possible_reward) * step / 150
        min_possible_reward = -2 + (2 + min_possible_reward) * step / 150
        if max_possible_reward < 1.0:
            max_possible_reward = 1.0
        if min_possible_reward > -1.0:
            min_possible_reward = -1.0
    
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    
    if os.getenv("REWARD_DEBUG", "0") == "1":
        print("\n======= Answer ======= ")
        print(answer[0])
        print("\n======= Responses ======= ")
        for idx, response in enumerate(responses):
            print(f"*** Response {idx+1}***\n{response}")

    for response, ans in zip(responses, answer):
        reward = min_possible_reward
        if "<response>" in ans and "<tool_call>" not in ans:
            pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<response>") == 1 and response.count("</response>") == 1:
                reward = max_possible_reward
        elif "<response>" not in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$" 
            if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1:
                reward = max_possible_reward
        elif "<response>" in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1 and response.count("<response>") == 1 and response.count("</response>") == 1:
                reward = max_possible_reward
        else:
            pattern = r"^<think>.*?</think>$"
            if re.search(pattern, response, re.DOTALL):
                reward = max_possible_reward
        
        rewards.append(reward)
        
    if os.getenv("REWARD_DEBUG", "0") == "1":
        print("\n======= Reward for <format> =======")
        print("Reward function for <format> is called ...")
        print(rewards)
    return rewards


# customized reward functions: length
def customize_length_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    # schedule length
    if os.getenv("SCHEDULELENGTH", 0) == "1":
        print("SCHEDULELENGTH is set to 1, so schedule max reward for length is used")
        max_reward_len = (640 - 384) * step / 105 + 384
    else:
        max_reward_len = 512
    
    """Reward function that gives higher scores to longer completions."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        if "<think>" not in response or "</think>" not in response:
            rewards.append(min_possible_reward)
            continue
        think_responses = response.split("<think>")[-1].split("</think>")[0].strip()
        reward = round(len(think_responses.split()) / max_reward_len, 2)
        if reward > 1.0:
            reward = 1.0
        
        final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
        rewards.append(final_reward)
    
    print("\n======= Reward for <length> =======")
    print("Reward function for <length> is called ...")
    print(rewards)
    return rewards
                

def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    if gt_tools == pd_tools:
        print("Max possible score:", "Exact Match!")
        print("Score:", max_possible_reward)
        return max_possible_reward
    
    if os.getenv("COARSEREWARD", 0) == "1":
        print("COARSEREWARD is set to 1, so coarse reward is used")
        if gt_tools != pd_tools:
            return min_possible_reward

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))
    
    local_max_possible = 1.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]
        
        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            print("INTERMEDIATEREWARD is set to 1, so local max possible is changed")
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0 + len(gt_params)
        
        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue
            
            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if gt_tool == pd_tool:
                    best_match = pd_tool
                    best_match_index = i
                    best_match_score = 1.0
                    break
                else:
                    continue
            
            pd_params = pd_tool["parameters"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            
            # Calculate correctness score for parameter values
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            total_score = param_score + correctness_score
            
            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    print()
    print("Max possible score:", local_max_possible)
    print("Score:", score)
    
    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


def compute_tool_call_vector(gt_tools, pd_tools):
    """Return ToolRL VPO vector components in [0, 1].

    Components are tool-name multiset F1, argument-key set F1, and argument-value token F1.
    The scalar reward path above is left intact for baseline GRPO compatibility.
    """
    if gt_tools == pd_tools:
        return 1.0, 1.0, 1.0

    if os.getenv("COARSEREWARD", 0) == "1" and gt_tools != pd_tools:
        return 0.0, 0.0, 0.0

    if not gt_tools or not pd_tools:
        return 0.0, 0.0, 0.0

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    name_score = match_score(gt_names, pd_names)

    key_scores = []
    value_scores = []
    used_pd_indices = set()
    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]
        best_match_index = -1
        best_key_score = 0.0
        best_value_score = 0.0
        best_total = -1.0

        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue
            pd_params = pd_tool["parameters"]
            key_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            value_score = sum(token_f1(value, pd_params[key]) for key, value in gt_params.items() if key in pd_params)
            value_score = value_score / max(1, len(gt_params))
            total = key_score + value_score
            if total > best_total:
                best_total = total
                best_match_index = i
                best_key_score = key_score
                best_value_score = value_score

        if best_match_index >= 0:
            used_pd_indices.add(best_match_index)
            key_scores.append(best_key_score)
            value_scores.append(best_value_score)
        else:
            key_scores.append(0.0)
            value_scores.append(0.0)

    arg_key_score = sum(key_scores) / len(key_scores) if key_scores else 0.0
    arg_value_score = sum(value_scores) / len(value_scores) if value_scores else 0.0
    return name_score, arg_key_score, arg_value_score


# custoimzed reward functions: tool call correctness
def customize_correctness_reward_tool(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = (max_possible_reward - 2) * step / 150 + 2
        min_possible_reward = (min_possible_reward + 2) * step / 150 - 2
        if max_possible_reward > 3.0:
            max_possible_reward = 3.0
        if min_possible_reward < -3.0:
            min_possible_reward = -3.0
    
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        reward = 0.0
        
        if "<tool_call>" not in ans:
            # if "<tool_call>" not in response and "</tool_call>" not in response:
            #     reward = max_possible_reward
            # else:
            #     reward = min_possible_reward
            rewards.append(reward)
            continue

        gt_tool_call = ans.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        gt_tools = gt_tool_call.split("\n")
        gt_tools = [json.loads(tool) for tool in gt_tools] # each diction contains "name" and "parameter"
        
        try:
            # Change here as a constrint in training: if the format is not correct, directly give the lowest possible score
            assert "<tool_call>" in response
            assert "</tool_call>" in response
            pd_tools = response.split("<tool_call>")[1].split("</tool_call>")[0].strip().split("\n")
            pd_tools = [json.loads(tool) for tool in pd_tools]
            reward = compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward) # top reward is 2
        except:
            reward = min_possible_reward
        
        rewards.append(reward)
    
    print("\n======= Reward for <tool call> =======")
    print("Reward function for <tool call> correctness is called ...")
    print(rewards)
    return rewards


def customize_tool_vector_reward(completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    vectors = []

    for response, ans in zip(responses, answer):
        if "<tool_call>" not in ans:
            vectors.append((0.0, 0.0, 0.0))
            continue

        try:
            assert "<tool_call>" in response
            assert "</tool_call>" in response
            gt_tool_call = ans.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            gt_tools = [json.loads(tool) for tool in gt_tool_call.split("\n")]
            pd_tool_call = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            pd_tools = [json.loads(tool) for tool in pd_tool_call.split("\n")]
            vectors.append(compute_tool_call_vector(gt_tools, pd_tools))
        except Exception:
            vectors.append((0.0, 0.0, 0.0))

    return vectors


def compute_attempt_vector(response, ground_truth, step=0):
    answer = [ground_truth]
    completions = [[{"role": "assistant", "content": response}]]
    format_score = customize_format_reward_func(completions, answer, step, 1.0, 0.0)[0]
    if "<tool_call>" not in ground_truth:
        return (format_score, 0.0, 0.0, 0.0)

    try:
        gt_tools = parse_tool_calls(ground_truth)
        pd_tools = parse_tool_calls(response)
        tool_name_score, arg_key_score, arg_value_score = compute_tool_call_vector(gt_tools, pd_tools)
    except Exception:
        tool_name_score = arg_key_score = arg_value_score = 0.0
    return (format_score, tool_name_score, arg_key_score, arg_value_score)


def compute_score(solution_str, ground_truth, step=0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    predict_str = extract_assistant_text(solution_str)
    multi_answer_count = int(os.getenv("MULTI_ANSWER_COUNT", "1"))
    if multi_answer_count > 1 or os.getenv("PAPER_TOOLRL_REWARD", "0") == "1":
        attempts = extract_multi_attempts(predict_str, multi_answer_count)
        candidate_vectors = [
            compute_attempt_vector(attempt, ground_truth, step)
            for attempt in attempts[:multi_answer_count]
        ]
        while len(candidate_vectors) < multi_answer_count:
            candidate_vectors.append((0.0, 0.0, 0.0, 0.0))
        scalar_scores = [sum(vector) / 4 for vector in candidate_vectors]
        best_index = max(range(len(scalar_scores)), key=lambda idx: scalar_scores[idx]) if scalar_scores else 0
        best_vector = candidate_vectors[best_index] if candidate_vectors else (0.0, 0.0, 0.0, 0.0)
        score = scalar_scores[best_index] if scalar_scores else 0.0
        correctness_score = sum(best_vector[1:]) / 3

        return (
            score,
            best_vector[0],
            correctness_score,
            0.0,
            best_vector[1],
            best_vector[2],
            best_vector[3],
            candidate_vectors,
        )
    
    if str(os.getenv("CORRECTMAX1", 0)) == "1":
        print("CORRECTMAX1 is set to 1, so max score is set to 1")
        tool_max_possible = 1.0
        tool_min_possible = -1.0
    else:
        tool_max_possible = 3.0
        tool_min_possible = -3.0
    
    format_max_possible = 1.0
    format_min_possible = 0.0
    
    length_max_possible = 1.0
    length_min_possible = 0.0
    
    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]
    
    fomrat_score = customize_format_reward_func(completions, answer, step, format_max_possible, format_min_possible)[0]
    correctness_score = customize_correctness_reward_tool(completions, answer, step, tool_max_possible, tool_min_possible)[0]
    
    if str(os.getenv("WITHLENGTH", 0)) == "1":
        print("WITHLENGTH is set to 1, so length score is set!")
        length_score = customize_length_reward_func(completions, answer, step, length_max_possible, length_min_possible)[0]
    else:
        length_score = 0

    tool_name_score, arg_key_score, arg_value_score = customize_tool_vector_reward(
        completions, answer
    )[0]

    score = fomrat_score + correctness_score + length_score

    return (
        score,
        fomrat_score,
        correctness_score,
        length_score,
        tool_name_score,
        arg_key_score,
        arg_value_score,
    )
