
_CONCISE_SUFFIX = "\n\nIMPORTANT: Reply with a single short sentence (under 20 tokens). Be concise and informative; skip preamble."

# Soft concision (for the strategize->compute->verify text-MAS pipeline): unlike the
# blunt single-sentence suffix above, this lets the compute agent still show its
# multi-step work — it only trims padding so the producer (verify) keeps 4096 budget.
_CONCISE_SOFT_SUFFIX = "\n\nBe concise: give only the essential steps and the key result. Do not restate the problem or add commentary."


def _minimal_prompt(question: str) -> str:
    """Minimal non-judger prompt for reasoning-distilled models.

    R1-Distill and similar models' training distribution is incompatible with
    the verbose 'You are a Planner Agent' framing. This template looks much
    closer to how those models were trained — a direct problem-solving prompt.
    """
    return f"Solve this problem step by step:\n\n{question}"


def _apply_concise(user_prompt: str, role: str, args) -> str:
    """Append a 'be concise' instruction for non-(judger/verify) agents when a
    concision flag is set. Two strengths:
      --concise_pipeline_prompt  : SOFT (essential steps only). Frees 4096 budget
                                   for the text-producer without crippling compute.
      --concise_nonjudger_prompt : BLUNT (single short sentence). For short-budget
                                   apples-to-apples latent comparisons.
    Soft wins if both are set. Text-producers (judger/verify) always keep full prompt.
    """
    if role in ("judger", "verify"):  # text-producers keep their full prompt
        return user_prompt
    if args is None:
        return user_prompt
    if getattr(args, "concise_pipeline_prompt", False):
        return user_prompt.rstrip() + _CONCISE_SOFT_SUFFIX
    if getattr(args, "concise_nonjudger_prompt", False):
        return user_prompt.rstrip() + _CONCISE_SUFFIX
    return user_prompt


def _apply_minimal(role: str, question: str, args, original: str) -> str:
    """If --minimal_persona_prompts is set and role isn't a text-producer, replace
    the role-specific prompt with a minimal problem-solving template."""
    if role in ("judger", "verify"):
        return original
    if args is None or not getattr(args, "minimal_persona_prompts", False):
        return original
    return _minimal_prompt(question)


def build_agent_message_sequential_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if role == "planner":
        user_prompt = f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""
    
    elif role == "critic":
        user_prompt = f"""
Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""
    
    elif role == "refiner":
        user_prompt = f"""
Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:
"""
    
    elif role == "judger":
        if args.task in ['gsm8k', 'aime2024', 'aime2025', 'math500']:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        
        elif args.task in ["arc_easy", "arc_challenge", "gpqa", 'medqa']:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif args.task in ["mbppplus", "humanevalplus"]:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve.

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block.

Now, reason step by step and output the final answer inside ```python
YOUR_PYTHON_CODE
```.
"""

        elif args.task in ["winogrande"]:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        else:
            raise NotImplementedError(f"Task {args.task} not implemented in v5 judger prompt.")

    # --- Math persona set: strategize -> compute -> verify ---
    # strategize/compute do latent steps (no text emitted); verify is the
    # text-producer (TEXT_PRODUCER_ROLES) that emits the final \boxed answer.
    elif role == "strategize":
        user_prompt = f"""You are a Strategy Agent. Given a math problem, devise a clear high-level approach: identify the key idea(s), which method or theorem to apply, and the sequence of steps needed. Do NOT carry out the arithmetic or state a final answer.

Question: {question}

Now outline your solution strategy below:
"""

    elif role == "compute":
        user_prompt = f"""You are a Computation Agent. A solution strategy for the problem is provided in latent KV representation. Execute that strategy: carry out the algebra and arithmetic step-by-step, tracking intermediate results, to work toward the answer.

Question: {question}

Now perform the computation below:
"""

    elif role == "verify":
        user_prompt = f"""Target Question: {question}

You are a Verification Agent. You are provided with latent information (a strategy and a computation for this problem) for reference; it may contain mistakes or irrelevant content. Carefully check the reasoning, correct any errors, and determine the final answer.

Reason step-by-step to verify and solve the Target Question, then output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    user_prompt = _apply_minimal(role, question, args, user_prompt)
    user_prompt = _apply_concise(user_prompt, role, args)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def build_agent_message_hierarchical_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if args.task in ['gsm8k', 'aime2024', 'aime2025', 'math500']:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""

    elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:

        if args.task == "medqa":

            if role == "planner":
                user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:
"""
            elif role == "critic":
                user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}     

Your response:
"""
            elif role == "refiner":
                user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:       
"""
            elif role == "judger":

                user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:
"""

        else:
            if role == "planner":
                user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:
"""
    
            elif role == "critic":
                user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}     

Your response:
"""
    
            elif role == "refiner":
                user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:       
"""
            elif role == "judger":

                user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:
"""

    elif args.task in ["mbppplus", "humanevalplus"]:
        
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 
    
Input Question: {question}

Your response:
"""

    elif args.task in ["winogrande"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""

    user_content = _apply_minimal(role, question, args, user_content)
    user_content = _apply_concise(user_content, role, args)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_sequential_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["text_mas"], "only for text_mas method"
    assert "qwen" in args.model_name.lower(), "only for qwen models"

    # truncate context if needed
    ctx = context[: args.text_mas_context_length]

    if role == "planner":
        user_content = f"""
You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{question}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below:
"""

    elif role == "critic":
        user_content = f"""
You are a Critic Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format.

Your job is to carefully evaluate the correctness and completeness of the plan and provide helpful feedback.

## Input Question:
{question}

## Plan from Planner Agent:
{ctx}

## Format your response as follows:
Critic Agent's Output:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""

    elif role == "refiner":
        user_content = f"""
You are a Refiner Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan together with Critic Agent's feedback in text format.

Your job is to incorporate the feedback and produce an improved, refined step-by-step plan.

## Input Question:
{question}

## Original Plan and Critic Feedback:
{ctx}

## Format your response as follows:
Refiner Agent's Output:
[Your refined and improved plan here]

Make sure your output plan is logically correct, concise, and sufficient to guide final problem solving.
Now, output your refined plan below:
"""

    elif role == "judger":
        task = getattr(args, "task", None)

        if task in ["gsm8k", "aime2024", "aime2025", "math500"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["mbppplus", "humanevalplus"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
"""
            
        elif task in ["winogrande"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        else:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and present your final answer clearly at the end.
"""

    # --- Math persona set (text MAS): strategize -> compute -> verify ---
    # Each agent's text output is appended to the running context; the next agent
    # sees it via {ctx}. verify is the text-producer that emits the final answer.
    elif role == "strategize":
        user_content = f"""You are a Strategy Agent in a sequential pipeline (strategize -> compute -> verify). Given a math problem, devise a clear high-level approach: identify the key idea(s), which method or theorem to apply, and the sequence of steps needed. Do NOT carry out the arithmetic or state a final answer.

Question: {question}

Now outline your solution strategy below:
"""

    elif role == "compute":
        user_content = f"""You are a Computation Agent in a sequential pipeline (strategize -> compute -> verify). You are given the problem and a strategy from the Strategy Agent. Execute that strategy: carry out the algebra and arithmetic step-by-step, tracking intermediate results, to work toward the answer.

Question: {question}

Strategy from the previous agent:
{ctx}

Now perform the computation below:
"""

    elif role == "verify":
        user_content = f"""Target Question: {question}

You are a Verification Agent in a sequential pipeline (strategize -> compute -> verify). The compute agent has already worked out a solution below; it may contain errors.

Work from previous agents:
{ctx}

Briefly check the computation's key steps and its final result. If it is correct, confirm it; if you find a clear error, fix only what is needed — do NOT redo the whole solution from scratch. Then output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    user_content = _apply_minimal(role, question, args, user_content)
    user_content = _apply_concise(user_content, role, args)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_hierarchical_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    assert method in ["text_mas"], "this prompt only for text_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"
    
    if args.task in ['gsm8k', 'aime2024', 'aime2025', 'math500']:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":

            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["mbppplus", "humanevalplus"]:
        
        if role == "planner":
            user_content = f"""
You are a math agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "critic":
            user_content = f"""
You are a science agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "refiner":
            user_content = f"""
You are a code agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer in markdown python code block.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["winogrande"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""

    user_content = _apply_minimal(role, question, args, user_content)
    user_content = _apply_concise(user_content, role, args)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_single_agent(question: str, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert args.method in ["baseline"], "this prompt only for baseline method (single agent)"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    task = args.task

    if task in ["gsm8k", "aime2024", "aime2025", "math500"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["mbppplus", "humanevalplus"]:
        user_content = f"""
Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
"""

    elif task in ["winogrande"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    else:
        user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.
Present your reasoning, and then clearly state your final answer at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

