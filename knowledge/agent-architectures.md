# AI Agent Architecture Patterns

## Architecture 1: Prompt Chaining
### What It Is
A technique where you break a complex task into multiple smaller steps, each handled by a separate LLM call.

### Why Use It
- Clarity: Simplifies complex prompts
- Flexibility: Each step can focus on a specific function

### Key Considerations
- Intermediate Checks
- Overhead management

## Architecture 2: Routing
### What It Is
A workflow that classifies incoming requests (queries) and routes them to specialized prompts, tools, or even different LLM models.

### Why Use It
- Categorization: When your system receives a variety of request types (e.g., refunds vs. tech support), a quick classification step ensures the correct handling.
- Efficiency: You can use smaller or cheaper models for simpler queries and reserve powerful models for complex ones.

### Key Considerations
- Accurate Classification: The initial classification step should be reliable—often a single LLM prompt returning a one-word category.
- Modular Design: Keep separate “handlers” or specialized prompts for each category.

## Architecture 3: Parallelization
### What It Is
Splitting a larger task into independent sub-tasks that can be processed in parallel (often via asyncio in Python).

### Why Use It
- Speed: For tasks that don’t depend on each other’s intermediate results, you can run multiple LLM calls at the same time (e.g., reviewing code from different angles).
- Diverse Opinions: Run the same prompt multiple times to get multiple perspectives, or run different prompts simultaneously.

### Key Considerations
- Concurrency: Use asyncio.gather() or similar concurrency frameworks to manage parallel calls.
- Task Independence: Ensure sub-tasks don’t need each other’s results in real-time. If they do, you might need a different approach (like chaining).

## Architecture 4: Orchestrator-Workers
### What It Is
A two-step (or multi-step) process where an “orchestrator” LLM plans a set of sub-tasks, and “worker” LLM calls carry out each sub-task. The orchestrator decides what needs to be done, then delegates how to do it.

### Why Use It
- Dynamic Planning: When the scope of changes or tasks isn’t known upfront (e.g., code refactoring across multiple files).
- Scalable Delegation: The orchestrator can create a JSON “plan,” and each plan item becomes a “worker” prompt.

### Key Considerations
- Plan Validation: You’ll likely want to parse and validate the orchestrator’s plan (e.g., ensure it’s valid JSON).
- Error Handling: If a worker sub-task fails or returns unexpected output, decide whether to re-plan or continue.

## Architecture 5: Evaluator-Optimizer
### What It Is
An iterative refinement loop where one LLM output is evaluated by another prompt (or the same model in a different phase), and then optimized until it meets certain criteria or a max iteration limit.

### Why Use It
- Incremental Improvement: Great for scenarios like text drafting, code debugging, or translation refinement.
- Clear Criteria: You can define “perfect” or “sufficient” conditions for the output.

### Key Considerations
- Termination: Make sure you have a stopping condition (e.g., “PERFECT” or a max number of iterations).
- Cost vs. Benefit: Each improvement cycle calls the LLM again—use only if incremental improvements are necessary.

## Agents—Autonomous Execution
### What It Is
An LLM “agent” that decides on its own steps and executes tool calls in a loop without direct human input each time.
The agent uses a “decision prompt” to figure out which action/tool to take next (search docs, run a calculation, write to a file, etc.). The loop continues until it declares the task complete.

### Why Use It
- Complex, Unpredictable Tasks: The agent can adapt to new information or partial results, deciding how to proceed at each step.
- Reduced Human Intervention: Once started, the agent can run through multiple steps automatically.

### Key Considerations
- Guardrails: Limit the number of steps or have safety checks to avoid runaway loops.
- Tooling: Provide the agent with well-defined “tools” and instructions on how to use them.
- Debuggability: Keep logs or step-by-step records, because it can be hard to debug if the agent goes astray.

## Combining Patterns
- Routing + Prompt Chaining
- Orchestrator + Parallelization
- Evaluator-Optimizer inside an Autonomous Agent

## Final Thoughts
Each Architecture addresses different requirements. Choose based on:
- Task complexity
- Cost/latency constraints
- Desired autonomy level

## Examples
### Common Helper Function
```python
import os
from groq import Groq

client = Groq()

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
```

### Example: Prompt Chaining
```python
def prompt_chaining_workflow(topic: str, target_language: str) -> dict:
    # Step 1: Generate marketing copy
    generation_prompt = f"Generate a short, creative marketing copy about {topic}."
    marketing_copy = call_llm(generation_prompt)
    
    # Optional check: Ensure the copy is long enough
    if len(marketing_copy) < 20:
        raise ValueError("Generated copy is too short.")
    
    # Step 2: Translate the copy
    translation_prompt = f"Translate the following text to {target_language}:\n\n{marketing_copy}"
    translated_copy = call_llm(translation_prompt)
    
    return {"original": marketing_copy, "translated": translated_copy}

# Example usage:
if __name__ == "__main__":
    result = prompt_chaining_workflow("eco-friendly water bottles", "Spanish")
    print("Original Copy:", result["original"])
    print("Translated Copy:", result["translated"])
```

### Example: Routing
```python
def routing_workflow(query: str) -> dict:
    # Step 1: Classify the query
    classification_prompt = (
        "Classify the following query into 'refund', 'technical', or 'general'. "
        "Answer with one word only.\n\nQuery: " + query
    )
    category = call_llm(classification_prompt).lower()
    if category not in ["refund", "technical", "general"]:
        category = "general"
    
    # Step 2: Choose the specialized prompt
    handlers = {
        "refund": "Handle a refund request professionally. ",
        "technical": "Provide technical troubleshooting steps. ",
        "general": "Answer a general inquiry. "
    }
    response_prompt = handlers.get(category, handlers["general"]) + query
    response = call_llm(response_prompt)
    
    return {"category": category, "response": response}

# Example usage:
if __name__ == "__main__":
    queries = [
        "I want a refund for my purchase.",
        "I can't turn on my device.",
        "Do you have any promotions?"
    ]
    for q in queries:
        res = routing_workflow(q)
        print(f"Query: {q}\nCategory: {res['category']}\nResponse: {res['response']}\n")
```

### Example: Parallelization
```python
import asyncio

async def async_call_llm(prompt: str) -> str:
    return await asyncio.to_thread(lambda: call_llm(prompt))

async def parallelization_workflow(code: str) -> dict:
    review_prompts = {
        "security": "Review the following code for security vulnerabilities:\n",
        "performance": "Review the following code for performance improvements:\n",
        "style": "Review the following code for code style and readability:\n"
    }
    
    tasks = [
        async_call_llm(prefix + code)
        for prefix in review_prompts.values()
    ]
    results = await asyncio.gather(*tasks)
    return dict(zip(review_prompts.keys(), results))

# Example usage:
if __name__ == "__main__":
    sample_code = """
        def process_data(data):
            # Connect to database with hardcoded credentials
            db = Database("admin", "admin123")
            result = db.query("SELECT * FROM data WHERE id=" + data)
            return result
        """
    async def run_parallel():
        feedback = await parallelization_workflow(sample_code)
        for aspect, review in feedback.items():
            print(f"{aspect.upper()} REVIEW:\n{review}\n{'-'*40}")
    
    asyncio.run(run_parallel())
```

### Example: Orchestrator-Workers
```python
import json

async def orchestrator_workers_workflow(task: str, files: dict) -> dict:
    # Step 1: Create a plan using the orchestrator
    plan_prompt = f"""
        You are a senior developer. Task: {task}
        Files available: {list(files.keys())}

        Provide a plan in valid JSON format. Each key should be a file name and its value a brief description of changes.
        Example: {{"file1.py": "Add input validation"}}
        """
    plan_json = await asyncio.to_thread(lambda: call_llm(plan_prompt))
    plan = json.loads(plan_json)
    
    # Step 2: Execute the plan using worker calls
    tasks = []
    for filename, change_desc in plan.items():
        worker_prompt = f"""
            File: {filename}
            Current content:
            {files.get(filename, "")}

            Task: {task}
            Change required: {change_desc}

            Return the updated file content only.
            """
        tasks.append(asyncio.to_thread(lambda prompt=worker_prompt: call_llm(prompt)))
    
    updated_contents = await asyncio.gather(*tasks)
    return dict(zip(plan.keys(), updated_contents))

# Example usage:
if __name__ == "__main__":
    async def run_orchestrator():
        task_desc = "Improve error handling and input validation in all functions."
        files_dict = {
            "service.py": """
                def process_request(data):
                    result = process(data)
                    return result
                """,
                            "utils.py": """
                def helper(x):
                    return x * 2
            """}
        updates = await orchestrator_workers_workflow(task_desc, files_dict)
        for fname, content in updates.items():
            print(f"Updated {fname}:\n{content}\n{'-'*50}")
    
    asyncio.run(run_orchestrator())
```

### Example: Evaluator-Optimizer
```python
def evaluator_optimizer_workflow(text: str, target_language: str, max_iterations: int = 3) -> str:
    # Initial translation
    translation_prompt = f"Translate the following text into {target_language}:\n\n{text}"
    translation = call_llm(translation_prompt)
    
    iteration = 1
    while iteration <= max_iterations:
        print(f"\nIteration {iteration}")
        print("Current Translation:", translation)
        
        # Evaluate the translation quality
        evaluation_prompt = (
            f"Evaluate the quality of this translation. "
            f"If it's perfect, reply with 'PERFECT'. Otherwise, explain what needs to be improved.\n\n{translation}"
        )
        evaluation = call_llm(evaluation_prompt)
        print("Evaluation:", evaluation)
        
        if "PERFECT" in evaluation.upper():
            print("Translation deemed perfect.")
            break
        
        # Optimize the translation based on feedback
        improvement_prompt = (
            f"Improve the following translation based on the feedback below. "
            f"Return only the improved translation.\n\nFeedback: {evaluation}\n\nCurrent translation:\n{translation}"
        )
        translation = call_llm(improvement_prompt)
        iteration += 1
    
    return translation

# Example usage:
if __name__ == "__main__":
    original_text = "The early bird catches the worm, but the second mouse gets the cheese."
    final_translation = evaluator_optimizer_workflow(original_text, "Spanish")
    print("\nFinal Translation:", final_translation)
```

### Example: Minimal Agent Implementation
```python
import json

class MinimalAgent:
    def __init__(self):
        self.tools = {
            "search_docs": self.search_docs,
            "calculate": self.calculate,
            "write_file": self.write_file
        }
        self.tools_description = (
            "search_docs(query): returns documentation snippet for 'pricing' or 'features'.\n"
            "calculate(expression): evaluates a math expression.\n"
            "write_file(content): simulates writing content to a file."
        )
    
    def search_docs(self, query: str) -> str:
        docs = {
            "pricing": "Basic plan: $10/month; Pro plan: $20/month",
            "features": "Features include AI chat, file storage, and API access"
        }
        return docs.get(query.lower(), "No info found")
    
    def calculate(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except Exception:
            return "Error evaluating expression"
    
    def write_file(self, content: str) -> str:
        return f"File updated with: {content}"
    
    def get_next_action(self, task: str, step: int) -> dict:
        """
        Ask LLM which tool to use next. It should return a JSON object, e.g.,
        {"tool": "calculate", "arguments": "2+2"} or {"tool": "TASK_COMPLETE"}.
        """
        prompt = f"""
Current Task: {task}
Step: {step}
Available tools: {list(self.tools.keys())}
Tools description: {self.tools_description}

Respond with a JSON object in one of these two formats:
1. {{"tool": "tool_name", "arguments": "your arguments"}}
2. {{"tool": "TASK_COMPLETE"}}
"""
        response = call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"tool": "TASK_COMPLETE"}
    
    def solve_task(self, task: str) -> list:
        step = 0
        results = []
        while True:
            step += 1
            action = self.get_next_action(task, step)
            tool = action.get("tool")
            if tool == "TASK_COMPLETE":
                print("Task complete.")
                break
            args = action.get("arguments", "")
            if tool not in self.tools:
                results.append(f"Tool '{tool}' not recognized.")
                continue
            result = self.tools[tool](args)
            results.append(result)
            print(f"Step {step}: Called '{tool}' with args '{args}' → Result: {result}")
        return results

# Example usage:
if __name__ == "__main__":
    agent = MinimalAgent()
    task_description = """
Create a pricing summary:
1. Search for pricing information.
2. Calculate the yearly cost for the basic plan ($10/month).
3. Write a summary to a file.
"""
    outputs = agent.solve_task(task_description)
    print("\nFinal Agent Outputs:", outputs)
```

### Examples using Behaviour Trees
```python
import os
import json
import py_trees
from py_trees.decorators import RepeatUntilSuccess
from openai import OpenAI

# ------------------------------------------------------------
# Helper: GPT-4 Call Function
# ------------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_gpt4(prompt: str) -> str:
    """
    Sends a prompt to GPT-4 and returns its response (stripped).
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ------------------------------------------------------------
# Workflow 1: Prompt Chaining (Generate copy then translate)
# ------------------------------------------------------------
class GenerateMarketingCopy(py_trees.behaviour.Behaviour):
    def __init__(self, name, topic):
        super(GenerateMarketingCopy, self).__init__(name)
        self.topic = topic

    def update(self):
        prompt = f"Generate a short, creative marketing copy about {self.topic}."
        result = call_gpt4(prompt)
        if len(result) < 20:
            self.feedback_message = "Generated copy is too short."
            return py_trees.common.Status.FAILURE
        # Write to the blackboard
        client_bb = self.attach_blackboard_client(name="GenerateMarketingCopy")
        client_bb.register_key(key="marketing_copy", access=py_trees.common.Access.WRITE)
        client_bb.marketing_copy = result
        self.feedback_message = "Marketing copy generated."
        return py_trees.common.Status.SUCCESS

class TranslateCopy(py_trees.behaviour.Behaviour):
    def __init__(self, name, target_language):
        super(TranslateCopy, self).__init__(name)
        self.target_language = target_language

    def update(self):
        client_bb = self.attach_blackboard_client(name="TranslateCopy")
        client_bb.register_key(key="marketing_copy", access=py_trees.common.Access.READ)
        client_bb.register_key(key="translated_copy", access=py_trees.common.Access.WRITE)
        marketing_copy = client_bb.marketing_copy
        if not marketing_copy:
            self.feedback_message = "No marketing copy found."
            return py_trees.common.Status.FAILURE
        prompt = f"Translate the following text to {self.target_language}:\n\n{marketing_copy}"
        translation = call_gpt4(prompt)
        client_bb.translated_copy = translation
        self.feedback_message = "Translation completed."
        return py_trees.common.Status.SUCCESS

def run_workflow_1():
    print("\n--- Running Workflow 1: Prompt Chaining ---")
    # Build a tree with a Sequence that first generates then translates
    root = py_trees.composites.Sequence("PromptChaining")
    root.add_children([
        GenerateMarketingCopy("GenerateCopy", topic="eco-friendly water bottles"),
        TranslateCopy("TranslateCopy", target_language="Spanish")
    ])
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(timeout=15)
    tree.tick_once()  # Single tick; in real usage, you might tick repeatedly.
    bb = py_trees.blackboard.Blackboard()
    print("Marketing Copy:", getattr(bb, "marketing_copy", None))
    print("Translated Copy:", getattr(bb, "translated_copy", None))

# ------------------------------------------------------------
# Workflow 2: Routing (Classify query then handle accordingly)
# ------------------------------------------------------------
class ClassifyQuery(py_trees.behaviour.Behaviour):
    def __init__(self, name, query):
        super(ClassifyQuery, self).__init__(name)
        self.query = query

    def update(self):
        prompt = (f"Classify the following query into 'refund', 'technical', or 'general'. "
                  f"Answer with one word only.\n\nQuery: {self.query}")
        category = call_gpt4(prompt).lower().strip()
        if category not in ["refund", "technical", "general"]:
            category = "general"
        client_bb = self.attach_blackboard_client(name="ClassifyQuery")
        client_bb.register_key(key="category", access=py_trees.common.Access.WRITE)
        client_bb.category = category
        self.feedback_message = f"Query classified as {category}."
        return py_trees.common.Status.SUCCESS

class HandleRefund(py_trees.behaviour.Behaviour):
    def __init__(self, name, query):
        super(HandleRefund, self).__init__(name)
        self.query = query

    def update(self):
        client_bb = self.attach_blackboard_client(name="HandleRefund")
        client_bb.register_key(key="category", access=py_trees.common.Access.READ)
        if client_bb.category != "refund":
            return py_trees.common.Status.FAILURE
        prompt = "Handle a refund request professionally. " + self.query
        response = call_gpt4(prompt)
        client_bb.register_key(key="response", access=py_trees.common.Access.WRITE)
        client_bb.response = response
        self.feedback_message = "Refund handled."
        return py_trees.common.Status.SUCCESS

class HandleTechnical(py_trees.behaviour.Behaviour):
    def __init__(self, name, query):
        super(HandleTechnical, self).__init__(name)
        self.query = query

    def update(self):
        client_bb = self.attach_blackboard_client(name="HandleTechnical")
        client_bb.register_key(key="category", access=py_trees.common.Access.READ)
        if client_bb.category != "technical":
            return py_trees.common.Status.FAILURE
        prompt = "Provide technical troubleshooting steps. " + self.query
        response = call_gpt4(prompt)
        client_bb.register_key(key="response", access=py_trees.common.Access.WRITE)
        client_bb.response = response
        self.feedback_message = "Technical response generated."
        return py_trees.common.Status.SUCCESS

class HandleGeneral(py_trees.behaviour.Behaviour):
    def __init__(self, name, query):
        super(HandleGeneral, self).__init__(name)
        self.query = query

    def update(self):
        prompt = "Answer a general inquiry. " + self.query
        response = call_gpt4(prompt)
        client_bb = self.attach_blackboard_client(name="HandleGeneral")
        client_bb.register_key(key="response", access=py_trees.common.Access.WRITE)
        client_bb.response = response
        self.feedback_message = "General response generated."
        return py_trees.common.Status.SUCCESS

def run_workflow_2():
    print("\n--- Running Workflow 2: Routing ---")
    query = "I want a refund for my purchase."
    root = py_trees.composites.Sequence("Routing")
    root.add_children([
        ClassifyQuery("ClassifyQuery", query=query),
        py_trees.composites.Selector("HandleQuery", children=[
            HandleRefund("HandleRefund", query=query),
            HandleTechnical("HandleTechnical", query=query),
            HandleGeneral("HandleGeneral", query=query)
        ])
    ])
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(timeout=15)
    tree.tick_once()
    bb = py_trees.blackboard.Blackboard()
    print("Category:", getattr(bb, "category", None))
    print("Response:", getattr(bb, "response", None))

# ------------------------------------------------------------
# Workflow 3: Parallelization (Run code reviews concurrently)
# ------------------------------------------------------------
class ReviewSecurity(py_trees.behaviour.Behaviour):
    def __init__(self, name, code):
        super(ReviewSecurity, self).__init__(name)
        self.code = code

    def update(self):
        prompt = "Review the following code for security vulnerabilities:\n" + self.code
        review = call_gpt4(prompt)
        client_bb = self.attach_blackboard_client(name="ReviewSecurity")
        client_bb.register_key(key="security_review", access=py_trees.common.Access.WRITE)
        client_bb.security_review = review
        self.feedback_message = "Security review done."
        return py_trees.common.Status.SUCCESS

class ReviewPerformance(py_trees.behaviour.Behaviour):
    def __init__(self, name, code):
        super(ReviewPerformance, self).__init__(name)
        self.code = code

    def update(self):
        prompt = "Review the following code for performance improvements:\n" + self.code
        review = call_gpt4(prompt)
        client_bb = self.attach_blackboard_client(name="ReviewPerformance")
        client_bb.register_key(key="performance_review", access=py_trees.common.Access.WRITE)
        client_bb.performance_review = review
        self.feedback_message = "Performance review done."
        return py_trees.common.Status.SUCCESS

class ReviewStyle(py_trees.behaviour.Behaviour):
    def __init__(self, name, code):
        super(ReviewStyle, self).__init__(name)
        self.code = code

    def update(self):
        prompt = "Review the following code for code style and readability:\n" + self.code
        review = call_gpt4(prompt)
        client_bb = self.attach_blackboard_client(name="ReviewStyle")
        client_bb.register_key(key="style_review", access=py_trees.common.Access.WRITE)
        client_bb.style_review = review
        self.feedback_message = "Style review done."
        return py_trees.common.Status.SUCCESS

def run_workflow_3():
    print("\n--- Running Workflow 3: Parallelization ---")
    code = """
def process_data(data):
    db = Database("admin", "admin123")
    result = db.query("SELECT * FROM data WHERE id=" + data)
    return result
"""
    # Create a Parallel composite so that all reviews run concurrently.
    root = py_trees.composites.Parallel("CodeReview",
                                        policy=py_trees.common.ParallelPolicy.SuccessOnAll())
    root.add_children([
        ReviewSecurity("ReviewSecurity", code=code),
        ReviewPerformance("ReviewPerformance", code=code),
        ReviewStyle("ReviewStyle", code=code)
    ])
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(timeout=15)
    tree.tick_once()
    bb = py_trees.blackboard.Blackboard()
    reviews = {
        "security": getattr(bb, "security_review", None),
        "performance": getattr(bb, "performance_review", None),
        "style": getattr(bb, "style_review", None)
    }
    for aspect, review in reviews.items():
        print(f"{aspect.capitalize()} Review:\n{review}\n{'-'*40}")

# ------------------------------------------------------------
# Workflow 4: Orchestrator-Workers (Plan then update files)
# ------------------------------------------------------------
class GeneratePlan(py_trees.behaviour.Behaviour):
    def __init__(self, name, task, files):
        super(GeneratePlan, self).__init__(name)
        self.task = task
        self.files = files

    def update(self):
        prompt = f"""
You are a senior developer.
Task: {self.task}
Files available: {list(self.files.keys())}
Provide a plan in valid JSON format where each key is a file name and its value is a brief description of changes.
Example: {{"file1.py": "Add input validation"}}
"""
        plan_json = call_gpt4(prompt)
        try:
            plan = json.loads(plan_json)
        except Exception:
            self.feedback_message = "Plan generation failed."
            return py_trees.common.Status.FAILURE
        client_bb = self.attach_blackboard_client(name="GeneratePlan")
        client_bb.register_key(key="plan", access=py_trees.common.Access.WRITE)
        client_bb.plan = plan
        self.feedback_message = "Plan generated."
        return py_trees.common.Status.SUCCESS

class UpdateFile(py_trees.behaviour.Behaviour):
    def __init__(self, name, filename, files, task):
        super(UpdateFile, self).__init__(name)
        self.filename = filename
        self.files = files
        self.task = task

    def update(self):
        client_bb = self.attach_blackboard_client(name="UpdateFile")
        client_bb.register_key(key="plan", access=py_trees.common.Access.READ)
        plan = client_bb.plan
        if self.filename not in plan:
            self.feedback_message = f"No plan for {self.filename}."
            return py_trees.common.Status.FAILURE
        change_desc = plan[self.filename]
        original = self.files.get(self.filename, "")
        prompt = f"""
File: {self.filename}
Current content:
{original}

Task: {self.task}
Change required: {change_desc}

Return only the updated file content.
"""
        updated_content = call_gpt4(prompt)
        client_bb.register_key(key="updated_files", access=py_trees.common.Access.WRITE)
        if not hasattr(client_bb, "updated_files"):
            client_bb.updated_files = {}
        client_bb.updated_files[self.filename] = updated_content
        self.feedback_message = f"Updated {self.filename}."
        return py_trees.common.Status.SUCCESS

def run_workflow_4():
    print("\n--- Running Workflow 4: Orchestrator-Workers ---")
    task = "Improve error handling and add input validation."
    files = {
        "service.py": "def process_request(data):\n    result = process(data)\n    return result",
        "utils.py": "def helper(x):\n    return x * 2"
    }
    bb = py_trees.blackboard.Blackboard()
    bb.files = files
    bb.task = task
    root = py_trees.composites.Sequence("OrchestratorWorkers")
    root.add_children([
        GeneratePlan("GeneratePlan", task=task, files=files),
        UpdateFile("UpdateService", filename="service.py", files=files, task=task),
        UpdateFile("UpdateUtils", filename="utils.py", files=files, task=task)
    ])
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(timeout=15)
    tree.tick_once()
    bb = py_trees.blackboard.Blackboard()
    for fname, content in getattr(bb, "updated_files", {}).items():
        print(f"Updated {fname}:\n{content}\n{'-'*50}")

# ------------------------------------------------------------
# Workflow 5: Evaluator-Optimizer (Iterative translation improvement)
# ------------------------------------------------------------
class InitialTranslation(py_trees.behaviour.Behaviour):
    def __init__(self, name, text, target_language):
        super(InitialTranslation, self).__init__(name)
        self.text = text
        self.target_language = target_language

    def update(self):
        prompt = f"Translate the following text into {self.target_language}:\n\n{self.text}"
        translation = call_gpt4(prompt)
        client_bb = self.attach_blackboard_client(name="InitialTranslation")
        client_bb.register_key(key="translation", access=py_trees.common.Access.WRITE)
        client_bb.translation = translation
        self.feedback_message = "Initial translation done."
        return py_trees.common.Status.SUCCESS

class EvaluateTranslation(py_trees.behaviour.Behaviour):
    def update(self):
        client_bb = self.attach_blackboard_client(name="EvaluateTranslation")
        client_bb.register_key(key="translation", access=py_trees.common.Access.READ)
        translation = client_bb.translation
        prompt = (f"Evaluate the quality of this translation. If it's perfect, reply with 'PERFECT'. "
                  f"Otherwise, explain what to improve.\n\n{translation}")
        evaluation = call_gpt4(prompt)
        client_bb.register_key(key="evaluation", access=py_trees.common.Access.WRITE)
        client_bb.evaluation = evaluation
        client_bb.register_key(key="is_perfect", access=py_trees.common.Access.WRITE)
        client_bb.is_perfect = ("PERFECT" in evaluation.upper())
        self.feedback_message = "Evaluation complete."
        return py_trees.common.Status.SUCCESS

class ImproveTranslation(py_trees.behaviour.Behaviour):
    def update(self):
        client_bb = self.attach_blackboard_client(name="ImproveTranslation")
        client_bb.register_key(key="translation", access=py_trees.common.Access.READ)
        client_bb.register_key(key="evaluation", access=py_trees.common.Access.READ)
        translation = client_bb.translation
        evaluation = client_bb.evaluation
        prompt = (f"Improve the following translation based on the feedback below. "
                  f"Return only the improved translation.\n\nFeedback: {evaluation}\n\nCurrent translation:\n{translation}")
        improved = call_gpt4(prompt)
        client_bb.translation = improved
        self.feedback_message = "Translation improved."
        return py_trees.common.Status.SUCCESS

def run_workflow_5():
    print("\n--- Running Workflow 5: Evaluator-Optimizer ---")
    text = "The early bird catches the worm, but the second mouse gets the cheese."
    target_language = "Spanish"
    root = py_trees.composites.Sequence("EvaluatorOptimizer")
    root.add_children([
        InitialTranslation("InitialTranslation", text=text, target_language=target_language),
        EvaluateTranslation("EvaluateTranslation")
    ])
    # Use a repeat decorator to iterate improvement until translation is perfect or 3 iterations
    repeater = RepeatUntilSuccess(child=ImproveTranslation("ImproveTranslation"), num_iterations=3)
    root.add_child(repeater)
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(timeout=15)
    tree.tick_once()
    bb = py_trees.blackboard.Blackboard()
    print("Final Translation:", getattr(bb, "translation", None))
    print("Final Evaluation:", getattr(bb, "evaluation", None))

# ------------------------------------------------------------
# Workflow 6: Agents—Autonomous Execution
# (An agent that decides which tool to call until the task is complete)
# ------------------------------------------------------------
class GetNextAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, task):
        super(GetNextAction, self).__init__(name)
        self.task = task
        self.step = 0

    def update(self):
        self.step += 1
        prompt = f"""
Current Task: {self.task}
Step: {self.step}
Available tools: search_docs, calculate, write_file
Tools description:
  - search_docs(query): returns docs snippet for 'pricing' or 'features'.
  - calculate(expression): evaluates a math expression.
  - write_file(content): simulates writing to a file.
Respond with a JSON object:
  {{ "tool": "tool_name", "arguments": "your arguments" }}
or {{ "tool": "TASK_COMPLETE" }} if done.
"""
        response = call_gpt4(prompt)
        try:
            action = json.loads(response)
        except Exception:
            action = {"tool": "TASK_COMPLETE"}
        client_bb = self.attach_blackboard_client(name="GetNextAction")
        client_bb.register_key(key="agent_action", access=py_trees.common.Access.WRITE)
        client_bb.agent_action = action
        self.feedback_message = f"Received action: {action}"
        return py_trees.common.Status.SUCCESS

class ExecuteTool(py_trees.behaviour.Behaviour):
    def update(self):
        client_bb = self.attach_blackboard_client(name="ExecuteTool")
        client_bb.register_key(key="agent_action", access=py_trees.common.Access.READ)
        action = client_bb.agent_action
        tool = action.get("tool", "")
        args = action.get("arguments", "")
        if tool == "TASK_COMPLETE":
            client_bb.register_key(key="agent_complete", access=py_trees.common.Access.WRITE)
            client_bb.agent_complete = True
            self.feedback_message = "Task complete."
            return py_trees.common.Status.SUCCESS
        result = ""
        if tool == "search_docs":
            docs = {"pricing": "Basic plan: $10/month; Pro plan: $20/month",
                    "features": "AI chat, file storage, API access"}
            result = docs.get(args.lower(), "No info found")
        elif tool == "calculate":
            try:
                result = str(eval(args))
            except Exception:
                result = "Error"
        elif tool == "write_file":
            result = f"File updated with: {args}"
        else:
            result = f"Unknown tool: {tool}"
        client_bb.register_key(key="agent_results", access=py_trees.common.Access.WRITE)
        if not hasattr(client_bb, "agent_results"):
            client_bb.agent_results = []
        client_bb.agent_results.append({tool: result})
        self.feedback_message = f"Executed {tool} with args '{args}', result: {result}"
        return py_trees.common.Status.SUCCESS

def run_workflow_6():
    print("\n--- Running Workflow 6: Autonomous Agent ---")
    task = "Create a pricing summary: search for pricing info, calculate yearly cost for a $10/month plan, and write a summary."
    bb = py_trees.blackboard.Blackboard()
    bb.task = task
    bb.agent_complete = False
    root = py_trees.composites.Sequence("AutonomousAgent")
    root.add_children([
        GetNextAction("GetNextAction", task=task),
        ExecuteTool("ExecuteTool")
    ])
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(timeout=15)
    iteration = 0
    # Loop until the agent signals completion or max iterations reached.
    while not getattr(bb, "agent_complete", False) and iteration < 5:
        tree.tick_once()
        iteration += 1
    print("Agent Task Complete. Results:")
    for res in getattr(bb, "agent_results", []):
        print(res)

# ------------------------------------------------------------
# Main: Run All Workflows
# ------------------------------------------------------------
if __name__ == "__main__":
    run_workflow_1()
    run_workflow_2()
    run_workflow_3()
    run_workflow_4()
    run_workflow_5()
    run_workflow_6()
```