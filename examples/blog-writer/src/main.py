import time
import py_trees
import openai
import threading
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

llm_client = Groq()
RATE_LIMIT_DELAY = 2  # Delay in seconds to respect the 30 requests/min limit

# Blackboard setup
class InitializeAgent(py_trees.behaviour.Behaviour):
    def __init__(self, name="InitializeAgent"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key="goal", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="plan", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="content", access=py_trees.common.Access.WRITE)
        self.blackboard.goal = "Write a short blog post about AI trends."
        self.blackboard.plan = None
        self.blackboard.content = None
    
    def update(self):
        return py_trees.common.Status.SUCCESS

# Helper function for LLM calls
def call_llm(model, messages, blackboard_key, behaviour):
    """Run LLM call in a separate thread and update blackboard asynchronously."""
    def worker():
        time.sleep(RATE_LIMIT_DELAY)  # Respect rate limit
        response = llm_client.chat.completions.create(model=model, messages=messages)
        result = response.choices[0].message.content
        print(f"[DEBUG] LLM Response Received for {blackboard_key}: {result}")
        behaviour.blackboard.set(blackboard_key, result)
        behaviour.processing = False  # Mark process as complete

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

# LLM Plan Node
class LLMPlanNode(py_trees.behaviour.Behaviour):
    def __init__(self, name="LLMPlanNode", model="llama-3.3-70b-versatile"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key="goal", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="plan", access=py_trees.common.Access.WRITE)
        self.model = model
        self.processing = False
    
    def update(self):
        if self.blackboard.plan:
            print(f"[DEBUG] {self.name} - Plan Ready: {self.blackboard.plan}")
            return py_trees.common.Status.SUCCESS

        if not self.processing:
            print(f"[DEBUG] {self.name} - Requesting plan from LLM")
            self.processing = True
            call_llm(self.model, 
                     [{"role": "system", "content": "You are an AI planner."},
                      {"role": "user", "content": f"Generate a structured plan to: {self.blackboard.goal}"}],
                     "plan", self)
        
        return py_trees.common.Status.RUNNING

# LLM Execute Node
class ExecutePlanNode(py_trees.behaviour.Behaviour):
    def __init__(self, name="ExecutePlanNode", model="llama-3.3-70b-versatile"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key="plan", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="content", access=py_trees.common.Access.WRITE)
        self.model = model
        self.processing = False
    
    def update(self):
        if not self.blackboard.plan:
            return py_trees.common.Status.FAILURE

        if self.blackboard.content:
            print(f"[DEBUG] {self.name} - Execution complete.")
            return py_trees.common.Status.SUCCESS

        if not self.processing:
            print(f"[DEBUG] {self.name} - Executing plan: {self.blackboard.plan}")
            self.processing = True
            call_llm(self.model, 
                     [{"role": "system", "content": "You are an AI content writer."},
                      {"role": "user", "content": f"Write a short blog post following this plan: {self.blackboard.plan}"}],
                     "content", self)

        return py_trees.common.Status.RUNNING

# Constructing the Behavior Tree
def create_agent_tree():
    root = py_trees.composites.Sequence(name="AgentRoot", memory=True)
    init_node = InitializeAgent("Init")
    main_sequence = py_trees.composites.Sequence(name="MainSequence", memory=True)
    plan_node = LLMPlanNode("Plan")
    execute_node = ExecutePlanNode("Execute")
    
    main_sequence.add_children([plan_node, execute_node])
    root.add_children([init_node, main_sequence])
    return root

if __name__ == "__main__":
    tree = create_agent_tree()
    print(py_trees.display.ascii_tree(tree))
    bt = py_trees.trees.BehaviourTree(tree)
    
    while True:
        print("[DEBUG] Running behavior tree...")
        bt.tick()
        time.sleep(20)  # Allow background threads to complete work
cd 