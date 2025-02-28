import py_trees
import groq
import os

# Initialize Groq client
client = groq.Client()

# Blackboard for shared memory using namespaces
blackboard = py_trees.blackboard.Blackboard()
blackboard.register_key(key="task/description", access=py_trees.common.Access.WRITE)
blackboard.register_key(key="task/context", access=py_trees.common.Access.WRITE)
blackboard.register_key(key="files/search_results", access=py_trees.common.Access.WRITE)
blackboard.register_key(key="code/generated", access=py_trees.common.Access.WRITE)
blackboard.register_key(key="validation/test_results", access=py_trees.common.Access.WRITE)

class ClarifyTask(py_trees.behaviour.Behaviour):
    @py_trees.decorators.OneShot()
    def update(self):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Clarify coding tasks for AI agents."},
                {"role": "user", "content": "What is the required implementation? Provide any necessary context."}
            ]
        )
        task_info = response.choices[0].message.content
        blackboard.set("task/description", task_info)
        blackboard.set("task/context", f"Additional context: {task_info}")
        return py_trees.common.Status.SUCCESS

class SearchFiles(py_trees.behaviour.Behaviour):
    def update(self):
        search_results = [f for f in os.listdir("./") if f.endswith(".py")]
        blackboard.set("files/search_results", search_results)
        return py_trees.common.Status.SUCCESS

class GenerateCode(py_trees.behaviour.Behaviour):
    @py_trees.decorators.OneShot()
    def update(self):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Generate Python code for the given task."},
                {"role": "user", "content": f"{blackboard.get('task/description')} \nContext: {blackboard.get('task/context')}"}
            ]
        )
        generated_code = response.choices[0].message.content
        blackboard.set("code/generated", generated_code)
        with open("generated_code.py", "w") as f:
            f.write(generated_code)
        return py_trees.common.Status.SUCCESS

class RunTests(py_trees.behaviour.Behaviour):
    @py_trees.decorators.Timeout(duration=30.0)
    def update(self):
        result = os.system("pytest generated_code.py")
        test_status = "Success" if result == 0 else "Failure"
        blackboard.set("validation/test_results", test_status)
        return py_trees.common.Status.SUCCESS if result == 0 else py_trees.common.Status.FAILURE

class ValidateResults(py_trees.behaviour.Behaviour):
    def update(self):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Validate if the generated code meets requirements."},
                {"role": "user", "content": f"Does this code solve the task? {blackboard.get('code/generated')} \nContext: {blackboard.get('task/context')}"}
            ]
        )
        validation = response.choices[0].message.content
        return py_trees.common.Status.SUCCESS if "valid" in validation.lower() else py_trees.common.Status.FAILURE

class Recovery(py_trees.behaviour.Behaviour):
    @py_trees.decorators.Retry(until=3)
    def update(self):
        blackboard.set("code/generated", "")
        return py_trees.common.Status.SUCCESS

# Behavior Tree Structure
root = py_trees.composites.Sequence("CodingAgent")
clarify = ClarifyTask("ClarifyTask")
search = SearchFiles("SearchFiles")
generate = GenerateCode("GenerateCode")
execute = py_trees.composites.Parallel("ExecuteAndValidate", policy=py_trees.common.ParallelPolicy.SuccessOnAll())
test = RunTests("RunTests")
validate = ValidateResults("ValidateResults")
recover = Recovery("Recovery")

execute.add_children([test, validate])
root.add_children([clarify, search, generate, execute, recover])

tree = py_trees.trees.BehaviourTree(root)

def run_agent():
    print(py_trees.display.unicode_tree(root))
    while True:
        tree.tick()
        if blackboard.get("validation/test_results") == "Success":
            print("Code successfully generated and validated.")
            break
        print("Retrying...")

if __name__ == "__main__":
    run_agent()
