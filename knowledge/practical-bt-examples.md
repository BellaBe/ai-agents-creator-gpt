# Practical Examples of Behavior Trees Using Py_Trees

## Overview

This document provides a set of practical examples that combine many components of the py_trees library. The examples recall concepts and demos such as:
- **Behavior Lifecycle / Counter** (py-trees-demo-behaviour-lifecycle)
- **Blackboard, Namespaces & Remappings** (py-trees-demo-blackboard, blackboard-namespaces, blackboard-remappings)
- **Context Switching** (py-trees-demo-context-switching)
- **Either-Or Decision Making** (py-trees-demo-either-or)
- **Eternal Guard** (py-trees-demo-eternal-guard)
- **Logging & Tree Stewardship** (py-trees-demo-logging, tree-stewardship)
- **Selector & Sequence** (py-trees-demo-selector, py-trees-demo-sequence)
- **Pick Up Where You Left Off** (py-trees-demo-pick-up-where-you-left-off)

## Example 1: Basic Lifecycle & Counter Behavior

```python
import py_trees
import time

class CounterBehavior(py_trees.behaviour.Behaviour):
    def __init__(self, name="Counter"):
        super(CounterBehavior, self).__init__(name)
        self.counter = 0

    def initialise(self):
        self.logger.debug("%s.initialise()" % self.name)
        self.counter = 0

    def update(self):
        self.counter += 1
        if self.counter < 3:
            self.feedback_message = f"Counting: {self.counter}"
            return py_trees.common.Status.RUNNING
        else:
            self.feedback_message = f"Done: {self.counter}"
            return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        self.logger.debug("%s.terminate()[%s -> %s]" % (self.name, self.status, new_status))

if __name__ == "__main__":
    root = py_trees.composites.Sequence(name="LifecycleSequence", memory=True)
    counter = CounterBehavior()
    root.add_child(counter)

    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(timeout=15)

    for i in range(1, 5):
        tree.tick()
        print(py_trees.display.unicode_tree(root, show_status=True))
        time.sleep(1)
```

## Example 2: Blackboard Integration with Namespaces & Remappings

```python
import py_trees

# Initialize a global blackboard
blackboard = py_trees.blackboard.Blackboard()
blackboard.register_key(key="conversation_history", access=py_trees.common.Access.WRITE)
blackboard.register_key(key="/parameters/default_goal", access=py_trees.common.Access.WRITE)

# Set values
blackboard.conversation_history = "User asked about travel recommendations."
blackboard.parameters.default_goal = "Plan a weekend getaway."

class BlackboardReader(py_trees.behaviour.Behaviour):
    def __init__(self, name="BB_Reader"):
        super(BlackboardReader, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name="Reader", namespace="parameters")
        self.blackboard.register_key(key="default_goal", access=py_trees.common.Access.READ)

    def update(self):
        goal = self.blackboard.default_goal
        self.feedback_message = f"Goal: {goal}"
        return py_trees.common.Status.SUCCESS

if __name__ == "__main__":
    reader = BlackboardReader()
    tree = py_trees.trees.BehaviourTree(reader)
    tree.setup(timeout=15)
    tree.tick_once()
    print(py_trees.display.unicode_blackboard())
```

## Example 3: Context Switching Behavior Tree

```python
import py_trees
import time

class ContextSwitch(py_trees.behaviour.Behaviour):
    def __init__(self, name="ContextSwitch"):
        super(ContextSwitch, self).__init__(name)
        self.backup_context = None

    def initialise(self):
        self.backup_context = "Default Context"
        self.feedback_message = "Switched to new context"
        self.logger.debug(f"{self.name} set new context.")

    def update(self):
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.feedback_message = "Restored context"
        self.logger.debug(f"{self.name} restored context from {self.backup_context}.")

class WorkAction(py_trees.behaviour.Behaviour):
    def __init__(self, name="WorkAction"):
        super(WorkAction, self).__init__(name)
        self.counter = 0

    def update(self):
        self.counter += 1
        self.feedback_message = f"Working... step {self.counter}"
        if self.counter >= 2:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

# Build the tree
context_switch = ContextSwitch()
work_sequence = py_trees.composites.Sequence(name="WorkSequence", memory=True)
for i in range(2):
    work_sequence.add_child(WorkAction(name=f"Task {i+1}"))

parallel = py_trees.composites.Parallel(name="ParallelWithContext", 
                                          policy=py_trees.common.ParallelPolicy.SuccessOnAll())
parallel.add_children([context_switch, work_sequence])

if __name__ == "__main__":
    tree = py_trees.trees.BehaviourTree(parallel)
    tree.setup(timeout=15)
    for i in range(4):
        tree.tick_once()
        print(py_trees.display.unicode_tree(parallel, show_status=True))
        time.sleep(1)
```

## Examples 4-8

```python
import operator
import py_trees

# Create two simple tasks
task_one = py_trees.behaviours.TickCounter(name="Task One", duration=2)
task_two = py_trees.behaviours.TickCounter(name="Task Two", duration=2)

# Use the either_or idiom from py_trees.idioms
either_or = py_trees.idioms.either_or(
    name="EitherOrDecision",
    conditions=[
        py_trees.common.ComparisonExpression("joystick_one", "enabled", operator.eq),
        py_trees.common.ComparisonExpression("joystick_two", "enabled", operator.eq)
    ],
    subtrees=[task_one, task_two],
    namespace="either_or"
)

if __name__ == "__main__":
    tree = py_trees.trees.BehaviourTree(either_or)
    tree.setup(timeout=15)
    for i in range(5):
        tree.tick_once()
        print(py_trees.display.unicode_tree(either_or, show_status=True))
```

```python
import py_trees

# Define two condition behaviors that simulate alternating outcomes
condition_one = py_trees.behaviours.StatusQueue(
    name="Condition One",
    queue=[py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE, py_trees.common.Status.SUCCESS],
    eventually=py_trees.common.Status.SUCCESS
)
condition_two = py_trees.behaviours.StatusQueue(
    name="Condition Two",
    queue=[py_trees.common.Status.SUCCESS, py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE],
    eventually=py_trees.common.Status.SUCCESS
)

# Define a task sequence that should run only if both conditions are met
task_sequence = py_trees.composites.Sequence(name="TaskSequence", memory=True)
task_sequence.add_child(py_trees.behaviours.Success(name="Worker 1"))
task_sequence.add_child(py_trees.behaviours.Running(name="Worker 2"))

# Build the eternal guard tree
eternal_guard = py_trees.composites.Sequence(name="EternalGuard")
eternal_guard.add_children([condition_one, condition_two, task_sequence])

if __name__ == "__main__":
    tree = py_trees.trees.BehaviourTree(eternal_guard)
    tree.setup(timeout=15)
    for i in range(4):
        tree.tick_once()
        print(py_trees.display.unicode_tree(eternal_guard, show_status=True))
```

```python
import py_trees
import time
import functools

def pre_tick_handler(tree):
    print("\n--- Pre-Tick: Tick count =", tree.count, "---\n")

def post_tick_handler(snapshot, tree):
    print("\n--- Post-Tick Tree Snapshot ---")
    print(py_trees.display.unicode_tree(tree.root, show_status=True))

# Create a simple tree
root = py_trees.composites.Sequence(name="StewardshipSequence", memory=True)
root.add_child(py_trees.behaviours.SuccessEveryN(name="EveryN", n=3))
root.add_child(py_trees.behaviours.Periodic(name="Periodic", n=2))
root.add_child(py_trees.behaviours.Success(name="Finisher"))

tree = py_trees.trees.BehaviourTree(root)
tree.add_pre_tick_handler(pre_tick_handler)

snapshot_visitor = py_trees.visitors.SnapshotVisitor()
tree.visitors.append(snapshot_visitor)
tree.add_post_tick_handler(functools.partial(post_tick_handler, snapshot_visitor))

if __name__ == "__main__":
    tree.setup(timeout=15)
    for i in range(6):
        tree.tick()
        time.sleep(0.5)
```

```python
import py_trees
import py_trees.idioms

# Define two tasks with simulated completion delays
task_one = py_trees.behaviours.StatusQueue(
    name="Task 1",
    queue=[py_trees.common.Status.RUNNING, py_trees.common.Status.SUCCESS],
    eventually=py_trees.common.Status.SUCCESS
)
task_two = py_trees.behaviours.StatusQueue(
    name="Task 2",
    queue=[py_trees.common.Status.RUNNING, py_trees.common.Status.RUNNING, py_trees.common.Status.SUCCESS],
    eventually=py_trees.common.Status.SUCCESS
)

# Use the pick up where you left off idiom
pickup = py_trees.idioms.pick_up_where_you_left_off(
    name="ResumeTasks", tasks=[task_one, task_two]
)

if __name__ == "__main__":
    tree = py_trees.trees.BehaviourTree(pickup)
    tree.setup(timeout=15)
    for i in range(10):
        tree.tick_once()
        print(py_trees.display.unicode_tree(pickup, show_status=True))
```

```python
import py_trees
import time

# Define a selector with two children:
# The first child fails initially, allowing the selector to choose the second child.
child_one = py_trees.behaviours.StatusQueue(
    name="FFS",
    queue=[py_trees.common.Status.FAILURE, py_trees.common.Status.FAILURE, py_trees.common.Status.SUCCESS],
    eventually=py_trees.common.Status.SUCCESS
)
child_two = py_trees.behaviours.Running(name="AlwaysRunning")

selector = py_trees.composites.Selector(name="MainSelector", memory=False)
selector.add_children([child_one, child_two])

if __name__ == "__main__":
    tree = py_trees.trees.BehaviourTree(selector)
    tree.setup(timeout=15)
    for i in range(3):
        tree.tick_once()
        print(py_trees.display.unicode_tree(selector, show_status=True))
        time.sleep(1)
```

## Conclusion

These examples combine:

- Core behaviors (initialise, update, terminate)
- Composite patterns (Sequence, Selector, Parallel)
- Decorator patterns and idioms
- Blackboard integration
- Context switching, logging, and stewardship

They are drawn from multiple demos in the py_trees library and provide a practical foundation for building robust, modular AI agents with behavior trees.