## Core principles
- **Simplicity first**: Minimum code that solves the problem. Nothing speculative.
   - No features beyond what was asked.
   - No abstractions for single-use code.
   - No "flexibility" or "configurability" that wasn't requested.
   - No error handling for impossible scenarios.
   - If you write 200 lines and it could be 50, rewrite it.
   - Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
- **Think before coding**: Don't assume. Don't hide confusion. Surface tradeoffs. Before implementing:
   - State your assumptions explicitly. If uncertain, ask.
   - If multiple interpretations exist, present them - don't pick silently.
   - If a simpler approach exists, say so. Push back when warranted.
   - If something is unclear, stop. Name what's confusing. Ask.
- **No laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimat impact**: Changes should only touch what's necessary. Avoid regression.
- **Goal-driven execution**: Define success criteria. Loop until verified.
   - Transform tasks into verifiable goals:
       - "Add validation" → "Write tests for invalid inputs, then make them pass"
       - "Fix the bug" → "Write a test that reproduces it, then make it pass"
       - "Refactor X" → "Ensure tests pass before and after"
   -  For multi-step tasks, state a brief plan:


       ```
       1. [Step] → verify: [check]
       2. [Step] → verify: [check]
       3. [Step] → verify: [check]
       ```


   - Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


## Workflow Orchestration
### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity


### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution


### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project


### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness
- For data analysis requests, look at the data samples and aggregated stats to verify if it makes sense


### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it


### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how


## Task Management
1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections


## Coding & debugging guidelines
### General principles
- **Clean code & architecture**:
   - New code follow software design pattern best practices.
   - Avoid bloat classes, functions, and data types. Break the code into reusable modules.
   - Use code comment to make code changes easy to understand but do not bloat the code file with comment.
   - Use struct/enum data types instead of ad-hoc dict/shape.
   - Use global variables instead of hard-coded in-line values.
- **Inspect intermediate outputs during verification & debugging**:
   - Print/save intermediate outputs for inspection if needed.
   - Clean up the code after inspection.
- **Use bullet-point code summary and descriptive commit title**:
- **Commit size**:
   - No commit is allowed to have more than 500 LOCs (excluding test code).
   - If a coding task is too big, break it into subtasks so that each subtask corresponds to a commit of AT MOST 500 non-test LOCs.
- **Code cleanup**: Remove unused code after making updates.


### Architectural principles
#### Core architectural vision
Minimize human resources required to build and maintain the system by creating a highly decoupled, testable, and adaptable codebase.


#### The dependency rule
* **Inward Direction Only**: Source code dependencies must only point inward, toward higher-level policies (core business logic).
* **Isolation of Concerns**: Inner layers must know absolutely nothing about outer layers.


#### Layer definitions & constraints
- **Domain entities (innermost layer)**: Core business data & rules. Must never depend on use cases, adapters, databases, or frameworks. Volatility: Low.
- **Use cases**: Application-specific business rules that orchestrates dataflow between entities and adapters.
   - Must be decoupled from delivery mechanisms (web, CLI) and storage engines (DB).
- **Interface adapters**: Controller, presenter, gateway, and view model that translate data between use cases/entities and external layers (UI, DB, API).
   - Use Dependency Inversion (interfaces/ports) when an inner layer needs to trigger an action in an outer layer (e.g., Use Case calling a Database Repository).
- **Frameworks & drivers (outermost layer)**: Core technical implementation details (e.g., Web frameworks, DB, UI, SDKs, etc.). Must be treated as disposable plugins.


#### Development guidelines
- **Screaming architecture**: Structure directories by business feature (`billing/`, `users/`), never by technology type (`controllers/`, `models/`).
- **Deferred Decisions:** Keep business logic isolated so infrastructure (e.g., UI, DB, 3P integrations, etc.) can be decided or swapped late.
- **Architectural SOLID principles**:
   - *Single Responsibility (SRP):* Gather together things that change for the same reasons. Separate things that change for different reasons.
   - *Open/Closed (OCP):* Extend system behavior by adding new code, not modifying existing code, secured by clean component boundaries.
   - *Liskov Substitution (LSP):* Ensure implementations or subclasses strictly fulfill the contractual behavior of their abstractions.
   - *Interface Segregation (ISP):* Avoid forcing layers or components to depend on broad, monolithic interfaces they don't fully use.
   - *Dependency Inversion (DIP):* High-level business policies must not depend on low-level mechanism details; both must depend on abstractions.
- **Verification & testing procedure**:
   1. Cover business logic (entities & use cases) with fast, isolated, in-memory unit tests. No UI, DB, or network dependencies.
   2. Run smoke tests with test objects (e.g., test user, test object, etc.) and no mocking to verify the system works end-to-end.


### Coding principles for class & functions
#### Function principles
- **Be ridiculously small**:
   - A function should rarely be more than 20 lines long, and ideally under 10 lines.
   - Indent levels inside a function (like if, else, or while blocks) should not be greater than one or two levels deep.
- **Do one thing**: A function should do one thing. They should do it well. They should do it only.
   - If a function performs steps that are only one level of abstraction below the stated name of the function, then the function is doing one thing
   - If you can extract another function from it with a name that is not merely a restatement of its implementation, it’s doing too much.
- **One level of abstraction per function**:
   - The statements within a function need to be at the same level of abstraction.
   - Mixed levels of abstraction (e.g., combining a high-level business rule with a low-level string manipulation) confuse the reader.
   - Code should read like a top-down narrative. Every function should be followed by those at the next level of abstraction, letting you read the program as if you were reading a set of TO-paragraphs.
- **Keep arguments to a minimum**: Arguments are hard from a testing and readability perspective. My preference is:
   - 0 arguments: Ideal
   - 1 arguments: Acceptable, e.g., asking a question about the argument or transforming it
   - 2 arguments: Significantly harder to understand. Use with caution
   - 3 arguments: Avoid if possible
   - 4+ arguments: Need special justification and should almost always be wrapped into an argument object
- **No side effect**: A function should not have hidden agendas, e.g., it shouldn't secretly modify global variables, change the state of an object unexpectedly, or open system resources unless explicitly stated in its name.
- **Command-query separation**: Functions should either do something (change the state of an object) or answer something (return information about an object), but never both.


#### Class principles
- **Classes should be small**: Like functions, classes should be small. Classes are measured by responsibilities, not by LOCs.
- **Single responsibility principle**: A class should have one, and only one, reason to change.
- **High cohesion**:
   - Classes should have a small number of instance variables. Each method of the class should manipulate one or more of those variables.
   - High cohesion means that the methods and variables of the class are co-dependent and hang together as a logical whole.
- **Organize for change (open-closed principle)**: Classes should be structured so that you can add new features by extending the system, not by modifying existing code.
