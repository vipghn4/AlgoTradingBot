# Code Smells & Clean Code Checklist

Use this checklist during code reviews and development to ensure the repository remains clean, maintainable, and aligned with Clean Code principles.

---

## 1. Core Philosophy (The Boy Scout Rule)
*   [ ] **Leave the playground cleaner than you found it:** Every time you touch a file, make at least one minor improvement (fix a typo, rename a variable, break up a long function).
*   [ ] **Readability First:** Code should read like well-written prose. If a function is hard to follow, refactor it.

---

## 2. Meaningful Names
*   [ ] **Intent-Revealing Names:** Do variable, function, and class names clearly describe their purpose and contents? (e.g., `days_elapsed` instead of `d`).
*   [ ] **No Magic Values:** Are hardcoded constants replaced with named variables or configuration parameters?
*   [ ] **Uniform Vocabulary:** Is the same word used for the same concept across the codebase? (e.g., don't mix `retrieve`, `fetch`, and `get` for the same action).
*   [ ] **Pronounceable & Searchable Names:** Can you easily discuss the name in a review, and can you search for it in the IDE?

---

## 3. Functions
*   [ ] **Small Size:** Are functions short (ideally under 20 lines) and focused on a single task?
*   [ ] **Do One Thing (Single Responsibility):** Does the function have only one reason to change, and only one level of abstraction?
*   [ ] **Low Argument Count:** Does the function have 0 (niladic) or 1 (monadic) arguments? Avoid functions with 3+ arguments (triadic) unless absolutely necessary. Never use flag arguments (boolean parameters that control execution flow).
*   [ ] **No Side Effects:** Does the function do *only* what its name implies without silently modifying global state or unrelated variables?

---

## 4. Comments
*   [ ] **Don't Comment Bad Code—Rewrite It:** Is the comment explaining *what* the code does because the code is confusing? If so, refactor the code.
*   [ ] **Explain the "Why", Not the "What":** Does the comment document business decisions, design constraints, or complex algorithms that aren't self-explanatory?
*   [ ] **No Commented-Out Code:** Is there dead code commented out? Delete it immediately (Git tracks history).
*   [ ] **No Redundant Comments:** Are there comments that simply restate the function signature? (e.g., `# This function adds two numbers` right above `def add(a, b):`).

---

## 5. Formatting (The Newspaper Metaphor)
*   [ ] **Vertical Structure:** Do high-level public functions appear at the top of the file, followed by mid-level helper functions, and low-level details at the bottom?
*   [ ] **Vertical Density:** Are related lines of code kept close together vertically, while blank lines separate distinct conceptual blocks?
*   [ ] **Horizontal Alignment:** Are line lengths reasonable (typically < 120 characters) and indentation consistent?

---

## 6. Objects vs. Data Structures
*   [ ] **Clear Separation:** Are pure data containers (e.g., Dataclasses or TypedDicts with no behavior) separated from behavioral objects (classes with methods that manipulate state)?
*   [ ] **Law of Demeter (Don't Talk to Strangers):** Does an object avoid calling methods on objects returned by other methods? (e.g., prefer `order.get_price()` over `order.get_product().get_pricing_detail().get_price()`).

---

## 7. Error Handling
*   [ ] **Use Exceptions, Not Return Codes:** Does the code raise clean exceptions instead of returning error flags or status codes?
*   [ ] **Don't Return or Pass Null/None:** Are functions designed to avoid returning or accepting `None` as a valid state where value is expected? (Use empty collections or raise exceptions instead).
*   [ ] **Define Clean Exception Classes:** Are standard Python exceptions (`ValueError`, `KeyError`) or custom domain exceptions used appropriately?

---

## 8. Unit Tests (F.I.R.S.T.)
*   [ ] **Fast:** Do tests run in milliseconds so developers are encouraged to run them constantly?
*   [ ] **Independent:** Can tests run in any order without sharing state or depending on previous test cases?
*   [ ] **Repeatable:** Do tests produce the same results in any environment (local, CI/CD, sandboxed) without relying on external network calls or database states?
*   [ ] **Self-Validating:** Do tests have clear boolean output (Pass/Fail) rather than requiring visual inspection of logs?
*   [ ] **Timely:** Are tests written alongside or immediately after coding the implementation?
*   [ ] **Arrange-Act-Assert (AAA) Structure:**
    *   *Arrange*: Set up inputs, mock dependencies, and prepare the state.
    *   *Act*: Invoke the target method/function.
    *   *Assert*: Verify the output and state changes.
