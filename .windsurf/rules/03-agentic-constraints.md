---
trigger: manual
---

## Code Structure & Modularity
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
- **Use clear, consistent imports** (prefer relative imports within packages).
- Always look for **existing code** to iterate on instead of creating new code.
- Do not drastically change the patterns before trying to iterate on existing patterns.
- Always prefer simple solutions
- Avoid duplication of code whenever possible, which means checking for other areas of the codebase that might already have similar code and functionality
- When fixing an issue or bug, do not introduce a new pattern or technology without first exhausting all options for the existing implementation. And if you finally do this, make sure to remove the old implementation afterwards so we don't have duplicate logic.
- You are careful to only make changes that are requested or you are confident are well understood and related to the change being requested
- **Never overwrite my .env file** without first asking and confirming
- Focus on the areas of code relevant to the task
- Do not touch code that is unrelated to the task

## AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from [[02_Tasks]]
- Python will be the primary language
- **Use `pydantic` for data validation**.
- Write **docstrings for every function** using the Google style:
  ```python
  def example():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```

## Task Completion
- Assert all tests pass prior to marking a task as completed
- Mark completed tasks in [[02_Tasks]] immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to [[02_Tasks]] under a “Discovered During Work” section.
- Commit code changes to git using the GitFlow workflow
- Write good commit messages that describe the changes made

## Documentation & Explainability
- Update [[00_Overview]] when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

## Code Execution
- Always change to the project root working directory (e.g. /home/tj/02_Windsurf_Projects/r03_Gimbal_Angle_Root/) prior to code execution
- Assert that you are using the project virtual environment (e.g. venv) to execute python or install dependencies. Warn me if not activated
- Assert that each script has header documentation that describes how to manually run the script including venv activation
