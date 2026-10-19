# Agent Guidelines

## Human collaboration

### Understand the request before editing

- Read the relevant code, tests, documentation, and repository instructions before proposing changes. Resolve questions from the codebase when possible.
- Distinguish requests for explanation or review from requests to implement changes. Do not edit the codebase for an advice-only request.
- If an uncertainty could materially affect the goal, scope, correctness, public behavior, or cost of the work, ask before making dependent edits.
- When asking, explain the decision briefly, offer likely options with their tradeoffs, and allow a free-form response. Recommend an option when justified.
- Group related questions together. Do not ask the user to decide routine implementation details that existing conventions already settle.
- For minor, reversible choices, use the project's conventions and state any assumption that matters to the result.
- While awaiting clarification, read-only investigation may continue; do not make edits that depend on the unresolved decision.

### Correct misunderstandings and discuss alternatives

- Be candid about incorrect premises, conflicting requirements, and tradeoffs. Ground corrections in evidence and distinguish confirmed facts from hypotheses.
- If a misunderstanding materially changes whether or how the requested work should be done, explain it and pause dependent edits until the user responds.
- If an alternative would materially better achieve the user's goal, describe the benefit, cost, and change in scope before implementing it. Wait for agreement before switching plans.

### Communicate clearly

- For substantial work, briefly state the intended approach and how you will check the result.
- Provide concise updates when a significant finding, blocker, or change of plan arises.
- Explain limitations directly. Do not invent requirements, APIs, results, or successful checks.
- At completion, summarize what changed, what was checked, and any remaining issues or decisions.

## Scope and protection of existing work

- Keep changes focused. Avoid unrelated refactors, formatting churn, dependency upgrades, and speculative features.
- Inspect the working tree before editing. Preserve changes made by the user or other agents; do not overwrite or revert work you did not create.
- If concurrent changes conflict with your work, stop and resolve the conflict rather than silently replacing them.
- Do not delete data, rewrite Git history, publish, deploy, or start substantial paid work without authorization covering that action.
- Do not expose secrets or sensitive data in code, logs, examples, or reports.
- Report unrelated issues separately unless they block the requested work.

## Design and implementation

- Prefer simple, readable solutions and follow the project's existing architecture and conventions.
- Use clear, descriptive names. Keep functions and modules cohesive.
- Reuse existing helpers where appropriate. Extract repeated or independently useful logic without introducing abstractions for hypothetical future needs.
- Fix the underlying cause when practical. If a workaround is necessary, explain its limits.
- Preserve public behavior unless a change is intentional and documented.
- Validate inputs at system boundaries and handle expected failures explicitly.
- Do not silently skip required work, coerce invalid inputs, or return success-shaped defaults after failures.
- Make optional behavior and fallback behavior explicit and documented.
- Keep secrets, environment-specific paths, and deployment configuration outside source code. Named constants for genuine invariants and documented defaults are fine.
- Prefer existing dependencies or the standard library. Justify new dependencies and avoid unrelated lockfile changes.

## Verification

- Add or update tests for meaningful behavior changes, including relevant edge cases and failure paths. For bug fixes, add a regression test when practical.
- Test observable behavior rather than mirroring implementation details. Keep tests deterministic, independent, and understandable.
- For randomized code, use controlled random generators and test invariants; avoid flaky assertions about individual random outcomes.
- Run the relevant tests, formatter, linter, and type checker using the project's configured commands. Start with focused checks and broaden when the change warrants it.
- Run `uv run pyright` whenever Python code or type-checking configuration changes.
- Do not weaken tests, disable checks, or add unjustified type suppressions to make a change pass.
- Distinguish failures caused by your changes from existing failures. If a check cannot run, report the command and reason.
- Review the final diff for unintended changes, debug output, secrets, and generated artifacts.

## Python

- Follow established formatting, import, typing, and test conventions.
- Use type hints for new or changed function signatures and non-obvious data structures, consistent with the project.
- Treat type checking as complementary to runtime validation; annotations alone do not validate external data.
- Narrow optional values explicitly. If a required value is `None`, raise a useful exception rather than silently skipping work.
- Raise specific exceptions with actionable messages: for example, `ValueError` for an invalid value and `TypeError` for an inappropriate type. Raise exception objects rather than returning them unless the API explicitly defines an error-as-data convention.
- Do not broadly catch exceptions unless there is a clear recovery, cleanup, or reporting policy. Preserve the original cause when wrapping exceptions.
- Use context managers for resources that require cleanup.
- Avoid mutable default arguments and unintended mutation of caller-owned data.

## Documentation

- Write descriptive docstrings for new or changed functions. Explain non-obvious inputs, outputs, side effects, constraints, and exceptions; avoid merely restating the signature.
- Use comments to explain reasoning and constraints rather than narrating obvious code.
- Update documentation and examples when behavior, configuration, interfaces, or usage changes.
- Keep `README.md` current with a concise description of each file in `scripts/`, unless the user specifies a different convention.
- Document any intentional compatibility break and the migration steps it requires.

## Project-specific instructions
- Never leave the codebase in a state where `git commit` would expose a credential. In particular, `.env` must be in the gitignore and API keys, tokens, etc. should not appear anywhere else in the codebase.

### Remote compute access

- For authorized work on remote compute, use the `grader-power-gpu` host alias in the ignored `configs/ssh_config.local`. From the repository root: `ssh -F configs/ssh_config.local grader-power-gpu`. This file is local connection configuration, not experiment configuration.
- If that file is absent, use `configs/ssh_config.example` as the setup reference and obtain the user's connection details. Do not guess an endpoint or assume a particular GPU provider.
- Keep real host addresses, ports, usernames, and identity-file paths in the ignored local file; do not copy them into tracked files, experiment logs, or reports. Private keys stay outside the repository and must never be read or copied by an agent.
- Preserve SSH host-key verification. Connection settings do not authorize provisioning, deployment, substantial paid work, or execution of model-generated code outside the required verified sandbox.
