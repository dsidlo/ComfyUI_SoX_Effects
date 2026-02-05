---
apply: always
---

# AI Rules Git

## Use ** Conventional Commit ** Message for git commit

Note source files where key logic has changed

### 1. Classic / Tim Pope Style (50/72 Rule) — Very Common Baseline
This is the most traditional "standard" you'll see referenced in Git tutorials and many large projects.

**Structure**:
```
Capitalized, short (50 chars or less) summary

More detailed explanatory text, if necessary.  Wrap it to about 72
characters or so.  In some contexts, the first line is treated as the
subject of an email and the rest of the text as the body.  The blank
line separating the summary from the body is critical (unless you omit
the body entirely).

Explain the problem that this commit is solving.  Focus on why you
are making this change as opposed to how (the code explains that).
Are there side effects or other unintuitive consequences of this
change? Here's the place to explain them.

Further paragraphs come after blank lines.

 - Bullet points are okay, too
 - Typically a hyphen or asterisk is used for the bullet, preceded
   by a single space, with blank lines in between, but conventions
   vary here

If you use an issue tracker, put references to them at the bottom,
like this:

Resolves: #123
See also: #456, #789
```

**Key Rules**:
- Subject line: **≤50 characters** (some say ≤72; 50 is safer for tools like `git log --oneline`).
- Use **imperative mood** / present tense: "Add feature X" not "Added feature X" or "Adds feature X".
- Capitalize the first letter of the subject.
- No period at the end of the subject line.
- Blank line between subject and body.
- Body wrapped at ~72 characters.
- Focus on **why** (not how — the code shows how).

### 2. Conventional Commits (Most Popular Modern Standard)
This is the **explicit specification** many teams adopt (especially in JavaScript/Node ecosystems, but increasingly everywhere). It's designed to be machine-readable for generating changelogs, bumping versions, etc.

**Official site**: https://www.conventionalcommits.org/en/v1.0.0/

**Structure**:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

- **type** (required): Common ones include:
  - `feat` — new feature
  - `fix` — bug fix
  - `docs` — documentation
  - `style` — formatting, missing semicolons, etc. (no code change)
  - `refactor` — code change that neither fixes a bug nor adds a feature
  - `perf` — performance improvement
  - `test` — adding missing tests or correcting existing tests
  - `build` — affecting build system or external dependencies
  - `ci` — CI configuration
  - `chore` — maintenance tasks
  - `revert` — reverts a previous commit

- **[optional scope]**: e.g., `(api)`, `(ui)`, `(auth)` — what part of the codebase.

- **description**: Short summary (imperative mood, present tense, no period at end).

- Body & footer: Optional detailed explanation, breaking changes, issue refs (e.g., `BREAKING CHANGE: ...`, `Closes #123`).

**Examples**:
```
feat(auth): add OAuth2 login flow

Implement Google and GitHub providers.
Add new env vars for client secrets.

Closes #45
```

```
fix(ui): prevent button flash on dark mode toggle

Resolves layout shift when theme changes.
```

```
refactor: rename UserService to AuthService

No functional changes.
```

