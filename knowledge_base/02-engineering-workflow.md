# Engineering Workflow

This is how a change goes from an idea in Jira to running in production. Following these steps keeps `main` healthy and makes code review fast.

## Branching

We use **trunk-based development**: there is one long-lived branch, `main`, and everyone merges small changes into it often (at least every few days).

Create a short-lived branch for each Jira ticket. Name it with a prefix, the ticket key and a short description:

- `feat/FLOW-123-add-retry-button`: a new feature
- `fix/FLOW-456-null-email-crash`: a bug fix
- `chore/FLOW-789-bump-fastapi`: maintenance, dependencies or tooling

```bash
# always branch from an up-to-date main
git switch main && git pull
git switch -c feat/FLOW-123-add-retry-button
```

## Commits

We follow **Conventional Commits**. Each commit message starts with a type:

- `feat:` adds something new
- `fix:` fixes a bug
- `chore:`, `docs:`, `test:`, `refactor:` cover everything else

Example: `fix: handle missing email when sending receipts`. Write the message so someone reading `git log` a year from now understands *why* the change was made.

## Pull requests

A pull request (PR) asks your teammates to review and merge your branch.

- **Keep PRs small.** Aim for under 400 changed lines. Small PRs get faster and better reviews.
- **Fill in the PR template:** what changed, why, how you tested it, and screenshots for UI changes.
- **Approvals:** every PR needs **1 approval**. PRs to the `infra` repository need **2 approvals**.
- **CI must be green.** All checks have to pass before you can merge.
- **Squash merge.** We use "Squash and merge" so each PR becomes one clean commit on `main`.
- **Delete your branch** after merging. GitHub offers a button for it.

## Code review

Reviews are how we share knowledge and catch bugs early, so they are part of everyone's job, including yours from week 3.

- **Respond within 1 working day** to review requests, even if it's only "I'll look this afternoon".
- **Be kind and specific.** Comment on the code, not the person. Explain *why* ("this query runs once per row, so it gets slow for large tenants"), and suggest an alternative.
- **Label nit-picks.** Prefix minor, optional suggestions with `nit:` so the author knows they aren't blocking.
- **As a newcomer reviewer:** you don't need to be an expert. Ask questions ("Why do we need this lock?"), check that the tests cover the change, and run the branch locally if you're unsure. Questions from fresh eyes are valuable.
- **As an author:** reply to every comment, either by fixing it or by explaining why not. Re-request review when you're done.

## CI/CD and releases

- **CI:** GitHub Actions runs linting, type checks and the full test suite on every PR.
- **Staging:** every merge to `main` **automatically deploys to staging**. Staging is a production-like environment for final checks.
- **Production:** production deploys ship with the **release train on Tuesdays and Thursdays at 14:00 IST**. Everything merged and verified on staging before the train leaves goes out together.
- **No Friday deploys:** there are no production deploys on Friday after 14:00, so nobody spends the weekend fixing a bad release.
- Need something urgently in production (a hotfix)? Ask in `#devops`. The Platform Infra team can run an extra deploy.

## Good first issues

Your first PR should be small and low-risk. In Jira, filter by the label **`good-first-issue`**. These tickets are hand-picked to touch one area of the code and have clear acceptance criteria.

Pick one with your onboarding buddy, pair on it if you like, and aim to ship it in your second week. Shipping a small fix end to end teaches you the whole workflow: branch, commit, PR, review, CI, staging, release train.
