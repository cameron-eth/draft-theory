# Draft Theory

Draft Theory studies how prospect information, draft cost, and positional value interact to determine which draft decisions create the most surplus NFL talent.

## Core Idea

The NFL Draft is not just about finding good players. It is about finding the best expected return relative to acquisition cost.

Every prospect profile contains information:

- combine testing
- size and athletic traits
- college production
- college level / context
- age, experience, and role

Once that profile is paired with draft capital, the player has an implied expected career outcome. In this project, that expected outcome is primarily framed through career value, using metrics like `AV` / `wAV`, and eventually position-specific success definitions.

That creates the central question:

**Given a player's pre-draft profile and draft capital, what should we expect his NFL outcome distribution to be, and which players or archetypes historically outperform that expectation the most?**

## What This Project Will Study

This project aims to surface a few core truths:

1. The actual importance of the combine by itself.
2. The importance of draft capital by itself.
3. The importance of the combine in tandem with draft capital.
4. The importance of the full pre-draft prospect profile: combine, college production, college level, age, and role.
5. Which players, positions, and archetypes historically overperform their expected value the most once draft capital is accounted for.
6. Which draft strategies create the best expected value by position.

## Main Research Lens

The project is built around three layers:

- `Signal`: what information actually predicts NFL outcomes?
- `Price`: how does the league price that information through draft capital?
- `Strategy`: how should teams exploit the gap between signal and price?

This means the project is not just about forecasting career success. It is about forecasting:

- expected career value
- range of outcomes
- probability of success
- probability of beating expectation
- positional surplus value at a given draft cost

## Prospect Outcome Framework

Every prospect, once assigned a draft slot, should have:

- a predicted career `AV/EV`
- a predicted range of outcomes
- a probability of success
- a probability of overperforming his draft-implied expectation

Success should not be treated as one simple binary. It should account for:

- peak performance
- sustained performance
- time to quality performance
- career duration
- availability / injury drag

The most important target is not just raw value, but **surplus value**:

`Actual Career Value - Expected Career Value given Draft Capital`

That lets us ask better questions:

- Who beats their slot the most?
- Which archetypes are underpriced?
- Which positions return the best value in each round?
- Where does the market appear efficient, and where does it misprice talent?

## Draft Theory Questions

The project is ultimately trying to answer questions like:

- How much does draft capital explain on its own?
- How much does the combine add beyond draft capital?
- Which parts of the pre-draft profile matter most by position?
- Where is the draft most and least predictable?
- Which positions are best purchased early?
- Which positions are better targeted later?
- When is trading down the better EV move?
- When is trading up justified?
- What kinds of prospects most often beat expectation?

For example:

- Is it better, from an EV / AV lens, to target `WR` in Round 2 rather than `QB`?
- When should teams draft `TE`?
- Are `RB`s more efficiently acquired later than the market assumes?
- Which positions show the best median return, and which show the best ceiling return?

## Positional Focus

The main strategic positions for this project are:

- `WR`
- `QB`
- `TE`
- `RB`

These positions matter because they differ sharply in:

- development curve
- variance
- ceiling outcomes
- replacement value
- market price

The goal is not just to rank prospects, but to understand the best **cost-adjusted talent acquisition strategy by position**.

## Blog Vision

This project will lead to a long-form blog built in stages.

### 1. Glossary and framing

Define the key language needed to think clearly:

- combine
- draft capital
- expected value
- surplus value
- hit rate
- median vs mean outcome
- variance
- positional value
- literal investment vs figurative investment

### 2. The combine and the pre-draft profile

Explain what the combine actually does and does not tell us, and how physical traits should be interpreted alongside:

- college production
- college competition level
- role
- age / experience

### 3. The draft as a market

Show how draft capital acts as a market price for prospects, and how much NFL value is already embedded in where a player is selected.

### 4. Draft theory

Translate all of the above into practical conclusions for:

- GMs
- scouts
- coaches
- analysts

This final stage will focus on draft decisions from an EV lens:

- pick allocation
- trading picks
- surplus value by position
- when to buy ceiling
- when to buy median outcomes
- which positions outperform average value as rounds progress

## Historical Overperformers

An important part of the project is not just predicting outcomes, but identifying which players historically overperformed expectation the most.

That includes:

- biggest positive residuals vs draft-implied value
- biggest position-level surplus bets
- archetypes that repeatedly beat market expectations
- players who became stars relative to modest draft cost

This is important because those cases help define what market inefficiency looks like in practice.

## 2026 Draft Use Case

The final application layer of the project will be prospect-facing.

For the `2026 NFL Draft`, the project will aim to identify:

- which prospects carry the most risk
- which prospects appear underpriced
- which players have the strongest expected surplus value
- which players are overpriced relative to likely return

The idea is to move from historical analysis to an actionable forward-looking framework.

## Current Direction

The current repo already explores:

- combine signal vs career value
- draft capital vs career value
- excess value relative to draft expectation
- post-draft early-career predictability
- peak outcome tiers
- year-1 threshold analysis

The next step is to unify these into one coherent research pipeline:

1. canonical dataset
2. canonical target definitions
3. position-specific models
4. surplus-value analysis by round and position
5. a final strategy layer for draft decision-making

## Project Structure

The repo is now organized by function instead of keeping everything at the top level:

- `analysis/`: standalone exploratory and research scripts
- `data/raw/`: raw input data files such as the combine CSV
- `draft_theory/`: reusable package code for ingestion, matching, and dataset building
- `figures/`: generated charts and image outputs
- `notes/`: planning and research scaffolding
- `reports/`: text outputs and deep-dive writeups

This keeps the project split between:

- raw data
- generated artifacts
- one-off analysis
- reusable pipeline code

## Data Pipeline Setup

The repo now includes a first-pass ingestion package in `draft_theory/` for:

- secure CFBD authentication
- college stat pulls
- team talent context
- combine-to-college prospect matching
- canonical prospect dataset construction

### Dependencies

Install the current project requirements:

```bash
pip install -r requirements.txt
```

### CFBD authentication

Do **not** hardcode the CFBD token in source files.

Set it locally instead:

```bash
export CFBD_API_KEY="your-rotated-token-here"
```

The API key should be rotated if it has ever been pasted into chat, logs, or committed anywhere.

### Current package entry points

- `draft_theory.cfbd_client.build_cfbd_apis()`
- `draft_theory.prospect_pipeline.build_prospect_dataset()`

### Current matching approach

The first-pass matching pipeline is intentionally conservative and scores candidates using:

- normalized player name
- school
- position family
- expected college season window relative to draft year

This is good enough for a first canonical dataset, but it should be audited before any final model training.
