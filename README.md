# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Mastomy2/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Mastomy2/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI (PS5) trophy's Exophase earn rate at 2026-11-26T00:00Z for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty.

## Model

`main.py` (Python 3.13, stdlib only) reads a case file and writes one forecast per trophy: five quantiles (5/25/50/75/95%) of its earn rate at the target.

- **Trend in log-time:** per trophy, a recency-weighted least-squares line of logit(earn rate) against ln(1 + hours since release). Trophy earning is fast early and saturates, so a line in log-time is a saturating S-curve in real time. Each slope is shrunk toward its tier's median slope.
- **Stale-page correction:** Exophase refreshes on its own schedule, so a case's rates are as of its last refresh, not the checkpoint. When the page sat frozen, the true tracked count kept growing while earners grew more slowly, so true rates sit below the shown ones. The model estimates the tracked-growth rate (from the history's own `players_tracked` when it moves, else a launch-night prior) and dilutes each trophy's current level accordingly, with earners catching up faster for prologue/chapter trophies than for grind trophies.
- **Domain caps:** the median at the target is capped by trophy class, read off the trophy's name/description — chapter trophies climb to tens of percent in week one, while 250 stunts, 100% completion and the platinum stay under a few percent.
- **Honest spread:** random-walk term (0.5 logit per sqrt-day) + the fit's prediction variance (floored, so flat history is never treated as certain) + a staleness term that widens while the page is frozen.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs produce byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=94ztz8n6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
