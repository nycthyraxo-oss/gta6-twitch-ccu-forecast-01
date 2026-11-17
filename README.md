[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nycthyraxo-oss/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nycthyraxo-oss/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Launch-Day Twitch CCU Forecaster

Forecasts how many people will be watching Grand Theft Auto VI live on
Twitch in each hour of its release day (2026-11-19 UTC), from the viewer
count at the end of the day's first hour — as five quantiles per hour
(5%, 25%, 50%, 75%, 95%).

Entry for the make.fun bounty
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).
Scored by mean pinball loss on the log scale; the house persistence
baseline scores about 0.387 on the practice cases.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads one case
file per run and writes one forecast file. It is a **kernel-weighted
comparable-launch log-ratio regression**:

1. Every case ships with the full hourly Twitch curves of 38–42 past big
   game launches. For each comparable launch, its release-day log-ratio
   curve is `ln(viewers_h / viewers_1)`.
2. Donors are weighted by similarity to the target: a Gaussian kernel on
   log hour-1 size, a kernel on the pre-release surge, tier weights
   (early-access-contaminated and weak launches downweighted), a
   console-only boost, and an unlock-shape weight (rolling-midnight
   launches count most for a rolling launch like GTA VI).
3. Per hour, a weighted ridge regression of the log-ratios on log-size
   difference and the pre-checkpoint ramp gives the centre; the weighted
   residual spread (winsorised, calibrated) gives the quantiles.
   Centre and spread are smoothed across neighbouring hours.

The target's own pre-release history supplies its surge and ramp; GTA
VI's own pre-release category history is excluded as a donor (it is not
a launch). Any failure falls back to the persistence baseline, so the
output is always valid. The program is deterministic and runs in
milliseconds per case.

## Validation

Leave-one-out over the 38 comparable launches in the kit's practice
cases (each launch forecast from the other 37):

| model | mean pinball loss | median |
|---|---:|---:|
| house persistence baseline | 0.930 | 0.325 |
| this model (all launches) | 0.395 | 0.274 |
| this model, hour-1 ≥ 10k viewers | 0.255 | 0.183 |
The practice targets and GTA VI itself sit in the large-launch regime
(hour-1 23k–270k viewers for the practice targets), where the model is
strongest; the final case additionally includes the four practice
launches as donors.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Data

Launch-day curves come from SullyGnome.com's hourly Twitch category
averages, quoted with attribution; see `NOTICE.txt` in the competition
kit for the sources.
