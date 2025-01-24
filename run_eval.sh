#!/usr/bin/env sh
VIDEOS=videos
SEED=5

# Record png files with figures and rendered movie
time python sac_continuous_action.py \
  --policy models/LunarLanderContinuous-v2__sac_continuous_action3__1__1737488051.pt \
  --capture_video \
  --n-eval-episodes 1 \
  --seed $SEED
