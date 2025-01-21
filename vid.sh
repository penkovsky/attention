#!/usr/bin/env sh
VIDEOS=videos
SEED=5

# Record png files with figures and rendered movie
time python sac_continuous_action.py \
  --policy models/LunarLanderContinuous-v2__sac_continuous_action3__1__1737488051.pt \
  --capture_video \
  --n-eval-episodes 1 \
  --seed $SEED

# Latest created folder
shopt -s nullglob
latest=""
for d in $VIDEOS/*/; do
    [[ -d $d ]] && [[ $d -nt "$latest" ]] && latest=$d
done
echo "$latest"

# Concert figures to video
ffmpeg -r 30 -i $VIDEOS/0_%03d.png -c:v libx264 -pix_fmt yuv420p $latest/graphs.mp4 && rm $VIDEOS/0_*.png

# Concatenate final video
ffmpeg -i "$latest/rl-video-episode-0.mp4" -i $latest/graphs.mp4 \
  -filter_complex "[0:v][1:v]vstack=inputs=2[v]" \
  -map "[v]" -c:v libx264 $latest/final_vertical_video.mp4
