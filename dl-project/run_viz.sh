#!/bin/bash
# MuZero Episode 시각화 실행 스크립트
# 시각화할 Episode 개수 정의
#     ./run_viz.sh --num-episodes <개수>

# Conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate chp_env

# 프로젝트 루트로 이동
cd /home/chp/ai_ws/AD/dl-project

# matplotlib 백엔드 및 DISPLAY 설정
export MPLBACKEND=Qt5Agg
#export DISPLAY=:0
export DISPLAY=:10.0

# 실행
python visualize_muzero_episode.py "$@"
