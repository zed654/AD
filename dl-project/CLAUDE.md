# CLAUDE.md - MuZero 자율주행 프로젝트 가이드

## 🎯 프로젝트 개요
ROS2 bag 파일에서 추출한 데이터로 MuZero 강화학습 모델을 학습시켜 자율주행 의사결정을 수행하는 프로젝트
추후 ROS2 Topic 으로 환경정보를 받아와 실시간 추론

## 📁 핵심 구조
```
main_muzero.py          → 학습 진입점
models/MuZero/
  ├── config.py         → 모든 하이퍼파라미터 설정
  ├── expert_data_loader.py → GT 데이터를 Episode로 변환
  ├── self_play.py      → MCTS로 policy/value 계산 (enrich_episode_with_mcts)
  ├── train.py          → 네트워크 학습 루프
  └── models.py         → FiLM+ConvGRU dynamics, DirectionalGating representation
data/db3_parser/
  ├── rosetta.py        → 모든 class_id, 색상, grid value 정의 (중앙집중)
  └── results/          → 학습용 JSONL 파일들 (GT 데이터)
```

## 🚀 주요 명령어
```bash
# 환경 설정
conda_init && conda activate chp_env

# 학습 실행
/home/chp/anaconda3/envs/chp_env/bin/python main_muzero.py train

# Episode 시각화 (GUI 환경 필요)
./run_viz.sh --num-episodes 2
```

## 🔧 핵심 설정 (config.py)
- **Observation**: (6, 99, 99) - [동적객체, 정적환경, semantic, 근거리방향, 목적지방향, ego위치]
- **Action Space**: 9개 위치 이동 (현재위치 + 8방향)
- **Grid**: 99x99 (19.8m×19.8m, 0.2m/cell), 중앙 (49,49)이 ego 시작점
- **학습 방식**: GT observation에서 시작하는 MCTS self-play (순수 self-play)

## 📊 데이터 플로우
1. **GT 로드**: `ExpertDataLoader`가 JSONL에서 초기 observation 추출
2. **Self-play**: MCTS가 action 선택하며 Episode 진행 (policy/value/reward 계산)
3. **학습**: Replay buffer에서 배치 샘플링 → 네트워크 업데이트

## ⚠️ 중요 규칙
- **좌표계**: 전방=row↑, 우측=col↑ (180도 회전된 뷰)
- **충돌 방지**: 합법 액션 마스킹으로 충돌 회피 (calculate_legal_mask 함수)
- **데이터 정의**: 모든 매핑은 `rosetta.py` 참조 (하드코딩 금지)
- **시각화 수정 금지**: `visualize_muzero_episode.py`는 데이터 그대로 표시

## 🐛 알려진 이슈
- Ctrl+C 중단 시 multiprocessing 오류 → PyTorch 로드 후 중단하거나 `killall python`

## 📝 최근 작업 (2025-11-05)
- multiprocessing 종료 처리 개선 시도 (미완성)