# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Preference
모든 응답은 한국어로 작성해주세요. (Please respond in Korean for all interactions.)

## Project Overview

This is a deep learning project for autonomous driving that processes ROS2 bag files (.db3) and implements reinforcement learning models, specifically MuZero for decision-making.
Conda 환경을 사용해야하며, conda_init 실행 후 conda activate pytorch_2_2 로 설정 가능

## Common Commands

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Extraction from ROS2 Bags
```bash
# Extract data from .db3 files in data/db3_parser/db3/
python data/db3_parser/db3_extractor.py
```

### Model Training
```bash
# Run MuZero training
python muzero.py

# Run with debug mode (includes visualization)
python muzero.py --debug
```

### Monitoring Training
```bash
# View training progress in TensorBoard
tensorboard --logdir models/MuZero/results
```

## Architecture Overview

### Data Processing Pipeline
The `data/db3_parser/` module extracts data from ROS2 bag files:
- **Input**: ROS2 .db3 files containing obstacle tracking, camera images, and odometry data
- **Output**: JSONL files with matched input-output pairs and extracted images
- **Key Topics**: `/apollo/agent_0/detected_object_array` (obstacles), `/carla/ego_vehicle/rgb_front/image` (camera), `/carla/ego_vehicle/odometry` (position)

### MuZero Reinforcement Learning
The `models/MuZero/` implements the MuZero algorithm for autonomous driving:
- **Action Space**: 16 actions (8 positions × 2 stop/go states)
- **Components**:
  - `environment.py`: Driving simulation environment
  - `mcts.py`: Monte Carlo Tree Search for action selection
  - `network.py`: Neural network with representation, dynamics, and prediction functions
  - `self_play.py`: Generates training data through self-play
  - `replay_buffer.py`: Stores and samples game experiences
  - `train.py`: Training loop implementation
- **Results**: Saved in `models/MuZero/results/` with timestamps

### SSR Models (Templates)
The `models/SSR/` contains template implementations for Structured State Representation models that need to be completed.

## Key Implementation Details

1. **MuZero Network**: Uses fully connected layers by default, configured in `models/MuZero/config.py`
2. **Training Data**: Generated through self-play, stored in replay buffer
3. **Observation Shape**: Configured as (10, 12) representing the driving environment state
4. **Time Matching**: Data extraction uses 100ms time windows to match inputs with ground truth

## Development Notes

- The project uses Korean documentation extensively - refer to README files for additional context
- Entry points defined in setup.py are not fully implemented (dl-extract, dl-train, dl-infer)
- SSR models are skeleton implementations requiring completion
- No formal testing framework or linting configuration exists yet