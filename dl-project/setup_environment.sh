#!/bin/bash
# Cursor 환경 자동 설정 스크립트

echo "=== Cursor 환경 설정 시작 ==="

# 1. 시스템 패키지 업데이트
echo "1. 시스템 업데이트..."
sudo apt update

# 2. 필요한 시스템 패키지 설치
echo "2. 시스템 패키지 설치..."
sudo apt install -y software-properties-common

# 3. Python 패키지 설치
echo "3. Python 패키지 설치..."
pip3 install --break-system-packages -r requirements.txt

# 4. ROS2 설치 (선택사항)
# echo "4. ROS2 Humble 설치..."
# sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
# echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
# sudo apt update
# sudo apt install -y ros-humble-desktop

echo "=== 환경 설정 완료 ==="
echo "실행: python3 muzero.py"