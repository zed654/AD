# Cursor 환경 정보

## 환경 지속성 규칙

### 1. 유지되는 것
- `/workspace` 디렉토리의 모든 파일
- Git으로 커밋된 변경사항
- Cursor 프로젝트 설정

### 2. 초기화되는 것
- 시스템 패키지 (apt install)
- Python 패키지 (pip install)
- 시스템 설정 변경사항
- 임시 파일 및 캐시

### 3. 세션 수명
- **활성 세션**: 작업 중일 때 유지
- **비활성 시간**: 약 30분~1시간 후 종료 가능
- **재시작**: 새로운 깨끗한 환경 생성

## 환경 복원 방법

```bash
# 1. 환경 설정 스크립트 실행
chmod +x setup_environment.sh
./setup_environment.sh

# 2. 또는 수동 설치
pip3 install --break-system-packages -r requirements.txt
```

## 현재 설치된 주요 패키지

### Python 패키지
- torch: 2.7.1
- numpy: 2.3.1
- tensorboard: 2.19.0
- matplotlib: 3.10.3
- pandas: 2.3.0
- scipy: 1.16.0

### 시스템 정보
- OS: Ubuntu (Linux 6.8.0-1024-aws)
- Python: 3.13
- 환경: Docker Container
- 호스트: AWS EC2

## 백업 권장사항

1. **GitHub 사용**
   ```bash
   git add .
   git commit -m "환경 설정 백업"
   git push
   ```

2. **중요 파일 목록**
   - `/workspace/dl-project/` (전체 프로젝트)
   - `requirements.txt` (패키지 목록)
   - `setup_environment.sh` (환경 설정)
   - 학습된 모델 파일 (.pth)

3. **로컬 백업**
   - Cursor에서 Download 기능 사용
   - 또는 tar로 압축 후 다운로드