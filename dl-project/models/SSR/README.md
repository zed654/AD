1. SSR_head_pt.py
구조: 원본 구조 최대한 살림.
외부 의존성: mmdet, mmcv, mmdet3d 관련 코드 싹 제거.
임베딩/브랜치: PyTorch 기본 레이어로 잘 대체.
forward/loss: 실제 논리(연산)는 비워둠(연구용 템플릿).
주석/가이드: 충청도 사투리 안내, 연구용임 명확히 표기.
문법/오타: 없음.
주의:
    실제로 동작하려면 forward/loss에 논리 채워야 함.
    TokenLearner, Latent/Way Decoder 등은 별도 구현 필요.


2. SSR_transformer_pt.py
구조: 원본 구조 최대한 반영.
외부 의존성: 싹 제거.
Transformer 구조: Dummy 레이어로 대체(실제 논리 없음).
forward: 템플릿 형태, 논리 직접 구현 필요.
문법/오타: 없음.
주의:
    실제 논리(Attention 등)는 직접 구현해야 함.
    DummyTransformerLayer는 예시일 뿐, 실제 모델로 쓰려면 수정 필요.


3. SSR_pt.py
구조: 원본 구조 최대한 반영.
외부 의존성: 없음.
forward: 템플릿 형태, 논리 직접 구현 필요.
문법/오타: 없음.
주의:
    TokenFuser 등은 tokenlearner_pt.py에서 import해서 써야 함.
    실제 동작하려면 forward에 논리 채워야 함.


4. tokenlearner_pt.py
구조: 원본과 거의 동일, PyTorch-only.
클래스: MlpBlock, TokenLearnerV11PT, TokenFuserPT 등 정상 구현.
문법/오타: 없음.
동작:
    이 파일은 바로 import해서 쓸 수 있음.
    논리도 실제 동작(MLP, Token select/fuse) 구현되어 있음.