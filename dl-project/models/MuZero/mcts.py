"""
Monte Carlo Tree Search (MCTS) 구현
MuZero의 핵심 계획 알고리즘
"""
import torch
import numpy as np
import math
from typing import List, Optional, Dict, Tuple
from .network import AbstractNetwork


class MinMaxStats:
    """Min-Max 정규화를 위한 통계 클래스"""
    
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')
        
    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
        
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    """MCTS 트리의 노드"""
    
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        
    def expanded(self) -> bool:
        """노드가 확장되었는지 확인"""
        return len(self.children) > 0
    
    def value(self) -> float:
        """평균 가치 반환"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, network_output, legal_mask=None):
        """네트워크 출력을 사용해 노드 확장

        Args:
            network_output: 네트워크 출력 (policy_logits, value 등)
            legal_mask: 합법 액션 마스크 (None이면 모든 액션 합법, Root node에만 적용)
        """
        # policy_logits를 확률로 변환
        logits = network_output['policy_logits'].clone()
        # Legal mask 적용 (Root node only) ... MCTS Sim 과정에서는 Expand() 에 lagal_mask 적용하지 않으므로, 합법 마스킹 적용 안됨
        if legal_mask is not None:
            for action_idx in range(len(legal_mask)):
                if legal_mask[action_idx] == 0.0:  # 불법 action
                    logits[action_idx] = -1e10  # 매우 작은 값으로 설정
        if isinstance(logits, torch.Tensor):
            priors = torch.softmax(logits, dim=-1).cpu().numpy()
        else:
            priors = logits
        
        # 각 액션에 대해 자식 노드 생성
        for action, prior in enumerate(priors):
            self.children[action] = Node(prior)
            
    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float, legal_mask=None):
        """루트 노드에 탐색 노이즈 추가"""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions)) # [alpha, alpha, ..., alpha] ... actions 개
        
        for action, n in zip(actions, noise):
            self.children[action].prior = (
                self.children[action].prior * (1 - exploration_fraction) +
                n * exploration_fraction
            )
        
        # Noise 추가 후 legal mask 재적용 (noise로 인해 불법 action의 prior가 0이 아니게 될 수 있음)
        if legal_mask is not None:
            for action_idx in range(len(legal_mask)):
                if legal_mask[action_idx] == 0.0:  # 불법 action
                    if action_idx in self.children:
                        self.children[action_idx].prior = 0.0

    
    def count_nodes(self) -> int:
        """현재 노드를 포함한 서브트리의 전체 노드 개수 카운트"""
        count = 1  # 자기 자신
        for child in self.children.values():
            count += child.count_nodes()
        return count


class MCTS:
    """Monte Carlo Tree Search 구현"""

    def __init__(self,
                 network: 'AbstractNetwork',
                 action_space_size: int,
                 num_simulations: int,
                 mcts_value_backup_discount: float = 0.99,
                 puct_discount: float = 0.997,
                 c1: float = 1.25,
                 c2: float = 19652,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 config=None):

        self.network = network
        self.action_space_size = action_space_size
        self.num_simulations = num_simulations
        self.mcts_value_backup_discount = mcts_value_backup_discount
        self.puct_discount = puct_discount
        self.config = config  # Config 저장

        # PUCT 하이퍼파라미터
        self.c1 = c1  # 탐색 상수
        self.c2 = c2  # 탐색 상수

        self.device = device
        self.min_max_stats = MinMaxStats()

    def _h_inverse(self, x):
        """
        MuZero transform의 역함수 (스칼라 버전)
        """
        epsilon = 0.001
        sign_x = 1 if x > 0 else -1 if x < 0 else 0
        abs_x = abs(x)
        return sign_x * (
            ((math.sqrt(1 + 4 * epsilon * (abs_x + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1
        )
        
    def run(self, observation: torch.Tensor, add_exploration_noise: bool = True, debug: bool = False, legal_mask=None) -> Dict:
        """Self-play를 위한 MCTS 실행

        Args:
            observation: 현재 관측값
            add_exploration_noise: 루트 노드에 탐색 노이즈 추가 여부
            debug: 시뮬레이션 진행 상황을 출력하는 옵션
            legal_mask: 합법 액션 마스크 (9개 원소의 0/1 배열, None이면 모두 합법)

        Returns:
            선택된 액션, 루트 가치, 정책 타겟 등을 포함한 딕셔너리
        """
        # GPU로 이동
        if not observation.is_cuda and self.device == 'cuda':
            observation = observation.to(self.device)
            
        # 루트 노드 초기화
        root = Node(0)
        
        # 초기 추론으로 hidden state 얻기
        network_output = self._initial_inference(observation)
        root.hidden_state = network_output['hidden_state']
        root.expand(network_output, legal_mask)

        # 루트 노드에 탐색 노이즈 추가
        # Episode 의 prior 값에 노이즈 넣어주는 것
        # 적정 alpha = 10/|Action| = 10/9 ≈ 1.1
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.mcts_dirichlet_alpha,
                exploration_fraction=self.config.mcts_exploration_fraction,
                legal_mask=legal_mask
            )

        # Min-Max 통계 초기화
        self.min_max_stats = MinMaxStats()
        
        # 시뮬레이션 실행
        for i in range(self.num_simulations):
            if debug and i % 10 == 0:  # 10회마다 출력
                print(f"시뮬레이션 {i}/{self.num_simulations}, 현재 노드 수: {root.count_nodes()}")
                
            node = root
            search_path = [node]
            current_action = None
            
            # 리프 노드까지 선택
            while node.expanded():
                # 여기 _select_child() 에서는 temperature 쓰면 안됌. Loss 깨짐
                # Root node에만 legal_mask 전달
                is_root = (node is root)
                current_action, node = self._select_child(node, legal_mask if is_root else None) # PUCT 통해 Action 선택. is_root 가 True일 경우만 regal_mask 적용
                search_path.append(node)
            
            # 리프 노드 평가 및 확장
            # root.expand()가 이미 실행되었으므로 search_path는 최소 2개 요소를 가짐
            parent = search_path[-2] if len(search_path) >= 2 else root
            
            # 첫 시뮬레이션에서 root가 리프인 경우는 없음 (root.expand() 이미 호출됨)
            # 하지만 혹시 모를 엣지 케이스를 위해 action 체크
            if current_action is None:
                # 이 경우는 발생하면 안 됨 (root는 이미 expand됨)
                raise ValueError("MCTS: No action selected. This should not happen.")
            
            # 네트워크 한 번만 호출
            network_output = self._recurrent_inference(parent.hidden_state, current_action)
            
            # 노드 정보 업데이트
            node.hidden_state = network_output['hidden_state']
            node.reward = network_output['reward']
            value = network_output['value']
            
            # 자식 노드 생성
            node.expand(network_output)
            
            # 백업 (역전파)
            self._backup(search_path, value)
            
        if debug:
            print(f"시뮬레이션 완료! 최종 노드 수: {root.count_nodes()}")
            
        # 액션 선택 및 정책 타겟 계산
        # 나중에 실제 학습할 때는 temperature 를 iteration 끝날 시점에 0 까지 줄이도록 해야함
        # 얘는 학습에 영향 안 줌. 왜냐하면, Episode 의 다음 Step 에서 다시 MCTS Sim 돌리는데, Root 는 초기화되어있기 때문.
        action = self._select_action(root, temperature=1.0) # 시뮬레이션 종료 후 방문 횟수에 따른 액션 선택
        # action = 6  # Test 용 - 최종 선택도 2로 강제 FIXME: remove this line
        policy_targets = self._get_pi_targets(root)
        
        # 트리의 전체 노드 개수 (디버깅용)
        total_nodes = root.count_nodes()
        
        return {
            'action': action,
            'root_value': root.value(),
            'policy_targets': policy_targets,
            'root_node': root,
            'total_nodes': total_nodes,  # 트리의 전체 노드 개수
            'legal_mask': None  # 합법 액션 마스크 (추후 구현 필요)
        }
        
    def _select_child(self, node: Node, legal_mask=None) -> Tuple[int, Node]:
        """PUCT를 사용해 자식 노드 선택

        Args:
            node: 현재 노드
            legal_mask: 합법 액션 마스크 (Root node에만 전달, None이면 모든 액션 고려)
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            # Legal mask 체크 (Root node인 경우만 적용). is_root 가 True일 경우만 regal_mask 적용
            if legal_mask is not None and legal_mask[action] == 0.0:
                continue  # 불법 action은 스킵

            score = self._puct_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            # 모든 액션이 불법 - 액션 0 (제자리)를 반환하여 시뮬레이션 계속 진행
            print(f"[MCTS 경고] 모든 액션이 불법입니다. 액션 0으로 대체합니다.")
            best_action = 0
            best_child = node.children[0]

        return best_action, best_child
        
    def _puct_score(self, parent: Node, child: Node) -> float:
        """PUCT (Predictor + UCT) 점수 계산
        
            score = Q(s, a) + U(s, a)
            Q(s, a) = W(s, a)/N(s, a)
                W(s, a) 는 자식 노드의 가치 합 ... Sigma(reward + γ * V(next)) -> V(next) 인 이유는, 액션 a 를 했을 때의 기대값이기 때문
            U(s, a) = c_puct * P(s, a) * sqrt(N(s, all)) / (N(s, a) + 1)

            Q(s, a) 는 예측항 (좋은 수 반복)
            U(s, a) 는 탐색항 (안가본 수 시도)

                이 수식이 유의미한 이유는, Q(s,a) 와 U(s,a) 모두 정규화되어있기 때문.
                    Q(s,a)는 self.min_max_stats.normalize(q_sa)로 정규화되고,
                    U(s,a) 는 Policy 자체는 0~1로 정규화되어있고, 방문횟수 수식부분(N../N(a)+1) 는 방문 초기에 값이 크지만 점점 떨어지게 유도되어있음
        """
        pb_c = math.log((parent.visit_count + self.c2 + 1) / self.c2) + self.c1
        # parent.visit_count가 0일 때도 탐색이 가능하도록 최소값 1 보장
        pb_c *= math.sqrt(max(parent.visit_count, 1)) / (child.visit_count + 1)
        
        prior_score = pb_c * child.prior
        
        if child.visit_count > 0:
            # Q(s,a) = reward + γ * V(next)
            # child.reward: s에서 a를 취했을 때의 즉각적 보상
            # child.value(): 자식 노드들의 평균 가치 (다음 상태의 가치)
            q_sa = child.reward + self.config.puct_discount * child.value()
            value_score = self.min_max_stats.normalize(q_sa)
        else:
            value_score = 0
            
        return value_score + prior_score
        
    # _evaluate 함수는 더 이상 필요없음 (인라인으로 처리)
    # def _evaluate(self, node: Node, parent_hidden_state: torch.Tensor, action: int) -> float:
    #     """리프 노드 평가"""
    #     # 재귀적 추론으로 다음 상태 예측
    #     network_output = self._recurrent_inference(parent_hidden_state, action)
    #     
    #     node.hidden_state = network_output['hidden_state']
    #     node.reward = network_output['reward']
    #     
    #     # 가치 반환
    #     return network_output['value']
        
    def _backup(self, search_path: List[Node], value: float):
        """검색 경로를 따라 가치 역전파"""
        # leaf -> parents -> root 순서로 업데이트
        # 핵심은, Parent 3 이라는 부모의 자식 Leaf 1, Leaf 2 가 있을 때, Parent 3 의 Value 는 Leaf 1, Leaf 2 의 Value 가 모두 반영되어야 한다는 것.
        for node in reversed(search_path):  # Leaf -> root 순서
            node.value_sum += value         # W(s, a) 업데이트
            node.visit_count += 1           # N(s, a) 업데이트
            
            # Q(s,a) = r + γV를 min-max 통계에 반영
            # 현재 전파 중인 value를 사용 (평균이 아닌 실제 백업 값)
            q_value = node.reward + self.config.mcts_value_backup_discount * value
            
            # Min-max 통계 업데이트 ... PUCT 계산(_puct_score() 함수)에서 q_sa 정규화에 사용됨
            self.min_max_stats.update(q_value)
            
            # 다음 노드를 위한 value 계산 (할인된 가치로 업데이트)
            value = q_value
            
    def _select_action(self, root: Node, temperature: float = 1.0) -> int:
        """방문 횟수에 기반한 액션 선택"""
        visit_counts = np.array([
            child.visit_count for child in root.children.values()
        ])
        actions = list(root.children.keys())
        
        if temperature == 0:
            # 그리디 선택
            action = actions[np.argmax(visit_counts)]
        else:
            # 온도에 따른 확률적 선택
            visit_count_distribution = visit_counts ** (1 / temperature) # np.array ** k -> 각 요소에 k 제곱
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            
            # 확률 분포 기반으로 선택
            action = np.random.choice(actions, p=visit_count_distribution)
            
        return action
        
    def _get_pi_targets(self, root: Node) -> np.ndarray:
        """π(pi) 타겟 계산 (방문 횟수 분포)"""
        pi = np.zeros(self.action_space_size)

        for action, child in root.children.items():
            pi[action] = child.visit_count

        # 정규화 (분모가 0인 경우 균등 분포 반환)
        total_visits = np.sum(pi)
        if total_visits > 0:
            pi = pi / total_visits
        else:
            # 방문이 없는 경우 균등 분포
            pi = np.ones(self.action_space_size) / self.action_space_size
        return pi
        
    def _initial_inference(self, observation: torch.Tensor) -> Dict:
        """초기 추론"""
        # 배치 차원 추가 (네트워크는 배치 입력 기대)
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
            
        with torch.no_grad():
            value, reward, policy_logits, hidden_state = self.network.initial_inference(observation)
            
        return {
            'value': value[0].cpu().numpy(),
            'reward': reward[0].cpu().numpy(),
            'policy_logits': policy_logits[0],
            'hidden_state': hidden_state[0]
        }
        
    def _recurrent_inference(self, hidden_state: torch.Tensor, action: int) -> Dict:
        """재귀적 추론"""
        # 배치 차원 추가
        if len(hidden_state.shape) == 1:
            hidden_state = hidden_state.unsqueeze(0)
        
        # hidden_state를 올바른 device로 이동
        hidden_state = hidden_state.to(self.device)
            
        # 액션을 텐서로 변환
        action_tensor = torch.tensor([action], device=self.device)
        
        with torch.no_grad():
            value, reward, policy_logits, next_hidden_state = self.network.recurrent_inference(
                hidden_state, action_tensor
            )
            
        # Support를 실제 값으로 변환 (config의 categorical_distribution_min/max 사용)
        # num_categorical_distribution_atoms개의 균등 간격 support 생성
        if hasattr(self.config, 'categorical_distribution_min'):
            support = torch.linspace(
                self.config.categorical_distribution_min,
                self.config.categorical_distribution_max,
                self.network.num_categorical_distribution_atoms,
                device=value.device,
                dtype=torch.float32
            )
        else:
            # Fallback: 기존 방식
            support_size = (self.network.num_categorical_distribution_atoms - 1) // 2
            support = torch.arange(
                -support_size,
                support_size + 1,
                device=value.device,
                dtype=torch.float32
            )
        
        # # 네트워크 출력: (B, 21)
        # logits = [0.1, 0.2, ..., 0.8, ...]  # 21개

        # # Softmax 후 확률
        # probs = softmax(logits)  # [0.05, 0.1, ..., 0.3, ...]

        # # 최종 scalar value
        # value = sum(probs * [-10, -9, ..., 0, ..., 10])
        # value = 0.05*(-10) + 0.1*(-9) + ... + 0.3*(0) + ...
        # value = -2.5  # 예시
        # -> 사실상 내적임


        # value와 reward를 스칼라 값으로 변환 (transformed 값 유지)
        # Categorical distribution: logits -> probabilities -> scalar
        if value.shape[1] == len(support):
            value_probs = torch.softmax(value[0], dim=0)
            transformed_value = torch.sum(value_probs * support).item()
            # MCTS 내부에서는 transformed 값을 유지 (h_inverse 제거)
            value_scalar = transformed_value
        else:
            # Support 크기가 맞지 않으면 0 반환
            value_scalar = 0.0

        if reward.shape[1] == len(support):
            reward_probs = torch.softmax(reward[0], dim=0)
            transformed_reward = torch.sum(reward_probs * support).item()
            # MCTS 내부에서는 transformed 값을 유지 (h_inverse 제거)
            reward_scalar = transformed_reward
        else:
            reward_scalar = 0.0

        return {
            'value': value_scalar,
            'reward': reward_scalar,
            'policy_logits': policy_logits[0],
            'hidden_state': next_hidden_state[0]
        } 