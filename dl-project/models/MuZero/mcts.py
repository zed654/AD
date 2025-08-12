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
    
    def expand(self, network_output):
        """네트워크 출력을 사용해 노드 확장"""
        # policy_logits를 확률로 변환
        if isinstance(network_output['policy_logits'], torch.Tensor):
            priors = torch.softmax(network_output['policy_logits'], dim=-1).cpu().numpy()
        else:
            priors = network_output['policy_logits']
            
        # 각 액션에 대해 자식 노드 생성
        for action, prior in enumerate(priors):
            self.children[action] = Node(prior)
            
    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float):
        """루트 노드에 탐색 노이즈 추가"""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        
        for action, n in zip(actions, noise):
            self.children[action].prior = (
                self.children[action].prior * (1 - exploration_fraction) +
                n * exploration_fraction
            )
    
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
                 discount: float = 0.997,
                 c1: float = 1.25,
                 c2: float = 19652,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.network = network
        self.action_space_size = action_space_size
        self.num_simulations = num_simulations
        self.discount = discount
        
        # PUCT 하이퍼파라미터
        self.c1 = c1  # 탐색 상수
        self.c2 = c2  # 탐색 상수
        
        self.device = device
        self.min_max_stats = MinMaxStats()
        
    def run(self, observation: torch.Tensor, add_exploration_noise: bool = True, debug: bool = False) -> Dict:
        """Self-play를 위한 MCTS 실행
        
        Args:
            observation: 현재 관측값
            add_exploration_noise: 루트 노드에 탐색 노이즈 추가 여부
            debug: 시뮬레이션 진행 상황을 출력하는 옵션
            
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
        root.expand(network_output)
        
        # 루트 노드에 탐색 노이즈 추가
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=0.3,
                exploration_fraction=0.25
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
                current_action, node = self._select_child(node) # PUCT 통해 Action 선택
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
        action = self._select_action(root, temperature=1.0) # 시뮬레이션 종료 후 방문 횟수에 따른 액션 선택
        # action = 6  # Test 용 - 최종 선택도 2로 강제 FIXME: remove this line
        policy_targets = self._get_policy_targets(root)
        
        # 트리의 전체 노드 개수 (디버깅용)
        total_nodes = root.count_nodes()
        
        return {
            'action': action,
            'root_value': root.value(),
            'policy_targets': policy_targets,
            'root_node': root,
            'total_nodes': total_nodes  # 트리의 전체 노드 개수
        }
        
    def _select_child(self, node: Node) -> Tuple[int, Node]:
        """PUCT를 사용해 자식 노드 선택"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = self._puct_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
        
    def _puct_score(self, parent: Node, child: Node) -> float:
        """PUCT (Predictor + UCT) 점수 계산
        
        MuZero 논문: Q(s,a) = r(s,a) + γ * V(s')
        여기서 r(s,a)는 즉각적 보상, V(s')는 다음 상태의 가치
        """
        pb_c = math.log((parent.visit_count + self.c2 + 1) / self.c2) + self.c1
        # parent.visit_count가 0일 때도 탐색이 가능하도록 최소값 1 보장
        pb_c *= math.sqrt(max(parent.visit_count, 1)) / (child.visit_count + 1)
        
        prior_score = pb_c * child.prior
        
        if child.visit_count > 0:
            # Q(s,a) = reward + γ * V(next)
            # child.reward: s에서 a를 취했을 때의 즉각적 보상
            # child.value(): 자식 노드들의 평균 가치 (다음 상태의 가치)
            q_sa = child.reward + self.discount * child.value()
            value_score = self.min_max_stats.normalize(q_sa)
        else:
            value_score = 0
            
        return prior_score + value_score
        
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
        for node in reversed(search_path):  # Leaf -> root 순서
            node.value_sum += value         # W(s, a) 업데이트
            node.visit_count += 1           # N(s, a) 업데이트
            
            # Q(s,a) = r + γV를 min-max 통계에 반영
            # 현재 전파 중인 value를 사용 (평균이 아닌 실제 백업 값)
            q_value = node.reward + self.discount * value
            
            # Min-max 통계 업데이트
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
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
            
        return action
        
    def _get_policy_targets(self, root: Node) -> np.ndarray:
        """정책 타겟 계산 (방문 횟수 분포)"""
        policy = np.zeros(self.action_space_size)
        
        for action, child in root.children.items():
            policy[action] = child.visit_count
            
        # 정규화 (분모가 0인 경우 균등 분포 반환)
        total_visits = np.sum(policy)
        if total_visits > 0:
            policy = policy / total_visits
        else:
            # 방문이 없는 경우 균등 분포
            policy = np.ones(self.action_space_size) / self.action_space_size
        return policy
        
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
            
        # Support를 실제 값으로 변환 (간단한 구현)
        # MuZero는 categorical representation을 사용
        # support_size = 10이면 범위는 [-10, -9, ..., 0, ..., 9, 10] (21개)
        support_size = (self.network.full_support_size - 1) // 2
        support = torch.arange(
            -support_size,
            support_size + 1,
            device=value.device,
            dtype=torch.float32
        )
        
        # value와 reward를 스칼라 값으로 변환
        # Categorical distribution: logits -> probabilities -> scalar
        if value.shape[1] == len(support):
            value_probs = torch.softmax(value[0], dim=0)
            value_scalar = torch.sum(value_probs * support).item()
        else:
            # Support 크기가 맞지 않으면 0 반환
            value_scalar = 0.0
        
        if reward.shape[1] == len(support):
            reward_probs = torch.softmax(reward[0], dim=0)
            reward_scalar = torch.sum(reward_probs * support).item()
        else:
            reward_scalar = 0.0
        
        return {
            'value': value_scalar,
            'reward': reward_scalar,
            'policy_logits': policy_logits[0],
            'hidden_state': next_hidden_state[0]
        } 