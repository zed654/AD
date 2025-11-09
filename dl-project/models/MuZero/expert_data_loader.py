"""
전문가 데이터(JSON)에서 Episode를 생성하는 모듈
GT 데이터를 학습용 Episode로 변환
"""
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from .replay_buffer import Episode
from .data_parser import GTDataParser
from .observation_provider import ObservationBuilder, ObservationProvider
from .action import Action


class ExpertDataLoader:
    """JSON GT 데이터에서 학습용 Episode 생성"""
    
    def __init__(self, config, observation_provider: Optional['ObservationProvider'] = None):
        """
        Args:
            config: MuZero 설정
            observation_provider: GT 데이터 제공자 (선택적)
        """
        self.config = config
        # self.discount = config.discount
        
        # ObservationProvider에서 parser와 데이터 가져오기
        if observation_provider and hasattr(observation_provider, 'get_parser'):
            self.parser = observation_provider.get_parser()
        else:
            # ObservationProvider가 없으면 직접 생성
            self.parser = GTDataParser(config.gt_data_path)
            
        self.obs_builder = ObservationBuilder(config)
        
        if self.parser:
            print(f"전문가 데이터 로드: {self.parser.get_frame_count()}개 프레임")
        else:
            print("경고: GT 데이터 파서를 찾을 수 없습니다")
    
    def create_episodes_from_json(self, 
                                 num_episodes: int = 10,
                                 episode_step_length: int = 50,
                                 episode_overlap_steps: int = 25) -> List[Episode]:
        """
        JSON 데이터에서 Episode 생성
        
        Args:
            num_episodes: 생성할 에피소드 수
            episode_step_length: 각 에피소드 길이
            episode_overlap_steps: 에피소드 간 겹치는 프레임 수
        
        Returns:
            Episode 리스트
        """
        if not self.parser:
            print("경고: GT 데이터가 없어 Episode를 생성할 수 없습니다")
            return []
            
        episodes = []
        total_frames = self.parser.get_frame_count()

        # 동적 시작점 관리 (효율적인 중복 스캔 방지)
        current_start_idx = 0
        ep_idx = 0

        while ep_idx < num_episodes and current_start_idx < total_frames - episode_step_length:
            print(f"Expert Episode {ep_idx + 1}/{num_episodes} 생성 중... (start_idx={current_start_idx})")

            episode, last_frame_idx = self._create_single_episode(current_start_idx, episode_step_length, ep_idx)

            # Episode 품질 검증 - 마지막 jsonl 의 Frame 만 누락됨
            if episode and self._validate_episode_quality(episode, episode_step_length):
                # 데이터 증강: 0°, 90°, 180°, 270° 회전
                for rot_idx in range(4):
                    if rot_idx or self.config.rotation_augmentation == False:
                        episodes.append(episode)  # 원본 (0°)
                    else:
                        # 원본 episode를 각 각도로 회전 (누적이 아님!)
                        rotated_episode = self._rotate_episode(episode, angle=rot_idx * 90)
                        episodes.append(rotated_episode)
                # episodes.append(episode)
                print(f"    ✓ Episode {len(episodes)} 추가: {len(episode.states)}개 스텝")
                # 다음 Episode는 마지막 사용 프레임에서 episode_overlap_steps 만큼 뺀 곳부터 시작
                current_start_idx = max(last_frame_idx - episode_overlap_steps, current_start_idx + 1)
            else:
                print(f"    ✗ Episode {ep_idx + 1} 제외: 품질 기준 미달")
                # Episode 생성 실패시 조금만 앞으로 이동 (데드락 방지)
                current_start_idx += episode_step_length // 4

            ep_idx += 1
        
        print(f"\n생성된 Episode: {len(episodes)}개")
        return episodes
    
    def _create_single_episode(self, start_idx: int, episode_step_length: int, ep_idx: int = 0) -> Tuple[Optional[Episode], int]:
        """단일 Episode 생성"""
        # 출처 정보 설정 - 시작 프레임의 실제 파일명 사용 . # episode_step_length: Episode 최대 스텝 수
        source_file = None
        start_frame_data = self.parser.get_frame_data(start_idx)
        if start_frame_data and '_source_file' in start_frame_data:
            source_file = start_frame_data['_source_file']
        elif hasattr(self.parser, 'filepath'):
            if self.parser.filepath.is_file():
                source_file = self.parser.filepath.name
            else:
                source_file = f"folder_{self.parser.filepath.name}"
        
        episode = Episode(
            source_file=source_file,
            start_frame=start_idx,
            end_frame=start_idx + episode_step_length - 1
        )
        
        # 상태 추적
        self.episode_start_ego = None  # Episode 시작 위치
        episode_near_direction = None  # Episode 전체에서 사용할 근거리 방향 (첫 스텝에서 고정)
        episode_final_direction = None  # Episode 전체에서 사용할 목적지 방향 (첫 스텝에서 고정)
        observations = []
        expert_ego_positions = []  # GT에서 추출한 전문가 ego 위치 (그리드 좌표)
        
        # 이동 기반 Episode 생성: 실제 이동하는 episode_step_length 개 스텝 수집
        collected_steps = 0
        frame_idx = start_idx
        prev_grid_pos = None
        skip_count = 0  # 스킵한 프레임 수

        # Episode.state 의 step 수집 루프 (Self-play 의 정보 빠진 상태)
        while collected_steps < episode_step_length and frame_idx < self.parser.get_frame_count() - 1:
            # 현재 프레임 데이터
            frame_data = self.parser.get_frame_data(frame_idx)
            if not frame_data:
                frame_idx += 1
                continue

            # Observation 생성
            ego_info = self.parser.parse_ego_info(frame_data)
            objects = self.parser.parse_objects(frame_data)
            ldm_points = self.parser.parse_ldm_points(frame_data)  # LDM 데이터 파싱
            near_destination_direction, final_destination_direction = self.parser.get_direction_vectors(frame_data)

            if not ego_info:
                frame_idx += 1
                continue

            # Episode 시작 위치 설정 (첫 프레임에서만)
            # 목적은, episode.state 의 첫 step의 위치를 기준으로 나머지 step 들을 변환시키기 위함.
            if collected_steps == 0:
                self.episode_start_ego = ego_info
                prev_grid_pos = None  # 첫 스텝은 무조건 포함
            
            # Grid[0]: 동적 장애물 (/obstacles_tracking_msg)
            # Grid[1]: 정적 환경 (/ldm_msg 의 GRID_VALUE_CLIFF, GRID_VALUE_STATIC_OBSTACLE, GRID_VALUE_MISSING_POINTS)
            # 좌표들은 전역좌표계(Odomgyro)이며, episode_start_ego 를 49, 49 위치 기준으로 변환함.
            observation = self.obs_builder.build_observation(
                ego_info, objects, ldm_points, near_destination_direction, final_destination_direction,
                episode_start_ego=self.episode_start_ego
            )

            # observation에서 필요한 정보 추출
            grid = observation['grid']  # 3채널 그리드 (3, 99, 99)
            vectors = observation['vectors']  # 방향 벡터 [near_x, near_y, dest_x, dest_y]

            # 현재 그리드 위치 계산 (ego 중심 좌표계를 그리드 좌표로 변환)
            current_grid_pos = self._global_to_grid_coords(
                ego_info.x, ego_info.y,
                self.episode_start_ego.x, self.episode_start_ego.y, self.episode_start_ego.yaw
            )

            # 제자리 이동 체크: 이전 위치와 동일하면 스킵
            if prev_grid_pos is not None and current_grid_pos == prev_grid_pos:
                skip_count += 1
                frame_idx += 1
                continue  # 제자리 이동은 스킵하고 다음 프레임으로

            # 첫 스텝에서만 방향 벡터를 저장하고, 이후 스텝에서는 재사용
            if collected_steps == 0 and vectors is not None and len(vectors) == 4:
                # 첫 스텝에서 Episode 전체에서 사용할 방향 저장
                if not (vectors[0] == 0 and vectors[1] == 0):  # near direction이 유효한 경우
                    episode_near_direction = {
                        'x': round(float(vectors[0]), 2), 
                        'y': round(float(vectors[1]), 2)
                    }
                if not (vectors[2] == 0 and vectors[3] == 0):  # destination direction이 유효한 경우
                    episode_final_direction = {
                        'x': round(float(vectors[2]), 2), 
                        'y': round(float(vectors[3]), 2)
                    }
            
            # observation은 dict 형태로 저장
            state_dict = {
                'obstacles': grid[0],  # Channel 0: 동적 장애물 (/obstacles_tracking_msg, 현재는 동적/정적 섞여있음. 정적은 ldm_msg 의 static 을 의미하는게 아님.)
                'static_environment': grid[1],  # Channel 1: 정적 환경 (/ldm_msg 의 Cliff, Static Obstacles, Missing Points)
                'ground_class_id': grid[2],  # Channel 2: 지면 semantic class ID (/ldm_msg 의 Semantic Segmentation Class ID)
                'near_destination_direction': episode_near_direction,  # 근거리 방향 (Episode 첫 스텝에서 고정)
                'final_destination_direction': episode_final_direction,  # 목적지 방향 (Episode 첫 스텝에서 고정)
                'simulated_ego_pos': (49, 49)  # 시뮬레이션된 ego 위치 (그리드 좌표, GT 데이터이므로 항상 중앙)
            }
            
            observations.append(state_dict)  # 전체 state dict 저장

            # 현재 그리드 위치를 expert_ego_positions에 추가
            expert_ego_positions.append(current_grid_pos)

            # 이전 위치 업데이트 (다음 스텝 비교용)
            prev_grid_pos = current_grid_pos

            # 수집된 스텝 수 증가
            collected_steps += 1

            # 다음 프레임으로 진행
            frame_idx += 1

        # 수집 완료 로그
        print(f"    - 수집 완료: {collected_steps}개 스텝 (스킵: {skip_count}개 프레임)")
        
        # Episode 구성 검증
        if len(expert_ego_positions) == 0 or len(observations) == 0:
            print(f"    - Episode 생성 실패: 유효한 데이터 없음")
            return None, frame_idx

        # Episode 생성 (states와 expert_ego_pos 길이를 맞춤)
        for i in range(len(observations)):
            # 디버깅 포인트 2: Episode step 추가 (필요시 주석 해제)
            # if i < 3:  # 처음 3개 step만
            #     print(f"\n--- Step {i} ---")
            #     print(f"Expert Action: x={expert_actions[i][0]:.2f}, y={expert_actions[i][1]:.2f}")
            
            # state dict를 Episode에 추가 (obstacles와 static_environment를 tensor로 변환)
            state_with_tensor = observations[i].copy() # 얘는 얕은복사여도 문제없음. observations 에 영향 안줌.
            state_with_tensor['obstacles'] = torch.tensor(observations[i]['obstacles'], dtype=torch.float32)
            
            # static_environment가 있으면 tensor로 변환
            if 'static_environment' in observations[i] and isinstance(observations[i]['static_environment'], np.ndarray):
                state_with_tensor['static_environment'] = torch.tensor(observations[i]['static_environment'], dtype=torch.float32)
            
            episode.add_step(
                state=state_with_tensor,  # dict 형태의 state
                action=0,           # self-play action은 0으로 초기화 (POS0: 현재 위치)
                expert_ego_pos=expert_ego_positions[i],  # GT에서 추출한 전문가 ego 위치 (그리드 좌표)
                reward=0.0,         # GT에서는 항상 0
                pi=None,        # MCTS/self-play에서 계산됨 (Action 개수만큼 있어야함)
                value=0.0,          # 0으로 초기화
                done=(i == len(expert_ego_positions) - 1),
                legal_mask=None  # 합법 액션 마스크 ... 추후 Self-play 단계에서 기입됨. 현재는 빈 값 사용
            )

        return episode, frame_idx

    def _rotate_episode(self, episode: Episode, angle: float) -> Episode:
        """Episode 전체를 회전시켜 data augmentation 수행

        Args:
            episode: 회전시킬 Episode
            angle: 회전 각도 (도 단위, 90, 180, 270 등)

        Returns:
            회전된 Episode
        """
        import scipy.ndimage

        center_row = center_col = 49  # 그리드 중심
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotated_states = []
        for i in range(len(episode.states)):
            original_state = episode.states[i]
            # 원본을 수정하지 않도록 새 딕셔너리 생성
            rotated_state = {}

            # 1. 그리드 데이터 회전 (obstacles, static_environment, ground_class_id)
            for key in ['obstacles', 'static_environment', 'ground_class_id']:
                if key in original_state and original_state[key] is not None:
                    if isinstance(original_state[key], torch.Tensor):
                        # Tensor를 numpy로 변환하여 회전
                        grid_np = original_state[key].numpy()
                        rotated = scipy.ndimage.rotate(grid_np, angle, reshape=False, order=0)
                        rotated_state[key] = torch.tensor(rotated, dtype=original_state[key].dtype)
                    elif isinstance(original_state[key], np.ndarray):
                        rotated_state[key] = scipy.ndimage.rotate(original_state[key], angle, reshape=False, order=0)

            # 2. 방향 벡터 회전
            for dir_key in ['near_destination_direction', 'final_destination_direction']:
                if dir_key in original_state and original_state[dir_key] is not None:
                    x = original_state[dir_key].get('x', 0)
                    y = original_state[dir_key].get('y', 0)
                    # 벡터 회전 변환
                    new_x = x * cos_a - y * sin_a
                    new_y = x * sin_a + y * cos_a
                    rotated_state[dir_key] = {'x': round(new_x, 2), 'y': round(new_y, 2)}

            # 3. simulated_ego_pos는 GT에서 항상 (49, 49)이므로 변경 불필요
            rotated_state['simulated_ego_pos'] = (49, 49)  # 유지

            # 나머지 키들도 복사 (참조 복사 방지, deepcopy 없이)
            for key in original_state:
                if key not in rotated_state:
                    value = original_state[key]
                    # 딕셔너리는 수동으로 얕은 복사
                    if isinstance(value, dict):
                        rotated_state[key] = {k: v for k, v in value.items()}
                    # 리스트는 얕은 복사
                    elif isinstance(value, list):
                        rotated_state[key] = value[:]
                    else:
                        rotated_state[key] = value

            rotated_states.append(rotated_state)

        # 새로운 Episode 생성
        rotated_episode = Episode()
        rotated_episode.states = rotated_states

        # expert_ego_pos 회전 (그리드 좌표)
        rotated_ego_pos = []
        for i, pos in enumerate(episode.expert_ego_pos):
            if pos is not None:
                row, col = pos
                # 중심 기준 상대 좌표
                rel_row = row - center_row
                rel_col = col - center_col
                # 회전 변환
                new_rel_row = rel_row * cos_a - rel_col * sin_a
                new_rel_col = rel_row * sin_a + rel_col * cos_a
                # 절대 좌표로 변환
                new_row = int(center_row + new_rel_row)
                new_col = int(center_col + new_rel_col)
                rotated_ego_pos.append((new_row, new_col))
            else:
                rotated_ego_pos.append(None)

        rotated_episode.expert_ego_pos = rotated_ego_pos

        # # 나머지 Episode 속성들 복사 (중첩 리스트 처리)
        # rotated_episode.actions = episode.actions[:]  # 얕은 복사 (숫자 리스트라 OK)

        # # pi가 중첩 리스트일 수 있음 [[0.1, 0.2], [0.3, 0.4]]
        # if episode.pi and episode.pi[0] is not None and isinstance(episode.pi[0], list):
        #     rotated_episode.pi = [p[:] if p is not None else None for p in episode.pi]  # 각 내부 리스트도 복사
        # else:
        #     rotated_episode.pi = episode.pi[:]

        # rotated_episode.values = episode.values[:]  # 숫자 리스트
        # rotated_episode.rewards = episode.rewards[:]  # 숫자 리스트

        # # target_values는 Episode 클래스에 없으므로 체크하고 처리
        # if hasattr(episode, 'target_values'):
        #     rotated_episode.target_values = episode.target_values[:] if episode.target_values else None

        # rotated_episode.dones = episode.dones[:]  # bool 리스트

        # # legal_mask도 복사
        # if hasattr(episode, 'legal_mask'):
        #     rotated_episode.legal_mask = episode.legal_mask[:]

        # Episode 출처 정보 복사
        rotated_episode.source_file = episode.source_file
        rotated_episode.start_frame = episode.start_frame
        rotated_episode.end_frame = episode.end_frame

        return rotated_episode

    def _global_to_grid_coords(self, global_x: float, global_y: float, 
                              ego_start_x: float, ego_start_y: float, ego_start_yaw: float) -> Tuple[int, int]:
        """전역 좌표를 그리드 좌표로 변환
        
        Args:
            global_x, global_y: 변환할 전역 좌표
            ego_start_x, ego_start_y, ego_start_yaw: Episode 시작점의 ego 정보
            
        Returns:
            (grid_row, grid_col): 그리드 좌표
        """
        # Episode 시작점 기준 상대 좌표로 변환
        dx = global_x - ego_start_x
        dy = global_y - ego_start_y
        
        # 자차 좌표계로 회전
        cos_yaw = np.cos(-ego_start_yaw)
        sin_yaw = np.sin(-ego_start_yaw)
        
        rel_x = dx * cos_yaw - dy * sin_yaw
        rel_y = dx * sin_yaw + dy * cos_yaw
        
        # 그리드 좌표로 변환 (중앙이 49, 49)
        # 주의: rel_x는 전방(forward)이고, rel_y는 좌우(left/right)
        # Grid에서 row는 x축(전방으로 큰 값), col은 y축(우측으로 큰 값)
        grid_center = self.config.observation_shape[1] // 2  # 49
        grid_row = grid_center + int(rel_x / self.config.cell_size)  # 상단으로 +
        grid_col = grid_center + int(rel_y / self.config.cell_size)  # 우측으로 +
        
        return grid_row, grid_col
    
    def _parse_ldm_to_grid(self, ldm_points: List, ego_info: Any, episode_start_ego: Any) -> Dict[str, np.ndarray]:
        """LDM 포인트를 그리드 좌표로 변환하고 obstacles_type별로 분리
        
        Args:
            ldm_points: LDM 포인트 리스트
            ego_info: 현재 프레임의 ego 정보
            episode_start_ego: Episode 시작점의 ego 정보
            
        Returns:
            obstacles_type별로 분리된 그리드 딕셔너리
            - 'cliff': 절벽/경사 (obstacles_type=1)
            - 'static_obstacles': 정적 장애물 (obstacles_type=2)
            - 'missing_points': 측정 불가 영역 (obstacles_type=3)
        """
        # 그리드 크기 (obstacles와 동일)
        grid_shape = (self.config.observation_shape[1], self.config.observation_shape[2])  # (99, 99)
        
        # 각 타입별 그리드 초기화
        static_env = {
            'cliff': np.zeros(grid_shape, dtype=np.float32),
            'static_obstacles': np.zeros(grid_shape, dtype=np.float32),
            'missing_points': np.zeros(grid_shape, dtype=np.float32)
        }
        
        if not ldm_points or not episode_start_ego:
            return static_env
        
        # 각 LDM 포인트를 그리드 좌표로 변환
        for ldm in ldm_points:
            # Episode 시작점 기준 그리드 좌표로 변환
            grid_row, grid_col = self._global_to_grid_coords(
                ldm.x, ldm.y,
                episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
            )
            
            # 그리드 범위 체크
            if 0 <= grid_row < grid_shape[0] and 0 <= grid_col < grid_shape[1]:
                # obstacles_type에 따라 적절한 그리드에 표시
                if ldm.obstacles_type == 1:  # cliff
                    static_env['cliff'][grid_row, grid_col] = 1.0
                elif ldm.obstacles_type == 2:  # static_obstacles
                    static_env['static_obstacles'][grid_row, grid_col] = 1.0
                elif ldm.obstacles_type == 3:  # missing_points
                    static_env['missing_points'][grid_row, grid_col] = 1.0
                # obstacles_type이 0 또는 기타 값인 경우는 무시
        
        return static_env

    def _validate_episode_quality(self, episode: 'Episode', target_length: int) -> bool:
        """Episode 품질 검증

        Args:
            episode: 생성된 Episode
            target_length: 목표 스텝 길이

        Returns:
            bool: 품질 기준 통과 여부
        """
        if not episode or not hasattr(episode, 'states') or not hasattr(episode, 'expert_ego_pos'):
            return False

        states_count = len(episode.states)
        positions_count = len(episode.expert_ego_pos)

        # 1. 최소 길이 검증 (목표의 80% 이상)
        min_length = max(target_length * 0.8, target_length - 10)
        if states_count < min_length:
            print(f"      길이 부족: {states_count} < {min_length}")
            return False

        # 2. 데이터 일관성 검증
        if states_count != positions_count:
            print(f"      데이터 불일치: states={states_count}, positions={positions_count}")
            return False

        # 3. 실제 이동 거리 검증
        if positions_count >= 2:
            total_distance = 0.0
            for i in range(1, positions_count):
                prev_pos = episode.expert_ego_pos[i-1]
                curr_pos = episode.expert_ego_pos[i]
                if isinstance(prev_pos, tuple) and isinstance(curr_pos, tuple):
                    distance = ((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)**0.5
                    total_distance += distance

            # 최소 이동 거리 (5 cells = 1m)
            min_distance = 5.0
            if total_distance < min_distance:
                print(f"      이동 거리 부족: {total_distance:.1f} < {min_distance}")
                return False

            print(f"      품질 통과: {states_count}스텝, 이동거리={total_distance:.1f}cells")

        return True
