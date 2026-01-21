#!/usr/bin/env python3
"""ROS2 Topic Subscriber Sample Script"""

import rclpy
from rclpy.node import Node
import torch
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from autonomy_ros2_message.msg import DetectedObjectArray, DetectedObject, GlobalWayInfo
from models.MuZero.data_parser import LDMPoint, ObjectInfo, EgoInfo, ObservationBuilder
from models.MuZero.config import MuZeroConfig
from models.MuZero.replay_buffer import Episode
import math
from visualize_muzero_episode import EpisodeInteractiveVisualizer
import matplotlib.pyplot as plt
from data.db3_parser.db3_extractor import DB3Extractor
from models.MuZero.models import get_network
from models.MuZero.self_play import enrich_episode_with_mcts_parallel_worker
from pathlib import Path

class SampleSubscriber(Node):
    def __init__(self, config):
        super().__init__('sample_subscriber')
        self.config = config        
        self.obs_builder = ObservationBuilder(config)
        self.db3_extractor = DB3Extractor()

        # 실시간 데이터 버퍼
        self.current_ldm_points = list()
        self.current_objects = list()
        self.current_odom_ego = None
        # self.near_destination_direction = dict()
        # self.final_destination_direction = dict()

        self.near_dir = {'x': 0.0, 'y': 0.0}
        self.final_dir = {'x': 0.0, 'y': 0.0}
        
        # Episode 수집
        self.episode_start_ego = None

        self.subscription = self.create_subscription(
            String,
            '/sample_topic',
            self.sample_string_callback,
            1
        )

        self.subscription = self.create_subscription(
            DetectedObjectArray,
            '/obstacles_tracking_msg',
            self.obstacles_tracking_callback,
            1
        )

        self.subscription = self.create_subscription(
            Odometry,
            '/odomgyro',
            self.ego_odometry_callback,
            1
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            '/planned_global_way_array',
            self.global_way_info_callback,
            1
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            'ldm_msg',
            self.ldm_callback,
            1
        )

        # Predicted Path Publisher (PointCloud2)
        self.predicted_path_publisher = self.create_publisher(PointCloud2, '/nn_predicted_path', 1)

        self.get_logger().info('Subscriber node started, listening to /sample_topic')

    def sample_string_callback(self, msg: String):
        self.get_logger().info(f'Received: {msg.data}')

    def ego_odometry_callback(self, msg: Odometry):
        self.get_logger().info('Received Odometry message')
        position = msg.pose.pose.position

                # 쿼터니언에서 yaw 계산
        x, y, z, w = (msg.pose.pose.orientation.x,
                      msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z,
                      msg.pose.pose.orientation.w)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        self.current_odom_ego = EgoInfo(
            x=position.x,
            y=position.y,
            yaw=yaw,
            velocity=msg.twist.twist.linear.x
        )
        orientation = msg.pose.pose.orientation
        self.get_logger().info(f'Position: x={position.x}, y={position.y}, z={position.z}')
        self.get_logger().info(f'Orientation: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}')


    def obstacles_tracking_callback(self, msg: DetectedObjectArray):
        # TODO: data_parser.py 의 parse_objects() 함수 참고
        self.current_objects = [
            ObjectInfo(
                id=str(obj.id),
                class_id=obj.class_id,
                global_x=obj.global_center.x,
                global_y=obj.global_center.y,
                global_yaw=obj.global_yaw_heading,
                velocity=obj.global_velocity,
                global_points=[(p.x, p.y) for p in obj.global_points]  # Point32 → 튜플 변환
            )
            for obj in msg.objects
        ]

    def global_way_info_callback(self, msg: PointCloud2):
        self.get_logger().info('Received GlobalWayInfo message')
        # self.get_logger().info(f'Number of waypoints: {len(msg.waypoints)}')
        global_ways = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        print( global_ways[0]['x'], global_ways[0]['y'], global_ways[0]['z'] )
        # global_ways_dict = self.db3_extractor.extract_points_from_pointcloud2(global_ways, max_points=999999)
        global_ways_list = [{'x': p['x'], 'y': p['y'], 'z': p['z']} for p in global_ways]
        current_odom_ego_dict = {'x': self.current_odom_ego.x, 'y': self.current_odom_ego.y, 'z': 0.0, 'yaw': self.current_odom_ego.yaw}
        self.near_dir_tmp, self.final_dir_tmp = self.db3_extractor.extract_waypoint_vectors(global_ways_list, current_odom_ego_dict, self.current_odom_ego.yaw, k_meters=10.0)

        self.near_dir['x'] = self.near_dir_tmp['x']
        self.near_dir['y'] = -self.near_dir_tmp['y']
        self.final_dir['x'] = self.final_dir_tmp['x']
        self.final_dir['y'] = -self.final_dir_tmp['y']
        
        # closest_global_way = None
        # for gw in global_way_list:
        #     time_diff = abs(gw["timestamp"] - self.current_odom_ego["timestamp"]) / 1_000_000_000.0
        #     if time_diff <= global_way_threshold:
        #         if closest_global_way is None or abs(gw["timestamp"] - self.current_odom_ego["timestamp"]) < abs(closest_global_way["timestamp"] - self.current_odom_ego["timestamp"]):
        #             closest_global_way = gw

        # if closest_global_way:
        #     # PointCloud2 메시지에서 포인트 추출
        #     global_waypoints = self.extract_points_from_pointcloud2(closest_global_way["msg"], max_points=999999)
        #     near_destination_direction, final_destination_direction = self.extract_waypoint_vectors(
        #         global_waypoints, current_position, current_yaw, k_meters
        #     )
        #     self.db3_extractor.extract_waypoints_from_msg(msg)
            # def extract_waypoint_vectors(self, global_waypoints, current_position, current_yaw, k_meters=10.0):



    # def extract_waypoint_vectors(self, global_waypoints, current_position, current_yaw, k_meters=10.0):
    #     # 전역경로에서 K미터 지점의 방위벡터와 목적지 방위벡터 추출 (차량 좌표계로 변환)
    #     if not global_waypoints or len(global_waypoints) < 2:
    #         return None, None
        
    #     # 현재 위치에서 가장 가까운 웨이포인트 찾기
    #     min_dist = float('inf')
    #     closest_idx = 0
        
    #     for i, wp in enumerate(global_waypoints):
    #         dx = wp['x'] - current_position['x']
    #         dy = wp['y'] - current_position['y']
    #         dist = math.sqrt(dx**2 + dy**2)
    #         if dist < min_dist:
    #             min_dist = dist
    #             closest_idx = i
        
    #     # K미터 지점 찾기
    #     accumulated_dist = 0.0
    #     k_meter_idx = closest_idx
        
    #     for i in range(closest_idx, len(global_waypoints) - 1):
    #         wp1 = global_waypoints[i]
    #         wp2 = global_waypoints[i + 1]
    #         dx = wp2['x'] - wp1['x']
    #         dy = wp2['y'] - wp1['y']
    #         segment_dist = math.sqrt(dx**2 + dy**2)
            
    #         if accumulated_dist + segment_dist >= k_meters:
    #             k_meter_idx = i + 1
    #             break
    #         accumulated_dist += segment_dist
        
    #     # 근접 방위벡터 (현재 위치에서 K미터 지점으로의 방향)
    #     k_meter_point = global_waypoints[k_meter_idx]
    #     near_destination_direction = self.calculate_direction_vector(current_position, k_meter_point)
        
    #     # 목적지 방위벡터 (현재 위치에서 목적지로의 방향)
    #     destination_point = global_waypoints[-1]
    #     final_destination_direction = self.calculate_direction_vector(current_position, destination_point)
        
    #     # 전역 좌표계 벡터를 차량 좌표계로 변환
    #     cos_yaw = math.cos(-current_yaw)
    #     sin_yaw = math.sin(-current_yaw)
        
    #     # Near destination 변환
    #     if near_destination_direction:
    #         global_x = near_destination_direction['x']
    #         global_y = near_destination_direction['y']
    #         local_x = global_x * cos_yaw - global_y * sin_yaw
    #         local_y = global_x * sin_yaw + global_y * cos_yaw
    #         near_destination_direction = {'x': local_x, 'y': local_y}
        
    #     # Final destination 변환
    #     if final_destination_direction:
    #         global_x = final_destination_direction['x']
    #         global_y = final_destination_direction['y']
    #         local_x = global_x * cos_yaw - global_y * sin_yaw
    #         local_y = global_x * sin_yaw + global_y * cos_yaw
    #         final_destination_direction = {'x': local_x, 'y': local_y}
        
    #     return near_destination_direction, final_destination_direction
    
    def ldm_callback(self, msg: PointCloud2):
        # TODO: data_parser.py 의 parse_ldm_points() 함수 참고
        self.get_logger().info('Received PointCloud2 message')
        self.get_logger().info(f'PointCloud width: {msg.width}, height: {msg.height}')
        self.get_logger().info(f'Available fields: {[f.name for f in msg.fields]}') #['x', 'y', 'z', 'rgb', 'step_height', 'slope', 'class_id', 'probability', 'origin_x', 'origin_y', 'obstacles_type', 'occluded_point']
        points = point_cloud2.read_points(msg, field_names=('x', 'y', 'z', 'class_id', 'obstacles_type'), skip_nans=True)
        self.current_ldm_points = [
            LDMPoint(x=p[0], y=p[1], class_id=int(p[3]), obstacles_type=int(p[4]))
            for p in points
        ]
        # points[0]['class_id']
        # points[0]['obstacles_type']    

        # # class_id 분포 확인
        # class_ids = [int(p[3]) for p in points]
        # unique_ids = set(class_ids)
        # self.get_logger().info(f'Unique class_ids: {unique_ids}')
        
        # # class_id별 개수
        # from collections import Counter
        # counts = Counter(class_ids)
        # self.get_logger().info(f'Class ID counts: {dict(counts)}')

    def get_current_episode(self) -> Episode:
        """추론용 현재 프레임으로 단일 스텝 Episode 생성"""
        if self.current_odom_ego is None:
            return None
        
        observation = self.obs_builder.build_observation(
            self.current_odom_ego,
            self.current_objects,
            self.current_ldm_points,
            near_destination_direction=self.near_dir,
            final_destination_direction=self.final_dir,
            episode_start_ego=self.current_odom_ego
        )
        
        state_dict = {
            'obstacles': torch.tensor(observation['grid'][0], dtype=torch.float32),
            'static_environment': torch.tensor(observation['grid'][1], dtype=torch.float32),
            'ground_class_id': observation['grid'][2],
            'near_destination_direction': self.near_dir,
            'final_destination_direction': self.final_dir,
            'simulated_ego_pos': (49, 49)
        }
        
        # max_episode_step_length 만큼 Episode 생성 (MCTS trajectory 예측용)
        episode = Episode(source_file="ros2_realtime")
        num_steps = self.config.max_episode_step_length

        # 첫 step만 실제 데이터, 나머지는 placeholder (self_play에서 덮어씀)
        episode.states.append(state_dict)
        episode.actions.append(0)
        episode.rewards.append(0.0)
        episode.pi.append(None)
        episode.values.append(0.0)
        episode.dones.append(False)
        episode.legal_mask.append(None)
        episode.expert_ego_pos.append((49, 49))

        for _ in range(num_steps - 1):
            episode.states.append(state_dict)  # 동일한 state로 채움 (self_play에서 덮어씀)
            episode.actions.append(0)
            episode.rewards.append(0.0)
            episode.pi.append(None)
            episode.values.append(0.0)
            episode.dones.append(False)
            episode.legal_mask.append(None)
            episode.expert_ego_pos.append((49, 49))

        return episode

    def grid_to_odomgyro(self, grid_path: list, ego_info: EgoInfo) -> list:
        """Grid 좌표를 odomgyro 좌표계로 변환

        Args:
            grid_path: [(row, col), ...] 형태의 grid 좌표 리스트
            ego_info: 변환 기준이 되는 EgoInfo (deepcopy된 스냅샷 사용 권장)

        Returns:
            [(x, y), ...] 형태의 odomgyro 좌표 리스트
        """
        if ego_info is None or not grid_path:
            return []

        # grid 중심 (ego 위치)
        grid_center_row = self.config.observation_shape[1] // 2  # 49
        grid_center_col = self.config.observation_shape[2] // 2  # 49
        cell_size = self.config.cell_size  # 0.2m

        # ego의 odomgyro 좌표와 yaw
        ego_x = ego_info.x
        ego_y = ego_info.y
        ego_yaw = ego_info.yaw

        odom_path = []
        for row, col in grid_path:
            # grid 좌표에서 ego 기준 상대 좌표 (미터)
            # row: 전방 방향 (row가 클수록 전방)
            # col: 좌우 방향 (col이 클수록 우측)
            local_x = (row - grid_center_row) * cell_size  # 전방
            local_y = -(col - grid_center_col) * cell_size  # 좌측이 +, 우측이 -

            # ego yaw로 회전하여 odomgyro 좌표계로 변환
            cos_yaw = math.cos(ego_yaw)
            sin_yaw = math.sin(ego_yaw)

            global_x = ego_x + local_x * cos_yaw - local_y * sin_yaw
            global_y = ego_y + local_x * sin_yaw + local_y * cos_yaw

            odom_path.append((global_x, global_y))

        return odom_path

    def publish_predicted_path(self, grid_path: list, ego_info: EgoInfo):
        """Predicted path를 ROS2 PointCloud2 메시지로 발행

        Args:
            grid_path: [(row, col), ...] 형태의 grid 좌표 리스트
            ego_info: 변환 기준이 되는 EgoInfo (deepcopy된 스냅샷 사용 권장)
        """
        odom_path = self.grid_to_odomgyro(grid_path, ego_info)
        if not odom_path:
            return

        # PointCloud2 메시지 생성
        points = [(x, y, 0.0) for x, y in odom_path]  # z=0

        header = point_cloud2.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'odom'  # odomgyro 좌표계

        # x, y, z 필드 정의
        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
        ]

        cloud_msg = point_cloud2.create_cloud(header, fields, points)

        self.predicted_path_publisher.publish(cloud_msg)
        self.get_logger().info(f'Published predicted path with {len(odom_path)} points')

__DEBUG_EPISODE_VISUALIZER__ = True
def main(args=None):
    rclpy.init(args=args)
    config = MuZeroConfig()
    node = SampleSubscriber(config)

    # 창 한 번만 생성
    plt.ion()
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # plt.tight_layout()
    visualizer = None

    if torch.cuda.is_available():
        if isinstance(config.num_usage_of_gpu_devices, list) and len(config.num_usage_of_gpu_devices) > 0:
            device = torch.device(f"cuda:{config.num_usage_of_gpu_devices[0]}")
        elif config.num_usage_of_gpu_devices is not None:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 네트워크 생성
    network = get_network(config, device)


    # 체크포인트 로드 (있는 경우)
    checkpoint_path = config.resume_checkpoint_path
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint['network_state_dict'])
    network_state_dict = network.state_dict()

    import time as time_module
    import copy
    try:
        while rclpy.ok():
            loop_start_time = time_module.time()

            rclpy.spin_once(node, timeout_sec=0.1)
            episode = copy.deepcopy(node.get_current_episode())

            # episode가 None이면 다음 루프로 건너뜀
            if episode is None:
                node.get_logger().warn('Episode is None, waiting for data...')
                continue

            # EgoInfo 스냅샷 (publish_predicted_path에서 사용)
            current_ego_snapshot = copy.deepcopy(node.current_odom_ego)

            # Loop time 출력
            loop_elapsed = time_module.time() - loop_start_time
            node.get_logger().info(f'Get Episode Loop time: {loop_elapsed*1000:.1f} ms ({1/loop_elapsed:.1f} Hz)')

            loop_start_time = time_module.time()
            # MCTS enrichment




            # 추론 모드 (operating_mode == 1): network를 6번째 인자로 전달하여 재사용
            args = (0, network_state_dict, config, device, episode, network)
            enriched_episode = enrich_episode_with_mcts_parallel_worker(args)

            # Loop time 출력
            loop_elapsed = time_module.time() - loop_start_time
            node.get_logger().info(f'Inference Model Loop time: {loop_elapsed*1000:.1f} ms ({1/loop_elapsed:.1f} Hz)')
            loop_start_time = time_module.time()

            # enriched_episode 의 Episode 별 simulated_ego_pos 의 위치를 파싱
            predicted_path_xy = []
            for st in enriched_episode.states:
                pos = st.get('simulated_ego_pos')
                if pos is not None:
                    predicted_path_xy.append(pos)  # (row, col) 형태

            # 디버그: predicted_path 좌표 출력
            if predicted_path_xy:
                node.get_logger().info(f'Predicted path points: {len(predicted_path_xy)}')
                node.get_logger().info(f'First 5 points (row, col): {predicted_path_xy[:5]}')
                if len(predicted_path_xy) > 5:
                    node.get_logger().info(f'Last 5 points (row, col): {predicted_path_xy[-5:]}')

            # Loop time 출력
            loop_elapsed = time_module.time() - loop_start_time
            node.get_logger().info(f'Parse Predicted Path Loop time: {loop_elapsed*1000:.1f} ms ({1/loop_elapsed:.1f} Hz)')

            if enriched_episode is not None:
                loop_start_time = time_module.time()
                if __DEBUG_EPISODE_VISUALIZER__:
                    if visualizer is None:
                        # 첫 번째 episode로 visualizer 생성
                        visualizer = EpisodeInteractiveVisualizer(
                            episodes=[enriched_episode],
                            config=config,
                            interval=200
                        )
                        # show() 내부 로직을 수동으로 실행 (plt.show() 제외)
                        visualizer.current_episode = 0
                        visualizer.current_step = 0
                        visualizer.fig = plt.figure(figsize=(16, 7))
                        gs = visualizer.fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.15])
                        visualizer.ax_grid = visualizer.fig.add_subplot(gs[0, 0])
                        visualizer.ax_info = visualizer.fig.add_subplot(gs[0, 1])
                        visualizer.ax_toggle = visualizer.fig.add_subplot(gs[0, 2])
                        visualizer.setup_toggle_buttons()
                        visualizer.fig.canvas.mpl_connect('key_press_event', visualizer.on_key)
                        plt.tight_layout()
                        visualizer.init_graphics()
                    else:
                        # episode 데이터 교체 및 캐시 갱신
                        visualizer.episodes = [enriched_episode]
                        visualizer.preprocess_episodes()

                    # predicted path 디버깅 시각화
                    if predicted_path_xy:
                        visualizer.set_predicted_path(predicted_path_xy)

                    # 프레임 업데이트
                    visualizer.update_frame()
                    visualizer.fig.canvas.draw()
                    visualizer.fig.canvas.flush_events()


                # Loop time 출력
                loop_elapsed = time_module.time() - loop_start_time
                node.get_logger().info(f'Visualize Loop time: {loop_elapsed*1000:.1f} ms ({1/loop_elapsed:.1f} Hz)')
                loop_start_time = time_module.time()

                # predicted path ROS2 Topic 발생
                if predicted_path_xy:
                    node.publish_predicted_path(predicted_path_xy, current_ego_snapshot)

                # Loop time 출력
                loop_elapsed = time_module.time() - loop_start_time
                node.get_logger().info(f'Publish Predicted Path Loop time: {loop_elapsed*1000:.1f} ms ({1/loop_elapsed:.1f} Hz)')
                loop_start_time = time_module.time()


            # predicted_path_xy = [
            #     st.get('simulated_ego_pos')
            #     for st in enriched_episode.states
            #     if st.get('simulated_ego_pos') is not None
            # ]



            # if episode is not None:
            #     # 시각화 (단일 Episode를 리스트로 감싸서 전달)
            #     visualizer = EpisodeInteractiveVisualizer(
            #         episodes=[episode],
            #         config=config,
            #         interval=200
            #     )
            #     visualizer.show()  # 또는 visualizer.run()

            # if episode is not None:
            #     state = episode.states[0]

            #     # 각 axes clear 후 다시 그리기
            #     for ax in axes:
            #         ax.clear()

            #     # Channel 0: 동적 객체 (obstacles)
            #     axes[0].imshow(state['obstacles'].numpy(), cmap='hot')
            #     axes[0].set_title('Obstacles (Ch0)')
            #     axes[0].plot(49, 49, 'go', markersize=8)

            #     # Channel 1: 정적 환경 (static_environment)
            #     axes[1].imshow(state['static_environment'].numpy(), cmap='hot')
            #     axes[1].set_title('Static Environment (Ch1)')
            #     axes[1].plot(49, 49, 'go', markersize=8)

            #     # Channel 2: 지면 class_id (ground_class_id)
            #     axes[2].imshow(state['ground_class_id'], cmap='tab20')
            #     axes[2].set_title('Ground Class ID (Ch2)')
            #     axes[2].plot(49, 49, 'go', markersize=8)

            #     fig.canvas.draw()
            #     fig.canvas.flush_events()

            print("야호")
        # rclpy.spin(node)
        # print("Spinning node...")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
