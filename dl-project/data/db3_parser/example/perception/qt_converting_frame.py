#!/usr/bin/env python3
import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QLabel,
    QListWidget,
    QSlider,
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QLabel,
    QListWidget,
    QInputDialog,
    QSizePolicy,
)  # QInputDialog 추가
from PyQt5.QtWidgets import QCheckBox, QGroupBox, QScrollArea, QVBoxLayout, QWidget

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from bag_parser import BagParser
from PyQt5.QtWidgets import QSizePolicy


class VideoPlayerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.paused = False
        self.slider_being_dragged = False
        self.setFocusPolicy(Qt.StrongFocus)  # 키보드 포커스 활성화
        self.setup_ui()

    def toggle_play(self):
        if self.cap is None:
            return

        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
            self.play_button.setText("⏵")  # 재생 아이콘
        else:
            self.timer.start()
            self.play_button.setText("⏸")  # 일시정지 아이콘

    def keyPressEvent(self, event):
        print(f"Key pressed: {event.key()}, Space key: {Qt.Key_Space}")
        if event.key() == Qt.Key_Space:
            print("Space key detected!")
            self.toggle_play()
        super().keyPressEvent(event)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 마진 제거하여 전체 공간 활용

        # 비디오 표시 레이블
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setSizePolicy(sizePolicy)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # 컨트롤 패널 컨테이너
        control_container = QWidget()
        control_container.setFixedHeight(100)  # 컨트롤 패널 높이 고정
        controls_layout = QVBoxLayout(control_container)
        controls_layout.setContentsMargins(5, 0, 5, 5)

        # 재생바
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        controls_layout.addWidget(self.slider)

        # 시간 표시
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.total_time_label = QLabel("00:00")
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        controls_layout.addLayout(time_layout)

        # 재생 컨트롤 버튼
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        self.play_button = QPushButton("|>")
        self.play_button.setFixedSize(40, 40)
        self.play_button.clicked.connect(self.toggle_play)

        self.stop_button = QPushButton("⏹")
        self.stop_button.setFixedSize(40, 40)
        self.stop_button.clicked.connect(self.stop_video)

        # 컨트롤 버튼 레이아웃에 프레임 이동 버튼 추가
        self.prev_frame_btn = QPushButton("⏮")  # 이전 프레임
        self.next_frame_btn = QPushButton("⏭")  # 다음 프레임
        self.prev_frame_btn.setFixedSize(40, 40)
        self.next_frame_btn.setFixedSize(40, 40)

        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)

        buttons_layout.addWidget(self.prev_frame_btn)
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.next_frame_btn)

        buttons_layout.addStretch()
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addStretch()

        controls_layout.addLayout(buttons_layout)
        layout.addWidget(control_container)

    def prev_frame(self):
        if self.cap is None:
            return

        # 재생 중이라면 일시정지
        if not self.paused:
            self.toggle_play()

        # 현재 프레임 위치 가져오기
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 이전 프레임으로 이동
        prev_frame = max(
            0, current_frame - 2
        )  # -2를 하는 이유는 읽을 때 1이 더해지기 때문
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)

        # 프레임 업데이트
        self.update_frame()

    def next_frame(self):
        if self.cap is None:
            return

        # 재생 중이라면 일시정지
        if not self.paused:
            self.toggle_play()

        # 프레임 업데이트
        self.update_frame()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_Left:  # 왼쪽 화살표
            self.prev_frame()
        elif event.key() == Qt.Key_Right:  # 오른쪽 화살표
            self.next_frame()
        super().keyPressEvent(event)

    def start_video(self, video_path):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # 비디오 정보 설정
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_duration = self.total_frames / self.fps

        # 슬라이더 범위 설정
        self.slider.setRange(0, self.total_frames)

        # 총 시간 표시 업데이트
        total_minutes = int(self.total_duration // 60)
        total_seconds = int(self.total_duration % 60)
        self.total_time_label.setText(f"{total_minutes:02d}:{total_seconds:02d}")

        self.paused = False
        self.play_button.setText("⏸")  # 일시정지 아이콘으로 변경
        self.timer.start(int(1000 / self.fps))  # fps에 맞춰 타이머 설정

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            # 현재 레이블 크기에 맞춰 프레임 크기 조정
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
            )

            # 레이블의 현재 크기에 맞춰 이미지 크기 조정
            label_size = self.video_label.size()
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

            # 시간 표시 업데이트
            if not self.slider_being_dragged:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.slider.setValue(current_frame)
                current_time = current_frame / self.fps
                minutes = int(current_time // 60)
                seconds = int(current_time % 60)
                self.current_time_label.setText(f"{minutes:02d}:{seconds:02d}")
        else:
            # 영상이 끝나면 처음으로 돌아가기
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.slider.setValue(0)
            self.current_time_label.setText("00:00")
            # 재생 상태 유지
            if not self.paused:
                self.timer.start()

    def toggle_play(self):
        print(f"Toggle play called. Cap: {self.cap}, Paused: {self.paused}")
        if self.cap is None:
            print("No video loaded")
            return

        self.paused = not self.paused
        if self.paused:
            print("Pausing video")
            self.timer.stop()
            self.play_button.setText("⏵")  # 재생 아이콘
        else:
            print("Resuming video")
            self.timer.start()
            self.play_button.setText("⏸")  # 일시정지 아이콘

    def stop_video(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, "video_label"):
            self.video_label.clear()
        self.play_button.setText("⏵")
        self.slider.setValue(0)
        self.current_time_label.setText("00:00")
        self.paused = False

    def slider_pressed(self):
        self.slider_being_dragged = True
        if not self.paused:
            self.timer.stop()

    def slider_released(self):
        if self.cap is not None:
            frame_pos = self.slider.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            if not self.paused:
                self.timer.start()
        self.slider_being_dragged = False

    def closeEvent(self, event):
        self.stop_video()
        event.accept()
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()  # 비디오 캡처 해제

        event.accept()


class VideoConverterThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(
        self,
        bag_file,
        filter_ids=None,
        filter_classes=None,
        output_suffix="_frame_reader",
    ):
        super().__init__()
        self.bag_file = bag_file
        self.parser = None  # parser 변수 추가
        self.filter_ids = filter_ids if filter_ids is not None else []
        self.filter_classes = filter_classes if filter_classes is not None else []
        self.output_suffix = output_suffix  # 새로운 변수 추가

    def run(self):
        try:
            rclpy.init(args=None)
            self.parser = BagParser()  # parser 인스턴스 저장
            self.parser.filter_ids = (
                self.filter_ids if self.filter_ids is not None else []
            )
            self.parser.filter_classes = self.filter_classes

            # output 파일명 설정 - 사용자 지정 suffix 사용
            base_name = os.path.splitext(os.path.basename(self.bag_file))[0]
            output_dir = os.path.dirname(self.bag_file)
            output_path = os.path.join(
                output_dir, f"{base_name}{self.output_suffix}.mp4"
            )
            self.parser.output_path = output_path

            # 진행상황을 업데이트하기 위한 콜백 설정
            def progress_callback(percent):
                self.progress.emit(percent)

            self.parser.set_progress_callback(progress_callback)

            # 파일 처리
            self.parser.read_bag(self.bag_file)
            self.parser.process_synchronized_data()

            # 완료 시그널 발생
            self.finished.emit(output_path)

        except Exception as e:
            print(f"Error in conversion thread: {str(e)}")
        finally:
            try:
                self.parser.destroy_node()
                rclpy.shutdown()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROS2 Bag Video Converter")
        self.setMinimumSize(1200, 800)

        # 메인 위젯과 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 왼쪽 패널
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)

        # 파일/폴더 선택 버튼
        self.select_file_btn = QPushButton("Select Bag File")
        self.select_folder_btn = QPushButton("Select Folder")
        self.select_file_btn.clicked.connect(self.select_bag_file)
        self.select_folder_btn.clicked.connect(self.select_folder)

        # db3 파일 리스트 레이블과 위젯
        db3_label = QLabel("DB3 Files:")
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(200)  # 높이 제한

        # mp4 파일 리스트 레이블과 위젯
        mp4_label = QLabel("Converted MP4 Files:")
        self.mp4_list = QListWidget()
        self.mp4_list.setMaximumHeight(200)  # 높이 제한
        self.mp4_list.itemClicked.connect(self.play_selected_video)  # 클릭 이벤트 연결

        # 파일 이름 설정 버튼 추가
        self.output_name_btn = QPushButton("Set Output Name")
        self.output_name_btn.clicked.connect(self.set_output_name)
        self.output_suffix = "_frame_reader"  # 기본값 설정
        # 진행 상황 표시
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")

        # 다음 파일 변환 버튼
        self.next_btn = QPushButton("Convert Next")
        self.next_btn.clicked.connect(self.convert_next_file)
        self.next_btn.setEnabled(False)

        filter_group = QGroupBox("Filter Classes")
        filter_layout = QVBoxLayout()

        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 체크박스 생성
        self.class_checkboxes = {}
        categories = {
            "person": ["person"],
            "vehicle": [
                "bicycle",
                "bus",
                "car",
                "motorcycle",
                "scooter",
                "truck",
                "cart",
            ],
            "static_obstacles": ["bollard"],
            "dynamic_obstacles": ["neubie"],
            "unclassified": ["animal", "wheelchair", "stroller"],
        }
        # 왼쪽 패널의 위젯 추가 순서를 정리
        left_layout.addWidget(self.select_file_btn)
        left_layout.addWidget(self.select_folder_btn)
        left_layout.addWidget(self.output_name_btn)  # 파일 이름 설정 버튼
        left_layout.addWidget(db3_label)
        left_layout.addWidget(self.file_list)
        left_layout.addWidget(mp4_label)
        left_layout.addWidget(self.mp4_list)
        left_layout.addWidget(filter_group)  # 필터 그룹
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.progress_label)
        left_layout.addWidget(self.next_btn)
        left_layout.addStretch()

        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.progress_label)
        left_layout.addWidget(self.next_btn)
        left_layout.addStretch()

        for category, classes in categories.items():
            category_label = QLabel(category.upper())
            category_label.setStyleSheet("font-weight: bold;")
            scroll_layout.addWidget(category_label)

            for class_name in classes:
                checkbox = QCheckBox(class_name)
                self.class_checkboxes[class_name] = checkbox
                scroll_layout.addWidget(checkbox)

        scroll.setWidget(scroll_widget)
        filter_layout.addWidget(scroll)
        filter_group.setLayout(filter_layout)

        # 왼쪽 패널에 필터 그룹 추가 (progress_bar 앞에)

        # 오른쪽 패널 (비디오 재생)
        self.video_widget = VideoPlayerWidget()
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setSizePolicy(sizePolicy)

        # 레이아웃에 패널 추가
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.video_widget, 1)

        # 변환 큐와 현재 처리 중인 파일 추적
        self.conversion_queue = []
        self.current_thread = None
        self._clean_up_needed = True
        self.setFocusPolicy(Qt.StrongFocus)
        # 윈도우 닫힐 때 정리
        self.setAttribute(Qt.WA_DeleteOnClose, True)

    def set_output_name(self):
        """출력 파일 이름 suffix 설정"""
        text, ok = QInputDialog.getText(
            self,
            "Set Output Name",
            "Enter suffix for output files (e.g., '_frame_reader'):",
            text=self.output_suffix,
        )
        if ok and text:
            self.output_suffix = text

    def _clean_up(self):
        # 비디오 위젯 정리
        if hasattr(self, "video_widget"):
            self.video_widget.stop_video()

        # 현재 실행 중인 변환 스레드 정리
        if self.current_thread is not None:
            self.current_thread.stop()
            self.current_thread.wait()
            self.current_thread = None

        # 리스트 위젯 정리
        if hasattr(self, "file_list"):
            self.file_list.clear()
        if hasattr(self, "mp4_list"):
            self.mp4_list.clear()

        self._clean_up_needed = False

    def keyPressEvent(self, event):
        print(f"Key pressed: {event.key()}, Space key: {Qt.Key_Space}")
        if event.key() == Qt.Key_Space:
            print("Space key detected!")
            self.toggle_play()
        super().keyPressEvent(event)

    def play_selected_video(self, item):
        """mp4 리스트에서 선택된 비디오 재생"""
        if not item:
            return
        filename = item.text()
        if self.conversion_queue:
            directory = os.path.dirname(self.conversion_queue[0])
        else:
            for file in self.file_list.findItems("", Qt.MatchContains):
                directory = os.path.dirname(file.text())
                break

        video_path = os.path.join(directory, filename)
        if os.path.exists(video_path):
            self.video_widget.start_video(video_path)

    def check_converted_files(self, files):
        """선택된 db3 파일들에 대해 이미 변환된 mp4 파일이 있는지 확인"""
        self.mp4_list.clear()
        if not files:
            return

        directory = os.path.dirname(files[0])
        existing_items = set()  # 중복 체크를 위한 set 사용

        for db3_file in files:
            base_name = os.path.splitext(os.path.basename(db3_file))[0]

            # 디렉토리 스캔 최적화
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if (
                            entry.name.endswith(".mp4")
                            and base_name in entry.name
                            and entry.name not in existing_items
                        ):
                            self.mp4_list.addItem(entry.name)
                            existing_items.add(entry.name)
            except OSError:
                continue

    def select_bag_file(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Bag Files", "", "ROS2 Bag Files (*.db3)"
        )
        if files:
            self.file_list.clear()
            sorted_files = sorted(files)
            for file in sorted_files:
                self.file_list.addItem(os.path.basename(file))

            self.conversion_queue = sorted_files
            self.next_btn.setEnabled(True)
            self.check_converted_files(sorted_files)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.file_list.clear()
            bag_files = sorted(
                [
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.endswith(".db3")
                ]
            )
            for file in bag_files:
                self.file_list.addItem(os.path.basename(file))

            self.conversion_queue = bag_files
            self.next_btn.setEnabled(True)
            self.check_converted_files(bag_files)

    def conversion_finished(self, video_path):
        # 비디오 재생
        self.video_widget.start_video(video_path)

        # mp4 리스트 업데이트
        self.mp4_list.addItem(os.path.basename(video_path))

        # 다음 파일이 있으면 다음 버튼 활성화
        if self.conversion_queue:
            self.next_btn.setEnabled(True)

    def convert_next_file(self):
        if self.current_thread is not None:
            self.current_thread.stop()
            self.current_thread.wait()
            self.current_thread = None

        if not self.conversion_queue:
            self.progress_label.setText("All files processed")
            self.next_btn.setEnabled(False)
            return

        current_file = self.conversion_queue.pop(0)
        self.progress_label.setText(f"Converting: {os.path.basename(current_file)}")
        self.progress_bar.setValue(0)

        # 선택된 클래스 수집
        selected_classes = [
            class_name
            for class_name, checkbox in self.class_checkboxes.items()
            if checkbox.isChecked()
        ]

        # Use a QInputDialog to get the IDs
        filter_ids, ok = QInputDialog.getMultiLineText(
            self, "Enter IDs", "Enter IDs to filter (separate with ,(comma)):", "-1"
        )
        if ok:
            temp_filter_ids_list = [
                int(id.strip()) for id in filter_ids.split(",") if id.strip()
            ]
            if all(id >= 0 for id in temp_filter_ids_list):
                filter_ids_list = temp_filter_ids_list
        else:
            filter_ids_list = None

        self.current_thread = VideoConverterThread(
            current_file,
            filter_ids_list,
            filter_classes=selected_classes,
            output_suffix=self.output_suffix,  # suffix 전달
        )
        self.current_thread.progress.connect(self.update_progress)
        self.current_thread.finished.connect(self.conversion_finished)

        self.current_thread.start()

        self.next_btn.setEnabled(False)

    def update_progress(self, value):
        self.progress_bar.setValue(value)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
