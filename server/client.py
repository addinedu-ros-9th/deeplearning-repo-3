import socket, json, cv2, threading, time, sys, queue, struct, base64, multiprocessing
import numpy as np

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import uic

from threading import Lock

from distance import *

import pandas as pd

from multiprocessing import Process, Queue, Manager

from typing import Any

from db_connector import DBConnector
from db_inserter import *
from datetime import datetime



# 서버 IP 및 포트 정보
LANE_SERVER_IP = "192.168.0.252"
TCP_LANE_PORT = 12345
UDP_LANE_PORT = 54321

OBJ_SERVER_IP = "192.168.0.19"
TCP_OBJ_PORT = 12346
UDP_OBJ_PORT = 54322

orig_frame = {}
lane_mask = {}
obj_mask = {}

frame_time = {}  # uuid: timestamp

HEADER_LENGTH = 4
UUID_LENGTH = 4

original_latency_check = {}
lane_latency_check = {}
obj_latency_check = {}

from_class = uic.loadUiType("client_video.ui")[0]

class UdpSender():
    def __init__(self):
        self.udp_lane = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.uuid_counter = 0

        self.fps_limit = 15
        self.frame_interval = 1 / self.fps_limit
        self.next_time = time.perf_counter()

    def send_frame(self):
        global original_latency_check

        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(0)
        
        try:
            if not self.cap.isOpened():
                print("[UDP] Webcam open failed")
                return

            while True:
                # 프레임 조절
                now = time.perf_counter()

                if now < self.next_time:
                    time.sleep(self.next_time - now)
                self.next_time += self.frame_interval
                
                # uuid
                self.uuid_counter += 1
                uuid_msg = self.uuid_counter.to_bytes(4, byteorder='big')

                # 프레임 캡처
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    continue

                frame = cv2.resize(frame, (512, 256))

                if not ret:
                    continue
                
                ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                
                if not ret:
                    continue

                try:
                    self.udp_lane.sendto(uuid_msg + b'||' + buffer.tobytes(), (LANE_SERVER_IP, UDP_LANE_PORT))
                    self.udp_obj.sendto(uuid_msg + b'||' + buffer.tobytes(), (OBJ_SERVER_IP, UDP_OBJ_PORT))

                    # time.sleep(0.05)
                    # print("origin:", type(frame)) # debug
                    udp_video_queue.put((self.uuid_counter, frame.copy()))
                    
                    # latency_check
                    original_latency_check[self.uuid_counter] = {"start":time.time()}
                    # print(original_latency_check)

                except Exception as e:
                    # print(f"[UDP SEND ERROR] {e}")
                    # import traceback
                    # traceback.print_exc()
                    return None
        finally:
            self.cap.release()
            cv2.destroyAllWindows()



    def close(self):
        self.udp_lane.close()
        self.udp_obj.close()
        self.cap.release()

def receive_tcp_lane(server_ip, server_port, lane_tcp_queue):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))
        sock.settimeout(5.0)

        def recv_exact(sock, num_bytes):
            buffer = b''
            while len(buffer) < num_bytes:
                chunk = sock.recv(num_bytes - len(buffer))
                if not chunk:
                    raise ConnectionError("Socket closed before expected data received")
                buffer += chunk
            return buffer

        while True:
            try:
                header = recv_exact(sock, 4)
                json_len = struct.unpack('>I', header)[0]

                uuid_raw = recv_exact(sock, 4)
                uuid = struct.unpack('>I', uuid_raw)[0]

                buffer = recv_exact(sock, json_len)
                result = json.loads(buffer.decode('utf-8'))

                if "pred_mask" not in result:
                    continue

                # base64 디코딩
                encoded = result["pred_mask"]
                data = base64.b64decode(encoded)
                nparr = np.frombuffer(data, np.uint8)
                pred_mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                lane_tcp_queue.put((uuid, pred_mask))
                # print(output_queue.qsize())
                
            except Exception as e:
                print(f"[receive_tcp_lane ERROR] {e}")
                continue
    finally:
        sock.close()

def lane_result_worker(lane_tcp_queue, lane_result_queue, lane_latency_check):
    try:
        while True:
            if not lane_tcp_queue.empty():
                # print("queue check")
                lane_data = lane_tcp_queue.get_nowait()
                uuid = lane_data[0]
                # print("uuid : ", uuid)
                # print("lane uuid : ", uuid) # debug
                pred_mask = lane_data[1]

                if pred_mask is None:
                    continue

                h, w = pred_mask.shape
                left_zone = pred_mask[int(h * 0.5):, int(w * 0.2):int(w * 0.4)]
                right_zone = pred_mask[int(h * 0.5):, int(w * 0.6):int(w * 0.8)]

                can_change_left = np.count_nonzero(left_zone == 2) > 40
                can_change_right = np.count_nonzero(right_zone == 2) > 40
                stop_line = np.count_nonzero(pred_mask == 4) > 50
                crosswalk = np.count_nonzero(pred_mask == 5) > 50

                msg = [0, 0, 0, 0]
                if can_change_left: msg[0] = 1
                if can_change_right: msg[1] = 1
                if stop_line: msg[2] = 1
                if crosswalk: msg[3] = 1

                lane_result_queue.put((uuid, pred_mask, msg))
                # print("lane: ", lane_result_queue.qsize())
                
                lane_latency_check[uuid] = {"lane": time.time()}

    except Exception as e:
        # print(f"[LANE RESULT PROCESS ERROR] {e}")
        # import traceback
        # traceback.print_exc()
        # continue
        return None

def receive_tcp_obj(server_ip, server_port, obj_tcp_queue: Queue):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))
        sock.settimeout(5.0)

        def recv_exact(sock, num_bytes):
            buffer = b''
            while len(buffer) < num_bytes:
                chunk = sock.recv(num_bytes - len(buffer))
                if not chunk:
                    raise ConnectionError("Socket closed before expected data received")
                buffer += chunk
            return buffer

        while True:
            try:
                header = recv_exact(sock, 4)
                # print(len(header))
                json_len = struct.unpack('>I', header)[0]

                uuid_raw = recv_exact(sock, 4)
                uuid = struct.unpack('>I', uuid_raw)[0]

                buffer = recv_exact(sock, json_len)
                json_data = json.loads(buffer.decode('utf-8'))

                obj_tcp_queue.put((uuid, json_data))
                # print(output_queue.qsize())

            except Exception as e:
                print(f"[receive_tcp_obj ERROR] {e}")
                continue

    finally:
        sock.close()

def obj_result_worker(obj_tcp_queue, obj_result_queue, obj_latency_check):

    color = {
        "car": (0, 255, 0),
        "child_protection": (255, 255, 0),
        "construction": (255, 0, 0),
        "person": (0, 0, 255),
        "speed_limit_30": (0, 200, 255),
        "speed_limit_50": (200, 0, 200),
        "stop_sign": (0, 255, 255),
        "veh_go": (255, 0, 255),
        "veh_goLeft": (128, 0, 255),
        "veh_stop": (0, 128, 255),
        "veh_warning": (255, 128, 0),
    }

    try:
        while True:
            if not obj_tcp_queue.empty():
                # print("queue check")
                obj_tcp_data = obj_tcp_queue.get_nowait()
                # print("queue extract")
                uuid = obj_tcp_data[0]
                # print("obj uuid : ", uuid) # debug

                mask = np.zeros((256, 512, 3), dtype=np.uint8)

                class_id = -1
                bbox = ()
                confidence = -1
                for obj_data in obj_tcp_data[1]:
                    if not isinstance(obj_data["bbox"], list):
                        continue

                    data_raw = obj_data['bbox']

                    x1 = data_raw[0]
                    y1 = data_raw[1]
                    x2 = data_raw[2]
                    y2 = data_raw[3]

                    bbox = (x1, y1, x2, y2)
                    confidence = obj_data['confidence']
                    class_name = obj_data.get('class_name', 'unknown')
                    #print(type(class_name))
                    label = f"{class_name}"
                    # color = (0, 255, 0)

                    class_id = obj_data.get('class_id', -1)
                    distance = estimate_obj_distance(class_name, (x1,y1,x2,y2))

                    class_id = obj_data.get('class_id', -1)
                    distance = estimate_obj_distance(class_name, (x1,y1,x2,y2))
                    if class_name != "unknown":
                        cv2.rectangle(mask, (x1, y1), (x2, y2), color[class_name], 2)
                        cv2.putText(mask, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[class_name], 2)
                    
                if not class_id == -1:
                    obj_result_queue.put((uuid, mask, class_id, confidence, bbox, distance))
                    # print("obj result : ", obj_result_queue.qsize())
                    
                    obj_latency_check[uuid] = {"obj": time.time()}

    except Exception as e:
        print(f"[OBJ RESULT PROCESS ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None
    
db_connector = DBConnector()
detected_object_inserter = DetectedObjectInserter(db_connector)
action_log_inserter = ActionLogInserter(db_connector)
drive_session_inserter = DriveSessionInserter(db_connector)
drive_session_updater = DriveSessionUpdater(db_connector)

def db_insert_worker_task(db_task_data, db_queue, db_config):
    """단일 DB 작업을 처리하는 함수 (기존 db_insert_worker 로직)"""
    try:
        # DB 연결 및 inserter 초기화
        
        
        task_type = db_task_data[0]
        raw_data = db_task_data[1]
        # print(raw_data)
        # print("type:", task_type)
        
        # 데이터 변환 및 삽입
        if task_type == "detected_object":
            if raw_data['session_id'] != None:
                #print("afdadsaf")
                object_id = detected_object_inserter.insert_detected_object(
                    session_id=raw_data['session_id'],
                    object_type_id=raw_data['object_type_id'],
                    detected_time=raw_data['detected_time'],
                    confidence=raw_data['confidence'],
                    bbox=raw_data['bbox'],
                    position=raw_data['position']
                )
                # print("queue insert")
                db_config.put(object_id)
                # print("queue insert success ", db_config.qsize())
        
        elif task_type == "action_log":
            if raw_data['object_id'] != None:
                action_log_inserter.insert_action_log(
                    object_id=raw_data['object_id'],
                    action_type_id=raw_data['action_type_id'],
                    performed_time=raw_data['performed_time'],
                    delay=raw_data['delay']
                )
        
        elif task_type == "drive_session":
            session_id = drive_session_inserter.insert_drive_session(
                start_time=raw_data['start_time'],
                end_time=raw_data['end_time'],
                total_distance=raw_data['total_distance']
            )
            db_config.put(session_id)
            # print(session_id)
        
        elif task_type == "update_session":
            drive_session_updater.update_end_time_and_distance(
                session_id=raw_data['session_id'],
                end_time=raw_data['end_time'],
                total_distance=raw_data['total_distance']
            )
                    
    except Exception as e:
        print(f"[DB INSERT TASK ERROR] {e}")

def db_insert_worker_wrapper(db_queue, db_config):
    """db_insert_worker를 감싸서 None (sentinel) 값을 처리합니다."""
    
    try:
        while True:
            task = db_queue.get()
            # print(task[1])
            if task is None:
                # print("[DB WORKER] Sentinel 값 수신, 종료합니다.")
                break
            
            # 원래의 워커 함수 호출
            db_insert_worker_task(task, db_queue, db_config)

    except Exception as e:
        #print(f"[DB WRAPPER ERROR] {e}")
        pass

def db_extactor(db_config, temp_queue, db_queue):
    session_id = 0
    while True:
        if session_id == 0 and not db_config.empty():
            
            session_id = db_config.get_nowait()
            print(f"[VideoUpdateThread] session_id 획득: {session_id}")
        else:
            pass
            
        if session_id != 0:
            while not temp_queue.empty():
                raw_data = temp_queue.get_nowait()

                obj_class = raw_data[0]
                detected_time = raw_data[1]
                confidence = raw_data[2]
                bbox = raw_data[3]
                position = raw_data[4]


                if session_id != None:
                    # print(session_id)
                    db_queue.put(("detected_object", {
                        "session_id": session_id,
                        "object_type_id": obj_class,
                        "detected_time": detected_time,
                        "confidence": confidence,
                        "bbox": bbox,
                        "position": position
                    }))
        
        

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("COVA II")
        #self.setStyleSheet(open("/home/lee/dev_ws/projects/DL_project/final/server/dark_style.css").read())

        self.prev_lane_mask = None
        self.prev_obj_mask = None

        

        self.video_thread = VideoUpdateThread(self.label_video_lane.width(), self.label_video_lane.height())
        # 예: MainWindow 클래스에서 VideoUpdateThread 객체를 생성 후 연결
        

        # Signal 연결
        self.video_thread.lane_message.connect(self.label_msg_lane.setText)
        self.video_thread.alert_message.connect(self.label_msg_alert.setText)
        self.video_thread.obj_message.connect(self.label_msg_obj.setText)
        self.video_thread.state_message.connect(self.label_msg_state.setText)

        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.start()


        self.udp_sender = UdpSender()

        self.lane_recv = Process(target=receive_tcp_lane,
                                args=(LANE_SERVER_IP, TCP_LANE_PORT, lane_tcp_queue))
        self.obj_recv = Process(target=receive_tcp_obj,
                                args=(OBJ_SERVER_IP, TCP_OBJ_PORT, obj_tcp_queue))
        
        self.lane_recv.start()
        self.obj_recv.start()

        self.lane_process = Process(target=lane_result_worker,
                                    args=(lane_tcp_queue, lane_result_queue, lane_latency_check))
        self.obj_process = Process(target=obj_result_worker,
                                args=(obj_tcp_queue, obj_result_queue, obj_latency_check))

        self.lane_process.start()
        self.obj_process.start()

        self.db_process = Process(target=db_insert_worker_wrapper,
                                   args=(db_queue, db_config))
        self.db_process.start()

        self.db_extractor = Process(target=db_extactor,
                                    args=(db_config, temp_queue, db_queue))
        self.db_extractor.start()
        
        

        threading.Thread(target=self.udp_sender.send_frame, daemon=True).start()

        
        db_queue.put(("drive_session", {
            "start_time": datetime.now(),
            "end_time": None,
            "total_distance": 0.0
        }))
        
    
    def update_frame(self, pixmap: QPixmap):
        self.label_video_lane.setPixmap(pixmap)

    def closeEvent(self, event):
        self.video_thread.stop()
        
        self.udp_sender.close()

        self.lane_process.terminate()
        self.obj_process.terminate()
        self.lane_process.join()
        self.obj_process.join()

        self.lane_recv.terminate()
        self.obj_recv.terminate()
        self.lane_recv.join()
        self.obj_recv.join()

        # db_process가 모든 메시지를 처리하고 종료될 때까지 기다립니다.
        db_queue.put(None) # Sentinel value
        self.db_process.join()

        event.accept()

class VideoUpdateThread(QThread):
    frame_ready = pyqtSignal(QPixmap)

    # 추가된 Signals (UI 업데이트용)
    lane_message = pyqtSignal(str)
    alert_message = pyqtSignal(str)
    obj_message = pyqtSignal(str)
    state_message = pyqtSignal(str)

    def __init__(self, label_width, label_height, parent=None):
        super().__init__(parent)
        self.label_width = label_width
        self.label_height = label_height
        self.running = True
        self.lane_msg = None
        self.obj_class = None

        self.MAX_WAIT_TIME = 0.3

        self.lane_color = {
            0: [0, 0, 0],
            1: [0, 0, 255],
            2: [0, 255, 255],
            3: [0, 255, 0],
            4: [255, 0, 0],
            5: [255, 0, 255],
        }

        self.lane_dist = None
        self.obj_dist = None
        self.prev_lane_mask = None
        self.prev_bbox = None
        self.bbox = None
        self.lane_vis = None
        self.obj_vis = None
        self.lane_mask = False
        self.obj_mask = False

        # DB 삽입을 위한 데이터 수집
        self.session_id = None
        self.session_start_time = datetime.now()
        self.object_id = None
        self.detected_time = None

    def colorize_mask_lane(self, mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for k, color in self.lane_color.items():
            color_mask[mask == k] = color
        return color_mask

    def update_msg(self, msg, cls_id, obj_time, obj_id, position=None, lane_mask = None):
        if msg is None or cls_id is None:
            return
        
        #if position is not None and position < 20:
            #print("position : ", position)
        if obj_id == 3 and position < 30:
            QTimer.singleShot(5000, lambda: self.alert_message.emit("사람이 있습니다. 정지."))
            QTimer.singleShot(5000, lambda: self.state_message.emit("정지"))
            if obj_id is not None:
                db_queue.put(("action_log", {
                    "object_id": obj_id,
                    "action_type_id": 2,
                    "performed_time": datetime.now(),
                    "delay": (datetime.now()-obj_time).total_seconds()
                }))

        # ✅ 1. 차선변경 여부 (수정하지 않음)
        if msg[0] == 1 and cls_id != 0:
            self.lane_message.emit("좌측 차선 변경 가능")
            if obj_id is not None:
                db_queue.put(("action_log", {
                    "object_id": obj_id,
                    "action_type_id": 6,
                    "performed_time": datetime.now(),
                    "delay": (datetime.now()-obj_time).total_seconds()
                }))
        elif msg[1] == 1 and cls_id != 0:
            self.lane_message.emit("우측 차선 변경 가능")
            if obj_id is not None:
                db_queue.put(("action_log", {
                    "object_id": obj_id,
                    "action_type_id": 7,
                    "performed_time": datetime.now(),
                    "delay": (datetime.now()-obj_time).total_seconds()
                }))
        elif cls_id in (10, 9, 8, 7):
            self.lane_message.emit("차선 변경 불가능")
            if obj_id is not None:
                db_queue.put(("action_log", {
                    "object_id": obj_id,
                    "action_type_id": 8,
                    "performed_time": datetime.now(),
                    "delay": (datetime.now()-obj_time).total_seconds()
                }))
        else:
            self.lane_message.emit("차선 변경 불가능")
            if obj_id is not None:
                db_queue.put(("action_log", {
                    "object_id": obj_id,
                    "action_type_id": 8,
                    "performed_time": datetime.now(),
                    "delay": (datetime.now()-obj_time).total_seconds()
                }))

        # ✅ 2. 정지선 (거리 20m 이내만 처리)
        if msg[2] == 1:
            self.lane_dist = estimate_lane_distance(lane_mask, 0.05)
            # print("lane_dist : ", self.lane_dist)
            if self.lane_dist is not None:
                if cls_id == 9 and self.lane_dist < 3:
                    self.state_message.emit("정지")
                    if obj_id is not None:
                        db_queue.put(("action_log", {
                            "object_id": obj_id,
                            "action_type_id": 2,
                            "performed_time": datetime.now(),
                            "delay": (datetime.now()-obj_time).total_seconds()
                        }))
                elif self.lane_dist > 3:
                    self.state_message.emit("주행 중")
                    if obj_id is not None:
                        db_queue.put(("action_log", {
                            "object_id": obj_id,
                            "action_type_id": 0,
                            "performed_time": datetime.now(),
                            "delay": (datetime.now()-obj_time).total_seconds()
                        }))


        # ✅ 3. 횡단보도 + 사람 (거리 20m 이내만 처리)
        if msg[3] == 1:
            if cls_id == 3:
                self.obj_dist = position
                if self.obj_dist is not None and self.obj_dist < 40:
                    self.alert_message.emit("횡단보도에 사람이 있습니다. 주의하세요.")
                    if obj_id is not None:
                        db_queue.put(("action_log", {
                            "object_id": obj_id,
                            "action_type_id": 1,
                            "performed_time": datetime.now(),
                            "delay": (datetime.now()-obj_time).total_seconds()
                        }))
                    if self.obj_dist < 30:
                        self.state_message.emit("정지")
                        if obj_id is not None:
                            db_queue.put(("action_log", {
                                "object_id": obj_id,
                                "action_type_id": 2,
                                "performed_time": datetime.now(),
                                "delay": (datetime.now()-obj_time).total_seconds()
                            }))
                else:
                    self.state_message.emit("주행 중")
            elif cls_id != 3:
                self.state_message.emit("주행 중")

        # ✅ 4. 객체별 (거리 20m 이내만 메시지 업데이트)
        if cls_id == 0:
            self.obj_dist = position
            if self.obj_dist is not None and self.obj_dist < 20:
                self.obj_message.emit(f"자동차 인식됨 (거리: {self.obj_dist:.2f}m)")
                if obj_id is not None:
                    db_queue.put(("action_log", {
                        "object_id": obj_id,
                        "action_type_id": 1,
                        "performed_time": datetime.now(),
                        "delay": (datetime.now()-obj_time).total_seconds()
                    }))

        elif cls_id == 1:
            self.obj_dist = position
            if self.obj_dist is not None and self.obj_dist < 30:
                self.alert_message.emit("어린이 보호구역 - 주의")
                self.obj_message.emit("어린이 보호구역")
                if obj_id is not None:
                    db_queue.put(("action_log", {
                        "object_id": obj_id,
                        "action_type_id": 4,
                        "performed_time": datetime.now(),
                        "delay": (datetime.now()-obj_time).total_seconds()
                    }))

        elif cls_id == 2:
            self.obj_dist = position
            if self.obj_dist is not None and self.obj_dist < 20:
                self.alert_message.emit("공사장 근처 - 주의")
                self.obj_message.emit("공사장")
                if obj_id is not None:
                    db_queue.put(("action_log", {
                        "object_id": obj_id,
                        "action_type_id": 1,
                        "performed_time": datetime.now(),
                        "delay": (datetime.now()-obj_time).total_seconds()
                    }))

        elif cls_id == 4:
            self.obj_dist = position
            if self.obj_dist is not None and self.obj_dist < 30:
                self.alert_message.emit("30km/h 이하로 주행")
                self.obj_message.emit("30km/h 속도제한 구간")
                if obj_id is not None:
                    db_queue.put(("action_log", {
                        "object_id": obj_id,
                        "action_type_id": 4,
                        "performed_time": datetime.now(),
                        "delay": (datetime.now()-obj_time).total_seconds()
                    }))

        elif cls_id == 5:
            self.obj_dist = position
            if self.obj_dist is not None and self.obj_dist < 30:
                self.alert_message.emit("50km/h 이하로 주행")
                self.obj_message.emit("50km/h 속도제한 구간")
                if obj_id is not None:
                    db_queue.put(("action_log", {
                        "object_id": obj_id,
                        "action_type_id": 5,
                        "performed_time": datetime.now(),
                        "delay": (datetime.now()-obj_time).total_seconds()
                    }))

        elif cls_id == 6:
            self.obj_dist = position
            if self.obj_dist is not None and self.obj_dist < 20:
                self.obj_message.emit("정지 표지판 인식됨")
                self.state_message.emit("정지")
                if obj_id is not None:
                    db_queue.put(("action_log", {
                        "object_id": obj_id,
                        "action_type_id": 2,
                        "performed_time": datetime.now(),
                        "delay": (datetime.now()-obj_time).total_seconds()
                    }))
                QTimer.singleShot(5000, lambda: self.state_message.emit("주행 중"))

    def run(self):
        while self.running:
            now = time.time()
            # print(db_config.qsize())
            # if not db_config.empty():
            #     self.session_id = db_config.get_nowait()
            #     print(f"[VideoUpdateThread] session_id 획득: {self.session_id}")

            # 프레임 수신
            if not udp_video_queue.empty():
                udp_data = udp_video_queue.get_nowait()
                uuid = udp_data[0]
                frame = udp_data[1]
                orig_frame[uuid] = frame
                frame_time[uuid] = now

            while not lane_result_queue.empty():
                lane_data = lane_result_queue.get_nowait()
                lane_uuid = lane_data[0]
                lane_result = lane_data[1]
                self.lane_msg = lane_data[2]
                lane_mask[lane_uuid] = lane_result

            while not obj_result_queue.empty():
                obj_data = obj_result_queue.get_nowait()
                obj_uuid = obj_data[0]
                obj_result = obj_data[1]
                self.obj_class = obj_data[2]
                obj_mask[obj_uuid] = obj_result
                confidence = obj_data[3]
                self.bbox = obj_data[4]
                position = obj_data[5]
                self.detected_time = datetime.now()

                temp_queue.put([self.obj_class, self.detected_time, confidence, self.bbox, position])
                # ======================================================================
                # print("temp_queue.qsize() : ", temp_queue.qsize())
                # print(self.bbox)
                # position = estimate_obj_distance(self.obj_class, self.bbox)
                obj_mask[obj_uuid] = obj_result
                
                #print(self.obj_class)
                #print(type(self.obj_class))
                

            try:
                while self.object_id is None:
                    if not db_config.empty():
                        self.object_id = db_config.get_nowait()
                        #print(self.object_id)
                        break
                    else:
                        break
                
                if self.bbox is not None:
                    self.update_msg(self.lane_msg, self.obj_class, self.detected_time, self.object_id, position, lane_result)
                else:
                    self.update_msg(self.lane_msg, self.obj_class, self.detected_time, self.object_id, lane_result)
            except Exception as e:
                # print("update_msg 예외:", e)
                pass

            ready_uuids = sorted(orig_frame.keys())
            for uuid in ready_uuids:
                frame_age = now - frame_time.get(uuid, now)
                lane = lane_mask.get(uuid)
                obj = obj_mask.get(uuid)

                if lane is None or obj is None:
                    if frame_age < self.MAX_WAIT_TIME:
                        continue
                    else:
                        # print(f"[WARN] {uuid}: 마스크 지연 - {frame_age:.2f}s → 부분 처리 진행")
                        pass

                frame = orig_frame.pop(uuid)
                frame_time.pop(uuid, None)
                lane = lane_mask.pop(uuid, None)
                obj = obj_mask.pop(uuid, None)

                # 차선 마스크 컬러화
                if lane is not None and len(lane.shape) == 2:
                    try:
                        lane_vis = self.colorize_mask_lane(lane)
                        self.lane_mask = True
                    except Exception:
                        pass
                # 객체 마스크는 그대로 사용 (BGR 이미지라고 가정)
                if obj is not None:
                    try:
                        obj_vis = obj.copy()
                        self.obj_mask = True
                    except Exception:
                        pass
                # 마스크 합성
                if self.lane_mask and self.obj_mask:
                    combined_mask = cv2.addWeighted(lane_vis, 0.4, obj_vis, 0.6, 0.0)
                    frame = cv2.addWeighted(frame, 0.75, combined_mask, 0.25, 0.0)
                elif self.lane_mask:
                    frame = cv2.addWeighted(frame, 0.8, lane_vis, 0.2, 0.0)
                elif self.obj_mask:
                    frame = cv2.addWeighted(frame, 0.8, obj_vis, 0.2, 0.0)
                # else: 아무것도 안함

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                scaled = pixmap.scaled(self.label_width, self.label_height, Qt.AspectRatioMode.KeepAspectRatio)
                self.frame_ready.emit(scaled)

            expire_threshold = 2.0
            for mask_dict in (lane_mask, obj_mask):
                expired = [uuid for uuid in mask_dict if now - frame_time.get(uuid, now) > expire_threshold]
                for uuid in expired:
                    mask_dict.pop(uuid, None)

            self.lane_dist = None
            self.obj_dist = None
            self.lane_mask = False
            self.obj_mask = False
            self.object_id = None
    
    def stop(self):
        # drive_session table에 end_time, total_distance 내용 추가
        session_end_time = datetime.now()
        delta = session_end_time - self.session_start_time
        self.total_distance = 0.3 * delta.total_seconds()
        # print(session_end_time)
        # print(self.total_distance)
        db_queue.put(("update_session", {
            "session_id": self.session_id,
            "end_time": session_end_time,
            "total_distance": self.total_distance
        }))

        self.running = False
        self.quit()


# Main
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    manager = Manager()

    udp_video_queue = manager.Queue()

    lane_tcp_queue = manager.Queue()
    obj_tcp_queue = manager.Queue()

    lane_result_queue = manager.Queue()
    obj_result_queue = manager.Queue()

    db_queue = manager.Queue()
    db_config = manager.Queue()

    temp_queue = manager.Queue()

    df_orig = pd.DataFrame.from_dict(original_latency_check, orient='index')
    df_lane = pd.DataFrame.from_dict(lane_latency_check, orient='index')
    df_obj  = pd.DataFrame.from_dict(obj_latency_check, orient='index')

    df_merged = pd.concat([df_orig, df_lane, df_obj], axis=1)

    df_merged.to_csv("latency_summary.csv", index_label="uuid")

    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()
    sys.exit(app.exec())
