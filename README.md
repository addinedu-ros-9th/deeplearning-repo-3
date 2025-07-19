# 🚗 K-Urban AutoDrive feat. Don't Crash
**도심형 자율주행 보조를 위한 딥러닝 보조 시스템**  
한국 도심 환경에 특화된 자율주행 시스템의 핵심 기술 개발 프로젝트

---

## 시연 영상
![img](https://youtu.be/y2GF0t9XZx4)

---

## 핵심 특징 요약
- 한국 도심 환경을 반영한 자율주행 시뮬레이션 트랙 제작
- YOLO 기반 딥러닝 모델을 활용한 교통 실시간 객체 인식
- 커스텀 CNN 기반 딥러닝 모델을 활용한 실시간 차선 인식
- 차선 정보 및 경고 메시지를 포함한 실시간 GUI 시각화 제공
- 서버-클라이언트 기반 실시간 영상 송수신 및 추론 결과 전송 구조 구축
- 객체 감지 및 주행 동작 기록을 위한 데이터베이스 구조 설계
- 정지선, 표지판, 교차로 등 다양한 시나리오 기반 테스트 수행
  
---

## 목차
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Team Information](#team-information)  
4. [Development Environment](#development-environment)  
5. [Design](#design)  
    5.1. [User Requirements](#user-requirements)  
    5.2. [System Requirements](#system-requirements)  
    5.3. [System Architecture](#system-architecture)  
    5.4. [Interface Specification](#interface-specification)  
    5.5 [Data Structure](#data-structure)  
    5.6 [Scenario](#scenario)  
    5.7. [GUI Configuration](#gui-configuration)  
    5.8. [Test Case](#test-case)  
6. [Limitations](#limitations)  
7. [Conclusion and Future Works](#conclusion-and-future-works)

---

## 1.Overview

**K-Urban AutoDrive**는 한국 도심 환경에 최적화된 자율주행 보조 시스템으로, 복잡한 교통 상황에서도 안전하고 효율적인 주행을 지원하기 위해 개발되었습니다.

본 시스템은 실제 한국 도심을 반영한 시뮬레이션 환경을 기반으로, 딥러닝 기반의 객체 인식 및 차선 인식 모델을 통해 실시간 주행 판단을 수행합니다.  
운전자 보조를 위한 GUI는 인식된 객체와 경고 정보를 직관적으로 시각화하며, 서버-클라이언트 구조는 낮은 지연 시간으로 실시간 데이터 송수신을 가능하게 합니다.

또한 감지된 객체, 수행된 동작, 주행 세션 등의 정보는 데이터베이스에 저장되어 향후 분석 및 개선에 활용될 수 있으며, 다양한 주행 시나리오 테스트를 통해 시스템의 실효성과 안정성을 입증하였습니다.

---
## 2.Key Features
1. **한국 도심 환경 기반 시뮬레이션 트랙 구현**  
   - 차선, 횡단보도, 신호등, 장애물, 도로 표지판 등을 반영한 **맞춤형 시뮬레이션 트랙** 제작  
   - 실제 한국 교통 환경을 반영한 다양한 테스트 시나리오 구현  

2. **객체 인식 딥러닝 모델 개발 및 통합**  
   - 차량, 사람, 공사 고깔, 신호등, 도로 표지판 등을 **실시간으로 인식**하기 위해 **YOLOv8m detection 모델** 적용  
   - 다양한 객체에 대한 개별 모델 대신, YOLOv8m 기반의 **통합 객체 인식 모델**을 구축하여 **리소스 효율성과 유지 보수성 향상**  
   - 실제 도심 맵 영상 라벨링을 통해 학습 데이터셋 보강 및 도심 환경 특화 성능 확보  

3. **차선 인식 딥러닝 모델 개발**  
   - 복잡한 교차로 및 다양한 차선 환경에서도 동작 가능한 **커스텀 CNN 모델 설계 및 적용**  
   - 외부 데이터셋(CULane 등) 활용과 자체 실험을 통해 **한국 도심에 적합한 차선 인식 성능 최적화**

4. **실시간 주행 보조 시스템 구현**  
   - 인식된 객체 및 차선 정보를 기반으로 **운전 상황에 맞는 실시간 주행 안내 메시지 출력**  
   - 예: "정지선 정지", "장애물 주의", "속도 제한 30", "좌회전 가능" 등 직관적 시각적 메시지 제공  

5. **지연 최소화된 전체 파이프라인 설계**  
   - **카메라 입력 → 추론 처리 → GUI 시각화 출력** 전체 흐름의 지연 시간을 최소화하여 실시간 운전자 보조에 적합  
   - 시스템 병렬화 및 최적화된 통신 구조로 처리 속도 개선  

6. **데이터 기반 주행 기록 관리 시스템**  
   - 주행 중 감지된 객체, 수행된 동작, 운전 회차 등의 로그를 **데이터베이스에 기록**  
   - 향후 주행 이력 분석, 모델 개선, 시나리오 기반 테스트 등에 활용 가능
     
---

## 3.Team Information

| 이름     | 구분   | 역할 및 담당 업무 |
|----------|--------|------------------|
| **이동훈** | 팀장   | - 차선 인식 모델 개발<br>- 서버-클라이언트 통신 구축 |
| **김진언** | 팀원   | - 차선 인식 모델 개발<br>- 서버-클라이언트 통신 구축<br>- GUI 설계 |
| **이동연** | 팀원   | - 장애물(차, 사람, 트래픽콘) 데이터셋 수집<br>- 객체 통합 모델 개발 및 최적화 |
| **유영훈** | 팀원   | - 신호등 데이터셋 수집<br>- 객체 통합 모델 개발<br>- GitHub README 제작 |
| **김태호** | 팀원   | - DB 설계 및 구축<br>- 도로 표지판 데이터셋 수집<br>- 객체 통합 모델 개발<br>- 발표자료 제작 |

---

## 4.Development Environment

| 분류           | 사용 기술 |
|----------------|-----------|
| **개발 환경**      | ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=flat&logo=ubuntu&logoColor=white) ![VSCode](https://img.shields.io/badge/VSCode-007ACC?style=flat&logo=visual-studio-code&logoColor=white) |
| **언어**           | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| **데이터베이스**   | ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat&logo=mysql&logoColor=white) |
| **형상 관리**      | ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) |
| **협업 도구**      | ![Confluence](https://img.shields.io/badge/Confluence-172B4D?style=flat&logo=confluence&logoColor=white) ![Jira](https://img.shields.io/badge/Jira-0052CC?style=flat&logo=jira&logoColor=white) ![Slack](https://img.shields.io/badge/Slack-4A154B?style=f)

---

## 5.Design

### 5.1.User Requirements
| UR ID   | Requirements                                                    | Priority |
|---------|------------------------------------------------------------------|----------|
| UR_01   | 차선 인식 결과를 자동으로 운전자한테 알려줘야 한다.              | HIGH     |
| UR_02   | 신호등 인식 결과를 자동으로 운전자한테 알려줘야 한다.            | HIGH     |
| UR_03   | 장애물에 대한 대응을 운전자한테 알려줘야 한다.                   | MEDIUM   |
| UR_04   | 표지판에 따른 안내 메시지를 운전자한테 알려줘야 한다.            | MEDIUM   |
| UR_05   | 차선 변경 가능 여부를 운전자한테 알려줘야 한다.                  | LOW      |
| UR_06   | 갑작스러운 장애물에 대응할 수 있도록 알려줘야 한다.             | LOW      |

### 5.2.System Requirements

| Group           | SR_ID   | SR Name             | Description                                                                                   | Priority |
|----------------|---------|---------------------|-----------------------------------------------------------------------------------------------|----------|
| 인식 기능       | SR_01   | 신호등 인식         | 신호등의 현재 상태를 인식하는 기능<br>∙ 적색불<br>∙ 청색불<br>∙ 황색불<br>∙ 좌회전 신호<br>∙ 우회전 신호 (option) | H |
| 인식 기능       | SR_02   | 표지판 인식         | 표지판을 인식하는 기능<br>∙ STOP<br>∙ 속도 제한: 30, 50<br>∙ 어린이 보호구역                     | H |
| 인식 기능       | SR_03   | 차선 인식           | 안전 주행을 위한 도로 위 차선을 인식<br>∙ 실선<br>∙ 점선<br>∙ 중앙분리선<br>∙ 정지선<br>∙ 횡단보도 | H |
| 인식 기능       | SR_04   | 장애물 인식         | 도로 위 장애물 인식 기능<br>∙ 정적 장애물: 공사장<br>∙ 동적 장애물: 차량, 오토바이, 사람         | H |
| 주행 보조 기능   | SR_05   | 서행 알림           | 특정 표지판 및 신호등 상황에서 서행 안내<br>∙ 속도 제한, 어린이 보호구역<br>∙ 황색불              | M |
| 주행 보조 기능   | SR_06   | 일반 주행 알림      | ‘주행 중’ 알림 표시 기능                                                                       | H |
| 주행 보조 기능   | SR_07   | 정지 알림           | 정지 표지판 인식 시 정지 알림 기능                                                              | M |
| 주행 보조 기능   | SR_08   | 급정지 알림         | 갑작스러운 장애물에 대한 급정지 알림 기능                                                       | H |
| 주행 보조 기능   | SR_09   | Stop and Go 알림   | 특정 객체 인식 시 사라질 때까지 정지 유지<br>∙ 예: 신호등에서 건너는 보행자                      | H |
| 모니터링 기능    | SR_10   | 영상 모니터링 기능  | 실시간 주행 영상에 인식된 객체를 표시해 운전자가 모니터링 가능                                 | H |
| 모니터링 기능    | SR_11   | 주행 영상 저장 기능 | 실시간 주행 영상 데이터를 저장하는 기능                                                         | L |
| 모니터링 기능    | SR_12   | 객체 탐지 로그 저장 | 객체 탐지 로그를 실시간으로 저장하는 기능                                                       | H |
| 경고 기능       | SR_13   | 시각적 경고 기능    | 주의 객체 탐지 시 경고 메시지로 운전자에게 시각적 경고 제공                                     | H |
| 경고 기능       | SR_14   | 청각적 경고 기능    | 주의 객체 탐지 시 경고음 또는 음성 메시지로 운전자에게 청각적 경고 제공                         | L |

### 5.3.System Architecture
#### Hardware Architecture
![hardware_architecture](https://github.com/addinedu-ros-9th/deeplearning-repo-3/blob/main/images/hardware_architecture.png)
#### Software Architecture
![software_architecture](https://github.com/addinedu-ros-9th/deeplearning-repo-3/blob/main/images/software_architecture.png)

### 5.4.System Scenario
#### 신호등 보면서 직진 또는 정지
![scenario1](https://github.com/addinedu-ros-9th/deeplearning-repo-3/blob/main/images/scenario1.png)
#### 도로 표지판 감지
![scenario2](https://github.com/addinedu-ros-9th/deeplearning-repo-3/blob/main/images/scenario2.png)
#### 차선 변경
![scenario3](https://github.com/addinedu-ros-9th/deeplearning-repo-3/blob/main/images/scenario3.png)
#### 교차로에서 좌회전
![scenario4](https://github.com/addinedu-ros-9th/deeplearning-repo-3/blob/main/images/scenario4.png)
#### 장애물(사람,차) 감지
![scenario5](https://github.com/addinedu-ros-9th/deeplearning-repo-3/blob/main/images/scenario5.png)

### 5.5.Interface Specification
#### COVAv2 Controller -> COVAv2 Lane/Object Server

| No.  | Sender             | Receiver               | Sort | UUID         | Frame                | Description                     |
|------|--------------------|------------------------|------|--------------|----------------------|---------------------------------|
| -    | -                  | -                      | -    | 4 bytes      | `len(buffer.tobytes)` | 프레임 고유 식별자 및 바이트 길이 |
| F_1  | COVAv2 Controller  | COVAv2 Lane/Obj Server | UDP  | Frame UUID   | Encoded JPEG bytes   | 각 모델로 처리할 프레임 전송     |

#### COVAv2 Lane Server -> COVAv2 Controller

| No.   | Sender              | Receiver            | Sort | Header            | UUID         | Pred_Mask             | Description                  |
|-------|---------------------|---------------------|------|--------------------|--------------|------------------------|------------------------------|
| -     | -                   | -                   | -    | 4 bytes            | 4 bytes      | JSON                   | 패킷 헤더 정의                |
| LB_1  | COVAv2 Lane Server  | COVAv2 Controller   | TCP  | length of JSON file | Frame UUID  | base64 encoded PNG     | Results of lane detection    |

#### COVAv2 Object Server -> COVAv2 Controller

| No.   | Sender                | Receiver            | Sort | Header              | UUID         | JSON Structure                                                                                                     | Description                  |
|-------|------------------------|---------------------|------|----------------------|--------------|---------------------------------------------------------------------------------------------------------------------|------------------------------|
| -     | -                      | -                   | -    | 4 bytes              | 4 bytes      | -                                                                                                                   | 패킷 헤더 정의                |
| OB_1  | COVAv2 Object Server   | COVAv2 Controller   | TCP  | length of JSON file  | Frame UUID   | `{"id": int, "name": str, "confidence": float, "bbox": [x1, y1, x2, y2]}`                                           | 객체 인식 결과 전송          |

### 5.6.Data Structure
![data_structure](https://github.com/addinedu-ros-9th/deeplearning-repo-3/blob/main/images/data_structure.png)

### 5.7.GUI Configuration
![gui_configuration](https://github.com/addinedu-ros-9th/deeplearning-repo-3/blob/main/images/gui_configuration.png)

### 5.8.Test Case

| No.   | Test List                                                                 | Pass / Fail |
|-------|---------------------------------------------------------------------------|-------------|
| tc1   | COVAv2 controller에서 영상 프레임 단위 송신                                 | ✅          |
| tc2   | COVAv2 Lane Server에서 수신한 프레임 추론 후 controller로 결과 송신        | ✅          |
| tc3   | COVAv2 Obj Server에서 수신한 프레임 추론 후 controller로 결과 송신         | ✅          |
| tc4   | 송수신한 프레임 동기화                                                    | ✅          |
| tc5   | 수신받은 결과를 후처리한 후 정상 시각화                                   | ✅          |
| tc6   | 차량 인식 후 차선 변경 불가능 메시지 출력                                  | ✅          |
| tc7   | 정지선과 적색불 인식 후 정지선과 충분히 가까워지면 정지 메시지 출력       | ✅          |
| tc8   | 청색불 인식 시 다시 주행중 메시지 출력                                    | ✅          |
| tc9   | 속도 제한 50 표지판 인식 후 “50 이하 속도 유지” 메시지 출력               | ✅          |
| tc10  | 어린이 보호구역 / 속도 제한 30 표지판 인식 후 메시지 출력                 | ✅          |
| tc11  | 정지 표지판 인식                                                          | ✅          |
| tc12  | 횡단보도 인식                                                             | ✅          |
| tc13  | 사람이 충분히 가까워지면 정지 메시지 출력                                 | ❌          |
| tc14  | 사람이 시야에서 없어지면 다시 주행중 메시지 출력                          | ❌          |
| tc15  | 횡단보도 위 보행자 인식                                                  | ✅          |
| tc16  | 차량이 시야에서 벗어나고 점선 인식 시 차선변경 가능 메시지 출력           | ✅          |
| tc17  | 차선 변경 후 좌회전 신호 인식                                             | ✅          |
| tc18  | 공사장 칼라콘 인식 시 “공사장 - 주의” 메시지 출력                         | ✅          |
| tc19  | 황색불 인식                                                               | ✅          |

## 6.Deeplearning Model Development
### 6.1. 차선 인식 모델
#### YOLOv8n-seg
| 테스트 번호 | 내용       | 설정  | 성능 지표                      | 평가 및 문제점                         |
|-------------|------------|-------|-------------------------------|----------------------------------------|
| 1           | 기본 학습   | 기본  | seg-loss = 0.5<br>mAP50 = 0.3 | 성능 부족, 자율주행 활용 어려움       |
#### YOLOv8m-seg
| 테스트 번호 | 내용                         | 설정                                               | 성능 지표                            | 평가 및 문제점                                        |
|-------------|------------------------------|----------------------------------------------------|-------------------------------------|--------------------------------------------------------|
| Test1       | 기본 학습                    | 기본 설정                                          | seg-loss = 0.4<br>mAP50 = 0.8       | 나노 모델 대비 성능 개선, 다른 영상 적용 시 인식률 부족 |
| Test2       | 학습률 조정                  | lr0 = 0.003, lrf = 0.0005                          | seg-loss = 0.5<br>mAP50 = 0.87      | Test1보다 수치 개선, 다른 영상 적용 시 인식률 부족      |
| Test3       | 학습률 및 optimizer 변경     | optimizer = SGD,<br>lr0 = 0.003,<br>lrf = 0.0001   | seg-loss = 0.4935<br>mAP50 = 0.566  | Test1, Test2보다 성능 부족                            |
#### custom UFLD

| 테스트 번호 | 내용                                 | 설정                                        | 성능 지표                           | 평가 및 문제점                     |
|-------------|--------------------------------------|---------------------------------------------|------------------------------------|------------------------------------|
| Test1       | MSE + BCE loss 함수                  | epoch = 5,<br>loss 함수 = MSE + BCE         | Accuracy = 0.8<br>Precision = 0.7  | 차선 두께 문제로 학습 부진         |
| Test2       | Smooth L1 + BCE loss 함수, 데이터셋 수정 | epoch = 5,<br>loss 함수 = Smooth + BCE      | Accuracy = 0.89<br>Precision = 0.81 | 데이터셋 재정비 필요                |
#### custom CNN
| 테스트 번호 | 내용                              | 설정                                           | 성능 지표         | 평가 및 문제점       |
|-------------|-----------------------------------|------------------------------------------------|------------------|----------------------|
| test1       | UNet1 + Cross Entropy loss 함수   | epoch = 5,<br>loss 함수 = cross entropy        | mean IoU = 79%   | 개선 여지 있음       |
| test2       | UNet2 + Dice + Focal loss 함수    | epoch = 5,<br>loss 함수 = Dice + Focal         | mean IoU = 89%   | 가장 우수한 성능     |
| test3       | UNet2 + Focal + Tversky loss 함수 | epoch = 5,<br>loss 함수 = Focal + Tversky      | mean IoU = 86%   | 괜찮은 성능          |
#### 최종 모델
custom CNN(UNet2 + Dice + Focal loss 함수)
### 6.2. 객체 인식 모델
#### 6.2.1.Obstacle Detection Model
##### YOLOv8n-seg

| 테스트 번호 | 내용                          | 설정                                                                 | 성능 지표                                                                 | 평가 및 문제점   |
|-------------|-------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------|------------------|
| test1       | 기본 학습                     | 기본 설정                                                            | seg-loss = 1.57<br>mAP50 = 0.5<br>mAP50-95 = 0.28                         | 부족한 성능      |
| test2       | 파라미터 수정 및 데이터 증강 | epochs = 150,<br>patience = 30,<br>optimizer = SGD,<br>lr0 = 0.005,<br>warmup_epochs = 3 | seg-loss = 1.34<br>mAP50 = 0.592<br>mAP50-95 = 0.321                     | 부족한 성능      |
##### YOLOv8m-seg
| 테스트 번호 | 내용                          | 설정                                                                 | 성능 지표                                                    | 평가 및 문제점                      |
|-------------|-------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------|-------------------------------------|
| test1       | 기본 설정                     | 기본 설정                                                            | seg-loss = 1.124<br>mAP50 = 0.81<br>mAP50-95 = 0.49            | YOLOv8n-seg 대비 성능 개선, 괜찮은 성능 |
| test2       | 추가 데이터 확보              | 기본 설정                                                            | seg-loss = 1.018<br>mAP50 = 0.859<br>mAP50-95 = 0.564          | 괜찮은 성능                         |
| test3       | 파라미터 수정 및 데이터 증강 | epoch = 150,<br>patience = 20,<br>optimizer = SGD,<br>lr0 = 0.0001   | seg-loss = 0.9335<br>mAP50 = 0.904<br>mAP50-95 = 0.611         | 가장 우수한 성능                    |

#### 6.2.2.Traffic Sign Detection Model
##### YOLOv8n
| 테스트 번호 | 내용                                                                 | 설정     | 성능 지표                          | 평가 및 문제점                                                   |
|-------------|----------------------------------------------------------------------|----------|------------------------------------|------------------------------------------------------------------|
| Test1       | 기본 학습 진행<br>- 클래스: 속도제한, 어린이보호구역, 정지             | default  | mAP50 = 0.97<br>box_loss = 0.36    | 모양 인식 성공, 유사한 모양의 다른 표지판도 함께 탐지되는 문제 |
| Test2       | 클래스 재분류<br>- 클래스: 속도제한30, 속도제한50, 어린이보호구역, 정지 | default  | mAP50 = 0.94<br>box_loss = 0.44    | 모양 인식 성공, 속도제한 클래스 구분 성공                       |

#### 6.2.3.Traffic Light Detection Model
##### YOLOv8n

| 테스트 번호 | 내용                    | 설정   | 성능 지표                               | 평가 및 문제점 |
|-------------|-------------------------|--------|------------------------------------------|----------------|
| test1       | yolov8n detection 모델  | 기본   | mAP50 = 0.5<br>mAP50-95 = 0.311          | 부족한 성능    |

##### YOLOv8m

| 테스트 번호 | 내용                                                                                           | 설정   | 성능 지표                              | 평가 및 문제점                                |
|-------------|------------------------------------------------------------------------------------------------|--------|-----------------------------------------|------------------------------------------------|
| test1       | yolov8m detection 모델                                                                         | 기본   | mAP50 = 0.59<br>mAP50-95 = 0.39         | yolov8n보단 낫지만 부족한 성능                |
| test2       | yolov8m에서 클래스 12개 → 4개 축소,<br>가로 세로의 2%보다 작은 bbox 제거,<br>클래스별 bbox 균등 조절 데이터셋으로 학습 | 기본   | mAP = 0.96<br>mAP50-95 = 0.80           | 뛰어난 성능, 다른 영상에 적용 시 인식률 부족  |
#### 6.2.4.Object Detection Model(통합 모델)
#####  독립 모델 사용의 한계
- 각 기능별로 개별 딥러닝 모델(CNN 기반 차선 인식, YOLO 기반 객체 인식 등)을 운영할 경우,
- **실시간 주행 시스템에서 과도한 연산 리소스(CPU/GPU)와 메모리 사용량**이 발생
- 특히 임베디드 환경 및 다중 입력 영상 처리 시 병목 발생

#####  통합 모델 적용
- 위 문제를 해결하기 위해 **YOLOv8m 기반의 통합 객체 인식 모델**을 채택
- 기존의 YOLOv8n 대비 성능 우수 (`mAP50=0.96`, `mAP50-95=0.80`)
- 다양한 객체(신호등, 보행자, 차량, 표지판 등)를 **단일 모델에서 효율적으로 인식** 가능



## 7.Limitations

1. **극단적 기상 조건에서 성능 저하**  
   - 폭우, 안개, 눈 등 기상 상황 악화 시 시야 확보가 어려워 인식 정확도가 크게 저하됨

2. **복잡한 교차로 및 다차선 도로 환경에서 인식률 감소**  
   - 신호등/표지판의 위치가 명확하지 않거나 여러 개일 경우, 잘못된 판단을 유발할 수 있음

3. **야간 또는 역광 상황에서 카메라 기반 인식 성능 저하**  
   - 조명이 부족하거나 강한 빛(예: 역광)으로 인해 객체 또는 차선 검출 정확도 저하

4. **훈련 데이터셋과 실제 환경의 차이**  
   - 실제 도심 환경과의 불일치로 인해 추론 결과가 불안정해질 수 있음
     
## 8.Conclusion and Future Works

### Conclusion
- 한국 도심 환경에 최적화된 **차선 인식 딥러닝 모델**을 성공적으로 개발하였으며, 복잡한 교차로와 다양한 도로 조건에서도 안정적인 인식 성능을 확보함
- 실시간 영상 프레임의 **전송 → 추론 → GUI 송출**까지 연결되는 전체 파이프라인에서 지연을 최소화하여 실시간성 요구를 충족함
- 일부 테스트에서 기상 변화(예: 흐림, 비, 역광 등)에 의한 성능 저하가 관찰되었으며, **기상 변화에 강인한 모델 개발은 향후 과제로 남아 있음**

### Future Works
- **신호등/표지판 등 다양한 교통 인프라 객체의 데이터셋 확대**를 통해 통합 모델의 범용성과 정확도를 향상
- **GPS 기반 위치 정보와 딥러닝 인식 결과를 융합**하여 객체 인식 및 주행 판단의 정확도 향상 추진
- **기상 조건 변화(폭우, 안개 등)에 강건한 인식 성능 확보**를 위한 추가 학습 및 데이터 보강 계획
- 개발된 소프트웨어를 **임베디드 하드웨어와 통합**하여 실제 자율주행 차량에 탑재, **도심 환경 실증 테스트** 진행 예정
