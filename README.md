# Water Metering System

## 2022 산학실전캡스톤디자인 프로젝트
- 팀명: 체크잇(CHECK-IT!)
- 수요기업명: 한국에프디엑스네트웍스
- 기술명: 수도계량기 원격검침을 위한 IoT 디바이스 및 네트워크 개발

![wf](/image/workflow.png)

# 1. CNN training process
* 1번 프로젝트는 수도계량기의 이미지를 통한 CNN 학습과정을 담고 있습니다.
* 결과물은 /1.Training tf_model/output 에 출력되어 있습니다.

## Install package
* 아래의 [pip package](https://www.tensorflow.org/install/pip) 를 다운로드

```
$ pip install tensorflow
$ pip install matplotlib
$ pip install -U scikit-learn
$ pip install numpy
```

## Process in CNN_training.ipynb

### 1. Preparing the training
- 파이선 라이브러리 로드

### 2. Load training data
- [학습용 데이터](https://github.com/jomjol/neural-network-digital-counter-readout/tree/master/ziffer_sortiert_resize) 다운로드
- 데이터셋 정제

### 3. Make CNN Model
- CNN Model 생성

### 4. Training
- Tensorflow 를 통한 모델학습

### 5. Result (Plot)
- Matplotlib 을 통한 결과 Loss 와 Accuracy 출력

### 6. Save the model
- 학습된 모델 저장
- tf -> tflite 모델 변환 과정 진행

### 7. Check accuracy with each Image
- 실제 데이터와 검증

## 2. Digit providing in water meter