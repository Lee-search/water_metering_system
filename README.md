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
- 단계별로 실행 가능한 "CNN_training.ipynb" 와 자동실행 "script.py" 로 분류.
- [해당문서](/1_training_tflite_model/README.md) 참조 바랍니다.

# 2. Activate digitizer
* 2번 프로젝트는 학습된 CNN 모델을 이용한 숫자인식 과정입니다.
* [해당문서](/2_activate_digitizer/README.md) 참조 바랍니다.




