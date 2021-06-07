# SRCNN-PyTorch
Super-Resolution Using Deep Convolutional Networks

## [Abstract]
<center><img src="examples/SRCNN_architecture.png"></center>

이 논문은 최초로 딥 러닝을 이용한 새로운 super-resolution(초해상화) 기술을 제안한다. SR은 저해상도 이미지에서 고해상도 이미지를 복원하는 기술이다. 이 대단한 기술에도 고질적인 문제가 있는데, 하나의 입력에 대해 복수의 결과물이 나올 수 있는 해결하기 어려운 문제가 있다. 

SRCNN은 Deep convolutional neural network (CNN)으로 구성되어 있으며, 저해상도 이미지와 고해상도 이미지가 한 쌍으로 이루어져 지도학습을 하는 딥러닝기반 인공지능이다. 기존에 있던 방법과는 다르게 각 CNN층을 별도로 관리하고 최적화한다.

SRCNN은 가벼운 구조를 가지고 있었지만 논문발표 당시 품질과 성능은 최고였다.

[더 보기](https://velog.io/@heaseo/내멋대로해석하는Image-Super-Resolution-Using-Deep-Convolutional-Networks-A.K.A-SRCNN)

<br>

## Usage
train.py
```bash
python train.py --train-file ${train_dataset} --eval-file ${test_dataset} --outputs-dir ${weights-dir} --scale ${2,3,4}
```
test.py
```bash
python test.py --weights-file ${best.pth} --image-file ${example.png} --scale ${2,3,4}
```

<br>

## Results

<table>
    <tr>
        <td><center>Bicubic</center></td>
        <td><center>SRCNN</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="examples/butterfly_bicubic_x3.bmp"></center>
    	</td>
    	<td>
    		<center><img src="examples/butterfly_SRCNN_x3.bmp"></center>
    	</td>
    </tr>
</table>
