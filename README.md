# EDSR-PyTorch
Enhanced Deep Residual Networks for Single Image Super-Resolution

## [Abstract]
<center><img src="https://images.velog.io/images/heaseo/post/039f88b7-4a06-44a6-849c-2f8a3def9a67/Screen%20Shot%202021-06-08%20at%2012.48.48%20PM.png"></center>

<center><img src="https://images.velog.io/images/heaseo/post/1290bf71-0e18-4b28-a74b-2ce28b8ffea7/Screen%20Shot%202021-06-08%20at%2012.50.19%20PM.png"></center>


최근 초해상화 관련 논문은 deep convolutional neural networks (DCNN)을 이용한 모델들이 많았다. 특히, residual learning 기술들이 초해상화 성능 향상에 크게 기여했다. 이 논문이 발표하는 enhanced deep super-resolution network (EDSR)는 논문을 발표한 시점의 다른 state-of-the-art 초해상화 기법보다 월등히 높은 성능을 낸다. 다른 모델보다 EDSR이 더 좋은 성능을 낼 수 있었던 중요한 원인은 ResNet에 필요 없는 모듈을 삭제하고 깊은 모델을 안정적으로 학습할 수 있었기 때문이었다. 이 논문에서 제안한 EDSR 모델은 이 당시 최고의 SR 모델들보다 뛰어난 benchmark 성능을 보여줬고 NTIRE2017Super-Resolution Challenge에서 우승도 했다.

[더 보기](https://velog.io/@heaseo/내멋대로해석하는Enhanced-Deep-Residual-Networks-for-Single-Image-Super-Resolution-A.K.A-EDSR)

<br>

## Usage
train.py

```bash
#Multi-gpus training
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --train-file ${train_dataset} --eval-file ${test_dataset} --outputs-dir ${weights-dir} --scale ${2,3,4} --num-feats ${number of features} --num-blocks ${number of blocks} --res-scale ${res_scale} --distributed 

#Single gpu training
CUDA_VISIBLE_DEVICES=0 python train.py --train-file ${train_dataset} --eval-file ${test_dataset} --outputs-dir ${weights-dir} --scale ${2,3,4} --num-feats ${number of features} --num-blocks ${number of blocks} --res-scale ${res_scale}
```

test.py
```bash
python test.py --weights-file ${best.pth} --image-file ${example.png} --scale ${2,3,4}
```

<br>

## Results

<center><img src="https://images.velog.io/images/heaseo/post/bd8ea304-4fe7-4dcb-9086-75df7e2281ee/result1.PNG"></center>
<center><img src="https://images.velog.io/images/heaseo/post/24151c77-288c-4963-9d3e-3984beec2f5c/result2.PNG"></center>