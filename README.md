# pca-pls
PCA와 PLS의 데이터 분석 실무에 활용 버전 패키지입니다.

PCA와 PLS는 차수감축의 대표적인 모델입니다.
<img width="1364" alt="image" src="https://user-images.githubusercontent.com/80809187/215822198-163b54ea-3781-4635-a509-12a6d95d8c20.png">

제조업 데이터 내에는 많은 변수가 존재합니다. 따라서 PCA를 통해 고차원의 데이터를 저차원의 데이터로 재구성할 수 있습니다.
또한 제조업 데이터는 다양한 노이즈를 갖고 있습니다. PLS는 차수감축을 통해 노이즈에 유연하게 대응(예측)할 수 있습니다.

데이터 분석 직무를 수행하면서 필요한 기능들을 모아 패키지로 만들었습니다.

### Import pca package
```
from pcaModelling import pcaModel
```

### Import pls package
```
from plsModelling import plsModel
```

### Example
1. Score Plot
![image](https://user-images.githubusercontent.com/80809187/215824223-72a625b1-0c4b-4af4-a80f-74366e0329a2.png)

2. Loading Plot
![image](https://user-images.githubusercontent.com/80809187/215824263-26d5ee77-ccbf-4429-a533-f31a79f9545b.png)

3. PLS Predict: SCatter Plot
![image](https://user-images.githubusercontent.com/80809187/215824355-60cf7635-f4c1-4c65-8a6b-76bff659ec01.png)

4. PLS VIP(Variable Importance in Projection)
![image](https://user-images.githubusercontent.com/80809187/215824456-5c026370-7f86-4c0f-9b07-9d8092527b36.png)

5. Fault/Anomaly Detection - Hotelling's T2
![image](https://user-images.githubusercontent.com/80809187/215824586-fa2bdafd-3394-49a2-8cb1-0acc94f15415.png)

6. Fault/Anomaly Detection - SPE/DMODX
![image](https://user-images.githubusercontent.com/80809187/215824631-277271a9-7894-40f3-96e6-2abe62a82ad1.png)

7. Fault Identify - Contribution
![image](https://user-images.githubusercontent.com/80809187/215824688-3f43e62f-deb7-4631-b739-9d2f5beccb7b.png)

