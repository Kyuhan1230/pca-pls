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
<img src="/imgs/scoreplot.png" width="300" height="300">

2. Loading Plot
<img src="/imgs/loadingplot.png" width="300" height="300">

3. PLS Predict: SCatter Plot
<img src="/imgs/plsscatterplot.png" width="300" height="300">

4. PLS VIP(Variable Importance in Projection)
<img src="/imgs/plsvip.png" height="300">

5. Fault/Anomaly Detection - Hotelling's T2
<img src="/imgs/pcat2.png" height="300">

6. Fault/Anomaly Detection - SPE/DMODX
<img src="/imgs/pcaspe.png" height="300">

7. Fault Identify - Contribution
<img src="/imgs/pcacontribution.png" height="300">

