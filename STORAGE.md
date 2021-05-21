# 結果置き場
## ごちゃごちゃ出た結果を雑に置いていくページ
### max_span=100


### 1024切
![](https://i.imgur.com/OEJh39V.png)
1024切　固定長　層広め
```
train_dataset, val_dataset = [750000, 50000]
Fixed(num_layer=32, mid_units=800, num_filters=256, length=1024）
Adam(lr=1e-3, weight_decay=1e-6, eps=1e-5)
Epoch 20/20
train Loss:3.0447 Timer:4340.012
val Loss:26.2889 Timer:72.4072
```
![](https://i.imgur.com/jCP0omq.png)
1024切 固定長　カーネル大きめ=129
```
train_dataset, val_dataset = [900000, 100000]
Fixed(num_layer=4, kernel_sizes=129, length=1024, mid_units=800)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
```
![](https://i.imgur.com/zOJWRPA.png)
カーネル大きい=513
```
train_dataset, val_dataset = [900000, 100000]
Fixed(num_layer=4, kernel_sizes=513, length=1024, mid_units=800)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
```



![](https://i.imgur.com/yDXNXMM.png)
1024切　可変長　深めの層
```
loss2.57
train_dataset, val_dataset = [750000, 50000]
Variable(num_layer=64, num_filters=128）
Adam(lr=1e-3, weight_decay=1e-6, eps=1e-5)
epochs = 20
```
![](https://i.imgur.com/BfOZZeM.png)
1024切　可変長　カーネル大きめ＝129
```
train_dataset, val_dataset = [900000, 100000]
Variable(num_layer=4, kernel_sizes=129)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
```


### 固定長256切
![](https://i.imgur.com/zilUb4V.png)
256切　固定長　薄層=4
```
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=4, mid_units=500)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:2.39 Timer:341.08
val Loss:2.4030 Timer:25.1957
```

![](https://i.imgur.com/GejIkyJ.png)
256切　固定長　薄層=4 カーネルでかい=255
```
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=4, kernel_sizes=255, length=256）
Adam(lr=1e-4, weight_decay=1e-6, eps=1e-5)
Epoch 10/20
train Loss:1.5241 Timer:3695.796
val Loss:1.655 Timer:144.736
```

![](https://i.imgur.com/aW9u8jd.png)
256切 mid_unit少ない=100
```
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=4, mid_units=100)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:2.664 Timer:221.4
val Loss:2.6670 Timer:21.188
```


![](https://i.imgur.com/3uM1Iz5.png)
256切　固定長　線形層でかい=2000 8層
```
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=8, mid_units=2000, length=256)
Adam(lr=1e-4, weight_decay=1e-6, eps=1e-5)
Epoch 20/20
train Loss:0.9438 Timer:707.71
val Loss:1.5906 Timer:31.9186
```

![](https://i.imgur.com/06f06Ti.png)
256切 固定長　深め＝32
```
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=32, length=256, mid_units=400)
Adam( lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:1.550 Timer:1527.344
val Loss:1.565 Timer:59.9620
```
![](https://i.imgur.com/eLPgqkw.png)
256切　固定長　線形層でかめ1000＋深めの層32
```
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=32, mid_units=1000, length=256)
Adam(lr=1e-4, weight_decay=1e-6, eps=1e-5)
Epoch 11/20
train Loss:1.1655 Timer:1628.555
val Loss:1.202516 Timer:46.341
```
![](https://i.imgur.com/VfC2Ftz.png)
256切　固定長 激深=64+線形デカめ＝１０００
```
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=64, mid_units=1000, length=256）
Adam(lr=1e-4, weight_decay=1e-6, eps=1e-5)
Epoch 20/20
train Loss:1.12 Timer:3022.88
val Loss:1.119 Timer:100.85105
```




### 可変長256切
![](https://i.imgur.com/qk75vVL.png)
256切　可変長 薄層=4
```
train_dataset, val_dataset = [1800000, 200000]
Variable(num_layer=4, num_filters=128)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:1.7626 Timer:211.313
val Loss:1.765 Timer:22.8907
```
![](https://i.imgur.com/MyTGp2p.png)
256切 ちょい深層=16
```
train_dataset, val_dataset = [1800000, 200000]
Variable(num_layer=16, num_filters=128, flag=False)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:1.1906 Timer:743.21267
val Loss:1.1948 Timer:36.5172
```
![](https://i.imgur.com/ulBEPZ6.png)
256切 深層=32
```
train_dataset, val_dataset = [1800000, 200000]
Variable(num_layer=32, num_filters=128)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:0.991 Timer:1461.375
val Loss:0.995 Timer:62.663
```

![](https://i.imgur.com/83LsK4P.png)
![](https://i.imgur.com/LYezo88.png)
256切いい感じを目指したやつ
```
Epoch 20/20
train Loss:1.0014 Timer:1264.2033
val Loss:1.1118 Timer:37.6093
cor 0.8494499578
num_layer=16, kernel_sizes=33
```






### 固定長512切
![](https://i.imgur.com/8jsy9PO.png)
512切 固定長 8層
```
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:2.8441012 Timer:701.018939
val Loss:2.841616110 Timer:42.9393
```
![](https://i.imgur.com/15Ky0Lc.png)
512切り 固定長　32層
```
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=32, length=512)
Adam(lr=1e-4, weight_decay=1e-6, eps=1e-5)
Epoch 20/20
train Loss:1.038542952 Timer:2383.57
val Loss:1.082621844 Timer:88.128
```
![](https://i.imgur.com/8lueWP8.png)
512切 固定長 64層
```
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:2.561405782 Timer:4602.838
val Loss:2.63288931 Timer:155.179824
```

![](https://i.imgur.com/n8gAeb2.png)
```
Epoch 15/15
train Loss:2.6187 Timer:648.2103
val Loss:2.6231 Timer:32.2095
num_layer=16, mid_units=200, num_filters=64, length=512
```

### 512切可変長
![](https://i.imgur.com/hACRyaW.png)
512切 可変長 8層
```
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:1.430261 Timer:604.9288
val Loss:1.430779 Timer:37.11496
```
![](https://i.imgur.com/o51CkT3.png)
512切 可変長16層
```
Epoch 10/10
train Loss:1.1997 Timer:1138.7787
val Loss:1.2014 Timer:51.3031
```
<!-- ![](https://i.imgur.com/sju72Z9.png) -->
![](https://i.imgur.com/Uxv7dyZ.png)
512切　可変長 32層
```
Epoch 10/10
train Loss:0.9316 Timer:2258.5318
val Loss:0.9386 Timer:85.7601
```
![](https://i.imgur.com/9ape5pH.png)
512切 可変長 64層
```
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:0.97961795 Timer:4555.177
val Loss:0.99704 Timer:157.397568
```



![](https://i.imgur.com/UrKgmJD.png)
カーネル変化させようと
```
num_layer=4, kernel_sizes=7
Epoch 10/10
train Loss:1.7727 Timer:291.8791
val Loss:1.7693 Timer:30.6114
```
![](https://i.imgur.com/nNTfrHB.png)
```
num_layer=4, kernel_sizes=13
Epoch 10/10
train Loss:1.5316 Timer:337.8994
val Loss:1.5269 Timer:25.8738
```
![](https://i.imgur.com/g1AXWam.png)
```
num_layer=4, kernel_sizes=23
Epoch 10/10
train Loss:1.4036 Timer:435.7368
val Loss:1.4186 Timer:29.0192
```
![](https://i.imgur.com/3rU5U9o.png)
```
num_layer=4, kernel_sizes=65
Epoch 10/10
train Loss:1.3510 Timer:786.1192
val Loss:1.3611 Timer:37.4801
```
![](https://i.imgur.com/675EChg.png)
```
num_layer=4, kernel_sizes=129
Epoch 10/10
train Loss:1.3748 Timer:1300.1245
val Loss:1.4014 Timer:31.3718
```
![](https://i.imgur.com/W48yTyn.png)
```
kernel=511
Epoch 8/10
train Loss:1.4705 Timer:7184.2716
val Loss:1.4950 Timer:259.2312
```

filterを変化させようと
![](https://i.imgur.com/YVTU4hI.png)
```
Epoch 15/15
train Loss:1.3285 Timer:611.1924
val Loss:1.3361 Timer:32.8887
num_layer=16, num_filters=64
```
![](https://i.imgur.com/6jzAtfe.png)
```
Epoch 15/15
train Loss:1.2255 Timer:1113.9561
val Loss:1.2433 Timer:52.0594
num_layer=16, num_filters=128
```
![](https://i.imgur.com/9Y7JjpG.png)
```
足りないぶんやってみたらなんかおかしい
Epoch 15/15
train Loss:2.4965 Timer:2833.4412
val Loss:2.5534 Timer:96.2675
num_layer=16, num_filters=256
```
![](https://i.imgur.com/GCcQDWI.png)
```
Epoch 15/15
train Loss:1.1537 Timer:8900.3974
val Loss:1.2047 Timer:264.4484
num_layer=16, num_filter=512
```


学習率調整
Epoch 10/10
![](https://i.imgur.com/4bnciGZ.png)
train Loss:1.4978 Timer:266.2852
val Loss:1.5016 Timer:68.2257
lr=0.001, weight_decay=1e-6, eps=0.001

![](https://i.imgur.com/5KqwWsK.png)
train Loss:1.8951 Timer:321.6913
val Loss:1.8914 Timer:97.3850
lr=1e-05, weight_decay=1e-6, eps=0.001

![](https://i.imgur.com/Djrsnqh.png)
train Loss:2.1608 Timer:325.7285
val Loss:2.1252 Timer:100.7100
lr=1e-07, weight_decay=1e-6, eps=0.001

![](https://i.imgur.com/gHBqm40.png)
train Loss:1.4768 Timer:337.4948
val Loss:1.4841 Timer:98.2803
lr=0.001, weight_decay=1e-6, eps=1e-05

![](https://i.imgur.com/B4YQHLM.png)
train Loss:1.8411 Timer:364.9771
val Loss:1.8324 Timer:98.5535
lr=1e-05, weight_decay=1e-6, eps=1e-05

![](https://i.imgur.com/KCHydue.png)
train Loss:2.1290 Timer:364.5059
val Loss:2.1041 Timer:94.9194
lr=1e-07, weight_decay=1e-6, eps=1e-05

![](https://i.imgur.com/B7JTJmN.png)
train Loss:1.4859 Timer:370.4707
val Loss:1.4909 Timer:88.7173
lr=0.001, weight_decay=1e-6, eps=1e-07

![](https://i.imgur.com/Q6vWQyJ.png)
train Loss:1.8525 Timer:343.1903
val Loss:1.8449 Timer:74.2790
lr=1e-05, weight_decay=1e-6, eps=1e-07

![](https://i.imgur.com/wkTCKjv.png)
train Loss:2.1307 Timer:347.8610
val Loss:2.1024 Timer:76.0626
lr=1e-07, weight_decay=1e-6, eps=1e-07


![](https://i.imgur.com/dOTFLyP.png)
![](https://i.imgur.com/NLZWI79.png)
いい感じの塩梅を狙って, 層だけ浅目
```
データ[1800000, 200000]
num_layer=8, num_filters=256,  kernel_sizes=65
Epoch 15/15
train Loss:0.8906 Timer:5394.1276
val Loss:1.0919 Timer:56.3772
```



### 固定長256のデータ数増やしていくやつ(W=20)
![](https://i.imgur.com/BfHnsOd.png)
![](https://i.imgur.com/tih72x6.png)
Epoch 20/20
train Loss:0.3659 Timer:86.56
val Loss:0.5413 Timer:31.33531
0.87664
![](https://i.imgur.com/aCyY3x6.png)
![](https://i.imgur.com/OdCMKeV.png)
Epoch 20/20
train Loss:0.1532436 Timer:263.561
val Loss:0.183557 Timer:30.71315
0.9573426
![](https://i.imgur.com/oqV7pjV.png)
![](https://i.imgur.com/qwgXRTB.png)
Epoch 20/20
train Loss:0.109166 Timer:437.54109
val Loss:0.11557745 Timer:30.597293
0.97259675
![](https://i.imgur.com/m6UTfPO.png)
![](https://i.imgur.com/Hb3UiC7.png)
Epoch 20/20
train Loss:0.0878557 Timer:609.209
val Loss:0.0920469 Timer:30.49430
0.977990
![](https://i.imgur.com/YHSeJIL.png)
![](https://i.imgur.com/FVzUxqb.png)
Epoch 20/20
train Loss:0.076795 Timer:787.2779
val Loss:0.0799042 Timer:30.9559230
0.98136084288




固定長可変長比較
![](https://i.imgur.com/h8NCJaG.png)
```
可変長
num_layer=8, num_filters=128
Epoch 10/10
train Loss:1.4978 Timer:492.7626
val Loss:1.5023 Timer:84.9615
```
![](https://i.imgur.com/9QVIUn2.png)
```
固定長
Epoch 10/10
train Loss:2.2111 Timer:572.6530
val Loss:2.3022 Timer:88.0472
num_layer=8, num_filters=128, length=512
```
![](https://i.imgur.com/nzWqzzO.png)
```
可変長
num_layer=32, num_filters=128
Epoch 10/10
train Loss:1.1082 Timer:1871.1643
val Loss:1.1244 Timer:215.7853
```
![](https://i.imgur.com/UNuiRG3.png)
```
固定長
Epoch 10/10
train Loss:2.0359 Timer:1983.5330
val Loss:2.0574 Timer:221.1142
num_layer=32, num_filters=128, length=512
```


### データ増やして頑張ったけど，，，
学習500万データ 512切 W=100
![](https://i.imgur.com/Tmb3Ae3.png)
```
train_dataset, val_dataset =  [5000000, 500000]
Epoch 20/20
train Loss:0.8917 Timer:12045.6844
val Loss:0.9307 Timer:212.6935
num_layer=32, num_filters=128, kernel_sizes=33
big_data2.pth
```
いっしょ
![](https://i.imgur.com/XCR6BR4.png)
```
train_dataset, val_dataset[5000000, 500000])
Epoch 20/20
train Loss:1.0400 Timer:10907.1554
val Loss:1.0462 Timer:349.1202
num_layer=64, num_filters=128, kernel_sizes=5
```


W=20のMobileNet検証
DSCNN
![](https://i.imgur.com/GLTWg7h.png)
![](https://i.imgur.com/vOU56FS.png)
Total Tensors: 17765864 	Used Memory: 67.84M
The allocated memory on cuda:0: 67.74M
Memory differs due to the matrix alignment or invisible gradient buffer tensors
Epoch 20/20
train Loss:1.5776 Timer:1724.5514
val Loss:1.6303 Timer:58.1669
num_layer=64, num_filters=256, kernel_sizes=11


## ダウンサンプリングtrial
512塩基
180万学習+20万テスト


![](https://i.imgur.com/n0MTp2A.png)
maxpoolダウンサンプリング Tconvアップサンプリング
```
2conv+maxpool
convtransposed
num_layer=8, num_filters=128, kernel_sizes=7
Total Tensors: 61885360 	Used Memory: 236.09M
Epoch 15/20
train Loss:1.3214 Timer:3721.4693
val Loss:1.4827 Timer:145.0116
```
![](https://i.imgur.com/Ylb0Vax.png)
strideダウンサンプリングTconvアップサンプリング
```
2conv
convtransposed
num_layer=8, num_filters=128, kernel_sizes=7
Total Tensors: 61885360 	Used Memory: 236.09M
Epoch 20/20
train Loss:1.4975 Timer:3230.7465
val Loss:1.5527 Timer:126.6187
```

![](https://i.imgur.com/EL2b38m.png)
```
maxpool
upsample
Epoch 20/20
train Loss:1.5202 Timer:4664.4286
val Loss:1.7447 Timer:216.9864
num_layer=8, num_filters=128, kernel_sizes=7
```

![](https://i.imgur.com/kY6RP1Y.png)
```
maxpool
upsample(nearest)
Epoch 15/20
train Loss:2.3705 Timer:1848.0404
val Loss:2.3645 Timer:67.0252
```

![](https://i.imgur.com/ZwDEYAS.png)
dilationダウンサンプリングTconv1層アップサンプリング
```
dilationあげていく
convtransposed 1層
num_layer=8, num_filters=128, kernel_sizes=11
Total Tensors: 67964969 	Used Memory: 259.27M
Epoch 10/20
train Loss:2.9278 Timer:12038.3039
val Loss:2.9786 Timer:377.1949
```


###### tags: `研究`