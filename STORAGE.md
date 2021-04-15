# 結果置き場
## ごちゃごちゃ出た結果を雑に置いていくページ
### max_span=100


![](https://i.imgur.com/OEJh39V.png)
```
1024切
train_dataset, val_dataset = [750000, 50000]
Fixed(num_layer=32, mid_units=800, num_filters=256, length=1024）
Adam(lr=1e-3, weight_decay=1e-6, eps=1e-5)
Epoch 20/20
train Loss:3.044788383875529 Timer:4340.012280225754
val Loss:26.288977885742188 Timer:72.40725111961365
```

![](https://i.imgur.com/yDXNXMM.png)
```
1024切
loss2.57
train_dataset, val_dataset = [750000, 50000]
Variable(num_layer=64, num_filters=128）
Adam(lr=1e-3, weight_decay=1e-6, eps=1e-5)
epochs = 20
```


![](https://i.imgur.com/GejIkyJ.png)
```
256切
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=4, kernel_sizes=255, length=256）
Adam(lr=1e-4, weight_decay=1e-6, eps=1e-5)
Epoch 10/20
train Loss:1.5241191647423638 Timer:3695.7967278957367
val Loss:1.65580967502594 Timer:144.73661732673645
```



![](https://i.imgur.com/zilUb4V.png)
```
256切
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=4, mid_units=500)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:2.39762034347958 Timer:341.08435916900635
val Loss:2.4030389239501955 Timer:25.195719242095947
```

![](https://i.imgur.com/aW9u8jd.png)
```
256切
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=4, mid_units=100)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:2.6644209328715007 Timer:221.46461987495422
val Loss:2.667066247558594 Timer:21.188048839569092
```


![](https://i.imgur.com/qk75vVL.png)
```
256切
train_dataset, val_dataset = [1800000, 200000]
Variable(num_layer=4, num_filters=128)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:1.7626298432074652 Timer:211.31326508522034
val Loss:1.765414800300598 Timer:22.890778064727783
```


![](https://i.imgur.com/3uM1Iz5.png)
```
256切
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=8, mid_units=2000, length=256)
Adam(lr=1e-4, weight_decay=1e-6, eps=1e-5)
Epoch 20/20
train Loss:0.9438841723399692 Timer:707.7170414924622
val Loss:1.5906616962051392 Timer:31.91866636276245
```

![](https://i.imgur.com/eLPgqkw.png)
```
256切
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=32, mid_units=1000, length=256)
Adam(lr=1e-4, weight_decay=1e-6, eps=1e-5)
Epoch 11/20
train Loss:1.1655881362173293 Timer:1628.5558025836945
val Loss:1.2025168954658507 Timer:46.341949224472046
```



![](https://i.imgur.com/MyTGp2p.png)
```
256切
train_dataset, val_dataset = [1800000, 200000]
Variable(num_layer=16, num_filters=128, flag=False)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:1.1906195311609904 Timer:743.2126786708832
val Loss:1.1948930996131897 Timer:36.51729130744934
```
![](https://i.imgur.com/ulBEPZ6.png)
```
256切
train_dataset, val_dataset = [1800000, 200000]
Variable(num_layer=32, num_filters=128)
Adam(lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:0.9912123876317342 Timer:1461.375155210495
val Loss:0.9951717900657654 Timer:62.6632285118103
```


![](https://i.imgur.com/06f06Ti.png)
```
256切
train_dataset, val_dataset = [1800000, 200000]
Fixed(num_layer=32, length=256, mid_units=400)
Adam( lr=0.001, weight_decay=1e-6, eps=1e-5)
Epoch 10/10
train Loss:1.5502054579120212 Timer:1527.3441843986511
val Loss:1.565917170677185 Timer:59.962098121643066
```