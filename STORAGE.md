# 結果置き場
### ごちゃごちゃ出た結果を雑に置いていくページ


![](https://i.imgur.com/OEJh39V.png)
```
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [750000, 50000])
net = model.Fixed(num_layer=32, mid_units=800, num_filters=256, length=1024, flag=False).to(device
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6, eps=1e-5)
Epoch 20/20
train Loss:3.044788383875529 Timer:4340.012280225754
val Loss:26.288977885742188 Timer:72.40725111961365
```

![](https://i.imgur.com/yDXNXMM.png)
```
loss2.57
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [750000, 50000])
net = model.Variable(num_layer=64, num_filters=128, flag=False).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6, eps=1e-5)
epochs = 20
```

