# 深層学習を使ったRNAアクセシビリティ予測の高速化
## 概要
RNAアクセシビリティは，配列内の任意の領域が分子内塩基対を形成するのを防ぐために必要なエネルギーのことで，RNA-RNA相互作用の予測に役に立つ.アクセシビリティはRNA二次構造の自由エネルギーから求めることができるが，この計算にはある程度の時間がかかる.

近年，次世代シークエンサーの登場で大量の配列データが蓄積されていることもあり，アクセシビリティ予測の計算時間はボトルネックになる.

本研究では，深層学習を使うことでアクセシビリティの予測を高速化することを目的とする.深層学習を用いることで塩基配列と正解データのみから配列の特徴を捉えることができ，エネルギーモデルを使わないアクセシビリティ予測をする．既存手法に対してある程度の精度を保ったまま，予測速度を飛躍的に向上させることを目指す．


![](https://i.imgur.com/EneiFqS.png)
$*$RaccessはO(NW^2)



## アクセシビリティ予測の既存手法[[表のlink]](https://academic.oup.com/view-large/79110482)


![](https://i.imgur.com/3iDCZo7.png)
* N: sequence length(配列長)
* W: maximal span(塩基対が結合可能な最大範囲)
* S: sequence window size (Raccessでは使わない)
* ’–’: 10時間以上
* ’x’: オーバーフロー



## モデル
![](https://i.imgur.com/L9RkEwK.png)
#### 固定長
* メリット：既存モデルが多い
* デメリット：固定長 -> **配列処理で解決可能**
#### 可変長
* メリット：配列処理が必要ない
* デメリット：~~今のところは固定長の方が精度速度が良い結果が出ている~~


## 速度比較
![](https://i.imgur.com/dc2oBby.png)
* max_span：W=20(研究の進めやすさのために小さめに設定)
* 固定長モデル
> W=100(実用的な値)にするともっと差が広がるはず
> 
> W=20ではいい感じの精度が出たので，現在はW=100に取組中


## 配列処理
タンパク質間相互作用予測をNNでおこなうという研究[[link]](https://academic.oup.com/bioinformatics/article/34/17/i802/5093239)

部分配列をオーバーラップさせることで可変長に対応できる
![](https://i.imgur.com/7eynHO0.png)
256塩基ごとに512塩基分切り出す(上記論文の場合)

これがRNAのアクセシビリティについても使えるか，Raccessを使って検証したらいけてそうだったので固定長モデルでも可変長に対応できる


#### 実際におこなった処理
![](https://i.imgur.com/MhVV2Xa.png)



※W=20の場合は配列処理から予測まで上手くいって，実データに対しても悪くないくらいの精度・速度になった


[W=20の結果（一旦終了）](https://hackmd.io/@bO5ufDZbQjSJyZCi2-wL9Q/r1pEOF4Lu)

[W=100(取組中)](https://hackmd.io/@bO5ufDZbQjSJyZCi2-wL9Q/BkS4e9NIO)

[結果置き場](https://hackmd.io/@bO5ufDZbQjSJyZCi2-wL9Q/HyuOM8-Lu)



## 今後の予定
* W=100の精度をあげる
* 固定長と可変長の両方進行
* 実データで悪さしていたっぽい'N'の処理


---

[進捗報告0423](https://hackmd.io/@kaisei-h/Hy6rltCLO)




###### tags: `研究`