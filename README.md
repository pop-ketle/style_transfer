# style_transfer
CNNで画像にスタイルを載せるアレ

#### サンプル

|コンテント|スタイル|出力|gif|
|:-:|:-:|:-:|:-:|
|<img src="./sample_images/inputs/cat1.jpg" width="128">|<img src="./sample_images/inputs/style1.jpg" width="128">|<img src="./sample_images/outputs/o1.jpg" width="128">|
<img src="./sample_images/outputs/o1.gif" width="128">|
|<img src="./sample_images/inputs/cat2.jpg" width="128">|<img src="./sample_images/inputs/style2.jpg" width="128">|<img src="./sample_images/outputs/o2.jpg" width="128">|<img src="./sample_images/outputs/o2.gif" width="128">|
|<img src="./sample_images/inputs/cat2.jpg" width="128">|<img src="./sample_images/inputs/style3.jpg" width="128">|<img src="./sample_images/outputs/o3.jpg" width="128">|<img src="./sample_images/outputs/o3.gif" width="128">|

結果はわりと画像依存  
他の実行例とか見るに、モデルとかパラメータをもう少し工夫すればもっといい結果を出せるんだと思う

#### 使い方
```
content_image_path = '[コンテント画像のパス]'
style_image_path   = '[スタイル画像のパス]'
output_dir_path    = '[出力先のディレクトリのパス]'
```

#### 説明
- [] 後で書く
