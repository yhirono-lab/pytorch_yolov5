# Attention-based Multiple Instance Learning with YOLO

YOLOのプログラム  
もともとのプログラムはYOLOv5(>https://github.com/ultralytics/yolov5)を用いており，ここから必要なコードのみを抽出したプログラムになっている．  
機能がよくわかっていないがそのままコピーしているコードもある．  
labelimg(参考>https://doc.gravio.com/manuals/gravio4/1/ja/topic/labelimg)を使ってパッチ画像(224x224)にアノテーションを行ったデータセットを用いる  

## 実行環境
ライブラリバージョン
- python 3.8.10
- numpy 1.21.2
- opencv-python 4.5.3.56
- openslide-python 1.1.2
- pillow 8.3.1
- torch 1.9.0
- torchvision 0.10.0

使用したマシン
- マシン noah
- CUDA Version 11.4
- Driver Version 470.86

## ファイル構造
必要なデータセット
```
./data
    ├ hyps/                 # 学習率などのパラメータが記載されている
    └ Hodgkin/
        ├ Hodgkin0/
        |   ├ labels/
        |   |   ├ val/      # 検証用アノテーションデータ(.txt)
        |   |   ├ train/    # 訓練用アノテーションデータ(.txt)
        |   |   └ test/     # テスト用アノテーションデータ(.txt)
        |   └ images/
        |       ├ val/      # 検証用アノテーションデータ(.tif)
        |       ├ train/    # 訓練用アノテーションデータ(.tif)
        |       └ test/     # テスト用アノテーションデータ(.tif)
        └ Hodgkin0.yaml
```
同じHodgkinのデータセットでもtrain, test, validを入れ替えた5つ分を用意している
それぞれのデータセットへの実験でもっともよかったパラメータをMILYOLOに用いている  


アノテーションデータは画像データと拡張子以外を同じファイル名にする．  
アノテーションデータの内容は以下のように記載される(labelimgでYOLOモードにしてアノテーションすればよい)
実際のアノテーションでは一種類の特徴的な細胞核しか扱わないのでlabel_numberは0のみになる
```
label_number x_pos y_pos width height
例
0 0.113 0.224 0.324 0.532
1 0.324 0.345 0.432 0.123
...
```

.yamlファイルには以下のようにtrain, test, validのimageへのパスを記入する
```
train: data/Hodgkin/Hodgkin0/images/train
val:   data/Hodgkin/Hodgkin0/images/val
test:  data/Hodgkin/Hodgkin0/images/test

nc: 1

names: ['Hodgkin']

---------------(以下説明)---------------------------
nc = number of class，ラベルの数
names = 各ラベルの名称
```

必要なソースコード
```
Source/
　　├ Arial.ttf             # 不明
　　├ best.py               # 実行テスト用のパラメータ
　　├ callbacks.py          # エラー処理を同時に実行されるプログラム
　　|                       # 必要性はないがエラーが起きるので実装している
　　├ dataloader            # pytorchのデータローダー変換用のプログラム
　　|                       # cutmixなどのAugmentationが含まれる
　　├ detect.py             # 検出テストのプログラム
　　├ loss.py               # 損失関数のプログラム
　　├ make_patch_tile.py    # 複数のパッチ画像をまとめるプログラム
　　├ plot_utils.py         # 検出テスト時の描画用サブ関数のプログラム
　　├ train.py              # 訓練用プログラム
　　├ utils.py              # 汎用的な関数のプログラム
　　└ yolo.py               # モデルのプログラム
```

プログラムの実行順序
結果は全て`runs/`に保存される
```
Source/
　　├ 1.train.py
　　├ 2.detect.py
　　└ 3.make_patch_tile.py
```

各プログラムには実行時にパラメータ引数の入力が必要

train.pyについて  
* 実行すると最後のエポックの重み(last.pt)と検証データの結果が最も良いエポックの重み(best.pt)が保存される  
```
--weights       基本入力しない，入力するとその重みを初期値として学習する
--data          データセットの.yamlファイルを指定 例：./data/Hodgkin/Hodgkin.yaml
--hyp           hypの.yamlファイルを指定(./data/hyps/hyp.scratch.yamlから変えたことはない)
--epochs        自由にepoch数を指定(early stopが実装されてるので120とかで止まる)
--batch-size    バッチサイズを指定
--name          保存する時の名前を指定(未入力ならexpになる)
```

detect.pyについて  
* パッチ画像単位で検出をおこなった結果が保存される
```
--weights       訓練時の重みを入力する 例：./runs/train/Hodgkin0/weights/best.pt
--data          テストデータのパスを指定 例：./data/Hodgkin/Hodgkin0/images/test
--conf_thres    検出結果を描画する時の検出確率の閾値を設定 例：0.50
--save-txt      検出結果の描画時にラベル名を表示するかどうかのフラグ
--save-conf     検出結果の描画時に検出確率を表示するかどうかのフラグ
--save-crop     検出結果の保存時に検出範囲のみの画像も保存するかどうかのフラグ
--save-true     検出結果の描画時に正解データも描画するかどうかのフラグ
--name          保存する時の名前を指定(未入力ならexpになる)
```