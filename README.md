# TL;DR
拡散モデルを使用した胸部CTの超解像
また、トポロジー制約を導入し、血管や肺胞を囲む小葉間隔壁などの連結を目的とする

## config/
設定ファイル
データセットの読み込む場所や、入力画像のサイズ、バッチサイズなどもここで変更できる

## model/sr3_module/diffusion.py
損失の関数が含まれている、ここにgudhiを使用した損失を追加していく

## Dockerfile
環境を作成することができる
```
docker build -t test/test:test
docker run <image id>
docker exec -it <container id>
```

# 実行
```
python3 sr.py
```
