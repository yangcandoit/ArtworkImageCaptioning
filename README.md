# ArtworkImageCaptioning

This repo is the implementation of "Image captioning for artworks". It currently includes my approach for this task. 

## Installation

```console
git clone https://github.com/yangcandoit/BiasChecker.git
```


open the ArtworkImageCaptioning folder

```console
cd ArtworkImageCaptioning

python main.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data_path /pathtodataset --batch_size 16
```

## Parameters

| parameters | meaning                                                                  |
|------------|--------------------------------------------------------------------------|
| mode       | train or test                                                            |
| exp_name   | the name of this experiment                                              |
| batch_size | the batch size of the data                                               |
| data_path  | the path of the dataset                                                  |
| resume     | whether resume the model                                                 |
| output     | output path. include the during training results  and the configurations |



