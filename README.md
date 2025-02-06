# test-yolo

## Setup dev env

first time.

```shell
$ conda create --prefix ./.venv python=3.12.7 -y
$ conda activate ./.venv
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

after open terminal.

```shell
$ conda activate ./.venv
```

exit env.

```shell
$ conda deactivate
```

## Model

https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/blob/main/models/Layout/YOLO/doclayout_yolo_ft.pt

## Dependencies
https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html