## Predictor service

## Training

```sh
cp .env.local .env.local
``` 

and replace placeholder with the mlflow user and password

Then you can manuall trigger the training with it
```sh
uv run predictor/src/train.py
```