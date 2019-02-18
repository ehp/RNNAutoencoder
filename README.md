# RNN Autoencoder

Simple RNN autoencoder example in PyTorch. Can be used as anomaly detection for timeline data. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Use python 3.x and libraries from requirements.txt file.

```
virtualenv --python /usr/bin/python3 venv
. ./venv/bin/activate
pip install -r requirements.txt
```

### Training

Clone NAB git repo with datasets:

```
git clone https://github.com/numenta/NAB.git
```

And then start training:

```
python -m autoencoder.rnntrainer \
--train_file NAB/data/artificialNoAnomaly/art_daily_small_noise.csv \
--test_file NAB/data/artificialWithAnomaly/art_daily_jumpsup.csv
```

See rnntrainer.py file for more options and default values.

## Authors

* **Petr Masopust** - *Initial work* - [EHP](https://github.com/ehp)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
