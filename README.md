# intellisar-object-detection
- Raspberry Pi 4B (Raspbian Buster)
- Python 3.7
- OpenCV 4.1.2 (compiled from source)
- Tensorflow 2

This was all setup in a virtual environment that I named "intellisar"

Follow step 3 of the OpenCV guide to setup virtual environment.

## OpenCV Setup
Follow this guide to setup OpenCV

https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/

Just installing OpenCV with pip is easy and might have sufficient functionality for our project, but I didn't test it. Probably worth trying first.

I compiled OpenCV 4.1.2 from source.

## Tensorflow Setup
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.0.0/tensorflow-2.0.0-cp37-none-linux_armv7l.whl

pip3 install tensorflow-2.0.0-cp37-none-linux_armv7l.whl

rm tensorflow-2.0.0-cp37-none-linux_armv7l.whl


## How to Run
python3 app.py

Webserver located at http://localhost:8080

### Optional Arguments
python3 app.py -h

## Model Training Data
model/

https://tensorboard.dev/experiment/KhB0cyZMRQ6wcyYacGYSaw


TFLite_model_1_ckpt-42526/ and TFLite_model_2_ckpt-36242/

https://tensorboard.dev/experiment/2tP6JO4ORHyVhzAWeoWQng
