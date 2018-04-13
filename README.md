# Derp Learning

Derp Learning is a Python package that collects data, trains models, and then controls an RC car for track racing.

## Getting Started

### Software

The developers of this project primarily use Ubuntu 16.04 on both x86_64 and aarch64 architectures.

Inside the _install_ folder there are series of prerequisite install scripts. Run _prerequisites.sh_ to run them. Then _setup.py_ script will install the Python package.

```bash
cd install
bash install.sh
```

### Hardware

You will need access to a drill to cut holes in the car's chassis and into the PVC sheet or 3D printed camera pylon. Select one (or more) of the compute options. TODO detailed instructions

#### Necessary

* Traxass 1/10 Slash 2WD RTR [$190](https://www.amazon.com/gp/product/B01DU474B0)
* Pololu Micro Maestro 7-Channel USB Servo Controller  [$18](https://www.amazon.com/gp/product/B004G54CHW)
* Adafruit BNO055 9-DOF Absolute Orientation IMU [$35](https://www.adafruit.com/product/3055)
* Playstation Dualshock 4 Controller [$45](https://www.amazon.com/gp/product/B01MD19OI2)
* 7.4V 8000mAh LiPo Battery [$55](https://www.amazon.com/gp/product/B013RUGOFE)
* Traxxas Parallel Wire Harness [$9](https://www.amazon.com/dp/B01AO4M0UE)
* LiPo Battery Bag [$8](https://www.amazon.com/gp/product/B00T01LLP8)
* Low Voltage Meter [$8](https://www.amazon.com/gp/product/B01H19NU90)
* Battery Charger [$55](https://www.amazon.com/gp/product/B00466PKE0)
* Dual Lock Reclosable Fastener [$19](https://www.amazon.com/gp/product/B00JHKTDMA)
* Brass Spacers/Offsets M2+M3 Offsets [$17](https://www.amazon.com/gp/product/B06XCNF6HK)
* Nylon Washers M2+M3 [$8](https://www.amazon.com/gp/product/B01G4U0D1K)
* TODO: Roll Cage 3D printed

#### Raspberry Pi Zero W Compute

* Raspberry Pi Zero W Camera Pack [$45](https://www.adafruit.com/product/3414)
* USB Battery Charger [$18](https://www.amazon.com/gp/product/B06XS9RMWS)

#### Raspberry Pi 3 Compute

* Raspberry Pi 3 B [$35](https://www.adafruit.com/product/3055)
* Raspberry Pi Camera [$30](https://www.adafruit.com/product/3099)
* Camera Cable [$2](https://www.adafruit.com/product/1648)
* USB Battery Charger [$18](https://www.amazon.com/gp/product/B06XS9RMWS)

#### Jetson Compute

* Nvidia Jetson TX1 or TX2 Developer Kit [$300](http://www.nvidia.com/object/jetsontx2-edu-discount.html)
* Orbitty Carrier Board [$173](http://www.wdlsystems.com/Computer-on-Module/Carrier-Boards/CTI-Orbitty-Carrier-for-NVIDIA-Jetson-TX1.html)
* USB Camera [$45](https://www.amazon.com/gp/product/B07143BJ6J)
* 11.1V 4000mAh LiPo Battery [$28](https://www.amazon.com/gp/product/B01I2544TW)

#### Optional, but helpful for debugging and expansions

* 4 Port USB 3.0 Hub [$10](https://www.amazon.com/gp/product/B00XMD7KPU)
* Portable HDMI Monitor [$110](https://www.amazon.com/gp/product/B01J52TWD4)
* Portable Wireless Keyboard/Mouse [$25](https://www.amazon.com/gp/product/B014EUQOGK)

## Usage

All of the following commands need to be run from the __src__ folder.


### Collect Data
On the car run:

```bash
python3 drive.py
```

Data by default is saved to files in the folder /data/ which is created in the parent directory of /derp_learning/


The data can be moved by swapping the SD card if the derplearning directory is located there or by using ssl rsync from the directory you want to move the data to on your device:
```bash
rsync rvP ${car}:/mnt/sdcard/data/* $DERP_ROOT/data/train
```

### Single Pass Pipeline
To move label and train a model on collected data use the shell script pipeline.sh. This is the ideal way to deploy a model trained on same day collected data. This option may be used instead of manualy performing the below steps.

```bash
bash pipeline.sh __NAME__ __BUTTON__ __FRESH_DATA_SOURCE__
```

Note: the data source is a location containing data you want to move to the local training data folder.

### Label Data
Any recorded data file can be labled creating a file /label.csv in the same folder as all other data files for a given recording.

```bash
python3 label.py --path data/???
```

### Build Dataset
This program prepares the recorded data for use in training and validation.

```bash
python3 clone_create.py --config __NAME__
```

### Train Model
Runs training on the dataset to build a model for deployment

```bash
python3 clone_train.py --config __NAME__
```

### Deploy Model
To deploy a model for use in control of a vehicle copy the model file to the desired button folder on the vehicle and rename the model to "clone.pt"

```bash
rsync -rvP $model ${car}:$DERP_ROOT/scratch/model/__BUTTON__/clone.pt
```

Once a model is deployed to the car it can be loaded by pressing the appropriate button and given control of the vehicle by pressing the playstation button.
