
# Derp Learning

Derp Learning is a Python package that collects data, trains models, and then controls an RC car for track racing.

## Hardware

You will need access to a drill and a soldering kit.

### Vehicles

The following vehicles have been tested and are supported:

#### SparkFun Jetbot

If you prefer something smaller, slower, with zero turn radius powered by individual-wheel drive motors the Sparkfun JetBot Kit [$275](https://www.sparkfun.com/products/15365) includes all parts, except for the optional IMU and Dualshock 4 controller. Please follow the setup guide at the site.

* Adafruit BNO055 9-DOF Absolute Orientation IMU [$35](https://www.adafruit.com/product/2472)
* Playstation Dualshock 4 Controller [$45](https://www.playstation.com/en-us/explore/accessories/gaming-controllers/dualshock-4/)

#### Traxxas-based Vehicles

Each of the following vehicles has been tested and works with the platform. 3D printed parts are TBD.

* LaTrax Rally (1/18 scale brushed) [$120](https://latrax.com/products/rally)
* Traxxas Slash (1/10 scale brushed) [$240](https://traxxas.com/products/models/electric/58034-1slash)
* Traxxas E-Revo (1/16 scale brushless) [$250](https://traxxas.com/products/models/electric/erevo-vxl-116-tsm)

* Jetson Nano [$100](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
* Pololu Micro Maestro 6-Channel USB Servo Controller [$19](https://www.pololu.com/product/1350)
* Dual Lock Reclosable Fastener [$19](https://www.amazon.com/gp/product/B00JHKTDMA)
* USB Battery [$16](https://www.amazon.com/dp/B07MNWPFG8/)
* USB to Barrel Cable [$7](https://www.amazon.com/dp/B075112RM6)
* USB Jumper [$5](https://www.amazon.com/dp/B077957RN7/)
* Brass Spacers/Offsets M2+M3 Offsets [$17](https://www.amazon.com/dp/B06XCNF6HK)
* Nylon Washers M2+M3 [$8](https://www.amazon.com/dp/B01G4U0D1K)
* Breadboard Jumper Wires [$7](https://www.amazon.com/dp/B01EV70C78/)
* Wifi Card [$20](https://www.amazon.com/dp/B01MZA1AB2)
* Wifi Antenna [$3](https://www.arrow.com/en/products/2042811100/molex)
* Camera [$30](https://www.sparkfun.com/products/15430)
* 64GB MicroSD Card [$12](https://www.amazon.com/dp/B06XX29S9Q)
* Adafruit BNO055 9-DOF Absolute Orientation IMU [$35](https://www.adafruit.com/product/2472)
* Playstation Dualshock 4 Controller [$45](https://www.playstation.com/en-us/explore/accessories/gaming-controllers/dualshock-4/)


## Software

This software has been tested on Ubuntu 18.04 using Python 3.6 on both x86_64 and aarch64 architectures.

```bash
bash install.sh
```

### Collect Data

```bash
bin/drive.py config/rally.yaml
```

### Label Data

```bash
bin/label.py data/recording-2020????-??????-????
```

### Build a dataset and train cloning model

```bash
bin/clone.py config/brain-clone.yaml
```

### Test model

```bash
bin/drive.py config/rally.yaml
```
