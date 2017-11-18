# Recording

You will need a Dualshock 4 controller unpaired from your car. Run the following command in a terminal logged into the car's ubuntu user account:

```bash
python3 drive.py --hw config/hw_hfov100.yaml
```

Then press the PS and Share buttons together until you get a double white blink, The blinks should look like this:
----.-.
If the controller starts flashing evenly then the sync attemp failed controller side. Once the device stops flashing you can try to sink by pushing the PS and Share buttons again. Even flashing looks like this:
--..--..
The program will take a few seconds to sync. Once it is ready the blinking will stop, you can now control the car. 

## Dualshock 4 Controls

Left and right analog sticks moved horizontally control steering. The right stick is more sensitive for smaller turns and should be used.

The left trigger is used to control forward speed. 

The right trigger is used to control reverse speed. You may need to double tap it to register due to a limitation of the stock ESC.

Triangle button stops autonomy, recording, and the car. However, it does not close the program.

Circle starts recording.

Square stops recording.

The touchpad stops the car and closes the program.

Left Arrow decreases Steer Offset. Right Arrow increases Steer Offset. This is used to set the true straight of the car when steering is set to zero.

Up Arrow increases Speed Offset. Down Arrow decreases Speed Offset. These are used to set the default speed of autonomous models that do not predict their own speed.

You can start Full Autonomy by pressing the PS key.
You can start Speed Autonomy with the Share button.
You can start Steer Autonomy with the Options button.
X (cross) button stops autonomy. However, you will have to take over speed/steer yourself.

L1 and R1 are unmapped.

## Driving Instructions

1. Start the program and sync the car.
2. Start recording by pressing the Circle key.
3. Drive the car around the course the way you wish to. Two recommended modes are:
  1. Follow the yellow line as best you can.
  2. Stay in lane.
4. When you're done press the Triangle key to stop the car and recording.

# Label

Labeling traces is a way to describe where in the video we have useful correct data. We describe data by selecting a "mode" and then navigating through the video in time to mark the stretch of data in time as that "mode." Data is "useful" when the racecar is driving in a behavior we wish to clone. To open the labeling tool:

```bash
python3 label.py --path ${DERP_DATA}/train/20170812T184653Z-paras
```

## Navigation
You can maneuver through the tool through the arrow keys.

* Left Arrow: Move backward in time 1 frame
* Right Arrow: Move forward in time 1 frame
* Up Arrow: Move forward in time 1 second
* Down Arrow: Move backward in time 1 second
* `: Move to the beginning of the file.
* 1: Move to 10% into the file.
* 2: Move to 20% into the file.
* 3: Move to 30% into the file.
* 4: Move to 40% into the file.
* 5: Move to 50% into the file.
* 6: Move to 60% into the file.
* 7: Move to 70% into the file.
* 8: Move to 80% into the file.
* 9: Move to 90% into the file.
* 0: Move to 100% into the file.

## Modes

* g: good data that we should use for data
* r: risky data that we mark is interesting but probably don't want to train
* t: trash data that we wish to junk and not use again

You an also clear the marker so that when you maneuver through the video you don't update the mode at the time.

* c: clear marker

## Saving

* s: Save video

# Create Dataset

Datasets are a collection of images and their correct output. You may configure what these settings are through a config file. The created dataset is in ${DERP_SCRATCH} based on the name of the sw config without the .yaml extension. The experiment is the name of the configuration experiment you wish to run in case there are multiple models you want to train for a config. At the moment it should always be "clone"

```bash
python3 train.py --sw config/sw_clone_fixspeed_delay-0.3.yaml --exp clone
```

# Train Model

Training takes a created dataset and repeatedly iterates on it. The arguments should be at least what you specified for training but you can add more. Useful arguments include model to specify which model (--model by default is BasicModel, though PyramidModel is pretty great) and which gpu (--gpu and it's index)

```bash
python3 train.py --sw config/sw_clone_fixspeed_delay-0.3.yaml --exp clone
```

# Autonomous

Autonomous is like recording, but you need to specify a software config and a model folder where you have the models. The names in the folder should mimic the variables you used for "exp" in create.py and train.py, ending with a ".pt"

```bash
python3 drive.py --hw config/hw_hfov075.yaml --sw config/sw_clone_fixspeed_delay-0.3.yaml --path $HOME/sw_clone_fixspeed_delay-0.3-basicmodel --speed 0.15
```

To verify the model is working you can also launch the program with a "--verbose" option.

You can edit any of the "params" arguments in the software config to change the behavior of the model to tune it. Everything in "predict" and "patch" can not be changed.
