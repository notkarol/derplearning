#!/usr/bin/env/python3


import argparse
from time import sleep, time, strftime, gmtime
import os
from shutil import copyfile
import derp.util

# Local Python Files
from derp.camera import Camera
from derp.command import Command
from derp.controller import Controller
from derp.servo import Servo
#from derp.inferer import Inferer

def main(args):
    
    # # Pepare variables for recording
    # date = strftime('%Y%m%dT%H%M%SZ', gmtime())
    # name = '%s' % (date, gethostname())
    # folder = os.path.join(os.environ['DERP_DATA'], name)
    # os.mkdir(folder)
    # state_fp = open(os.path.join(folder, "state.csv"), 'w')
    # state_fp.write("timestamp,speed,steer\n")
    # copyfile(args.config, os.path.join(folder, 'config.yaml'))

    # # Initialize relevant classes
    # config_model = util.loadConfig(args.model)
    # config_camera = util.loadConfig(args.camera)
    # camera = Camera(config)
    # inferer = Inferer(config, folder, mode, args.model) if args.model else None
    command = Command()
    controller = Controller(command)
    servo = Servo()

    # Main loop
    while True:
        sleep(0.02)
        controller.process()
        print("%.6f %s" % (time(), command))
        servo.move(command.speed)
        servo.turn(command.steer)
    #     timestamp = int(time() * 1E6)
    #     #frame = camera.getFrame()
    #     speed, steer = servo.speed, servo.steer
    #     if autonomous:
    #         #Important! All control calculations must be made in the inferer class.
    #         #Otherwise the driving emulator will not properly predict road performance.
    #         nn_speed, nn_steer, nn_thumb = inferer.evaluate(frame, timestamp, speed, steer)
    #         #servo.move(nn_speed)
    #         servo.turn(nn_steer)
    #     state_fp.write(",".join([str(x) for x in (timestamp, speed, steer)]))
    #     state_fp.write("\n")
    #     camera.record()            
        
    # # Cleanup sensors
    # if args.model:
    #     del model
    # del servo
    # del camera
    # state_fp.close()
    
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str, required=True, help="Configuration to use")
    #parser.add_argument('--model', type=str, default="", help="Model to run")
    args = parser.parse_args()
    main(args)
