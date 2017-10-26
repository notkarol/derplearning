import evdev
from datetime import datetime
import asyncio

class Controller:
    def __init__(self):

        # Initialize known device names and their handles
        self.device_handles = {'Wireless Controller' : self.handle_ds4}
        
        # Find the first available device to use
        self.gamepad = None
        devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
        print(devices)
        for device in devices:
            if device.name in self.device_handles:
                self.gamepad = device
                break

        # Run handler
        self.device_handles[self.gamepad.name]()


    # Prepare necessary variables to process Dualshock 4
    def handle_ds4(self):
        self.code_map = {0: 'left_stick_horizontal',
                         1: 'left_stick_vertical',
                         2: 'right_stick_horizontal',
                         3: 'left_trigger',
                         4: 'right_trigger',
                         5: 'right_stick_vertical',
                         16: 'arrow_horizontal',
                         17: 'arrow_vertical',
                         304: 'square',
                         305: 'x',
                         306: 'circle',
                         307: 'triangle',
                         308: 'l1',
                         309: 'r1',
                         310: 'l2',
                         311: 'r2',
                         312: 'share',
                         313: 'options',
                         314: 'left_stick_press',
                         315: 'right_stick_press',
                         316: 'menu',
                         317: 'touchpad'}
        self.astick_codes = [0, 1, 2, 5]
        self.deadzone = 8
        self.process = self.process_ds4


    
    async def process_ds4(self, event):
        command = Command()
        async for event in self.gamepad.async_read_loop():
        
            # Handle sticks
            if event.code in self.astick_codes:

                # if in dead zone, don't do anything
                if event.value == 0 or 128 - self.deadzone < event.value <= 128 + self.deadzone:
                    return
                dt = datetime.fromtimestamp(event.sec).strftime('%Y-%m-%d %H:%M:%S') + ".%06i" % event.usec
                print(dt, self.code_map[event.code], event.value)
            else:
                print("Unknown!", event)
        
    def read(self):
        self.process()



c = Controller()
while True:
    c.read()
    
