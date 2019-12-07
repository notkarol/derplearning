import time
import capnp
import messages_capnp
import derp.util

class Keyboard:

    def __init__(self, config):
        self.config = config
        self.device = None
        self.__connect()
        self.speed = 0
        self.steer = 0
        self.speed_offset = 0
        self.steer_offset = 0
        self.record = False
        self.auto = False
        self.control_message = None
        self.state_message = None
        self.__context, self.__publisher = derp.util.publisher('/tmp/derp_keyboard')

        # Prepare key code map so we can use strings to understand what key was pressed
        self.code_map = {1: 'escape', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7',
                         9: '8', 10: '9', 11: '0', 12: '-_', 13: '=+', 14: 'backspace', 15: 'tab',
                         16: 'q', 17: 'w', 18: 'e', 19: 'r', 20: 't', 21: 'y', 22: 'u', 23: 'i',
                         24: 'o', 25: 'p', 26: '[', 27: ']', 28: 'enter', 29: 'left_ctrl', 30: 'a',
                         31: 's', 32: 'd', 33: 'f', 34: 'g', 35: 'h', 36: 'j', 37: 'k', 38: 'l',
                         39: ';', 40: "'", 41: '`', 42: 'left_shift', 43: '\\', 44: 'z', 45: 'x',
                         46: 'c', 47: 'v', 48: 'b', 49: 'n', 50: 'm', 51: ',', 52: '.', 53: '/',
                         54: 'right_shift', 55: 'right_*', 56: 'left_alt', 57: 'space',
                         58: 'capslock', 59: 'f1', 60: 'f2', 61: 'f3', 62: 'f4', 63: 'f5',
                         64: 'f6', 65: 'f7', 66: 'f8', 67: 'f9', 68: 'f10', 69: 'numlock',
                         70: 'scrolllock', 71: 'keypad_7', 72: 'keypad_8', 73: 'keypad_9',
                         74: 'keypad_-', 75: 'keypad_4', 76: 'keypad_5', 77: 'keypad_6',
                         78: 'keypad_+', 79: 'keypad_1', 80: 'keypad_2', 81: 'keypad_3',
                         82: 'keypad_0', 83: 'keypad_..', 96: 'keypad_enter', 97: 'right_ctrl',
                         98: 'keypad_/', 100: 'right_alt', 102: 'home', 103: 'arrow_up',
                         104: 'pagedown', 105: 'arrow_left', 106: 'arrow_right', 107: 'end',
                         108: 'arrow_down', 109: 'pagedown', 110: 'insert', 111: 'delete',
                         125: 'super'}
        self.run()

    def __del__(self):
        if self.device is not None:
            self.device.close()
        self.__publisher.close()
        self.__context.term()

    def __connect(self):
        self.device = derp.util.find_device(self.config['device_names'])
        return self.device is not None

    def __process(self, event):
        control_changed = False
        state_changed = False
        if event.code == 0 or event.type == 4 or not event.value:
            return control_changed, state_changed
        if self.code_map[event.code] == 'arrow_left':
            self.steer -= 16 / 256
            control_changed = True
        elif self.code_map[event.code] == 'arrow_right':
            self.steer += 16 / 256
            control_changed = True
        elif self.code_map[event.code] == 'arrow_up':
            self.speed += 4 / 256
            control_changed = True
        elif self.code_map[event.code] == 'arrow_down':
            self.speed -= 4 / 256
            control_changed = True
        elif self.code_map[event.code] == '1':
            self.steer_offset -= 1 / 256
            state_changed = True
        elif self.code_map[event.code] == '2':
            self.steer_offset += 1 / 256
            state_changed = True
        elif self.code_map[event.code] == '3':
            self.speed_offset -= 16 / 256
            state_changed = True
        elif self.code_map[event.code] == '4':
            self.speed_offset += 16 / 256
            state_changed = True
        elif self.code_map[event.code] == 'r':
            self.record = True
            state_changed = True
        elif self.code_map[event.code] == 'a':
            self.auto = True
            state_changed = True
        elif self.code_map[event.code] == 'escape':
            self.speed = 0
            self.steer = 0
            self.speed_offset = 0
            self.steer_offset = 0
            self.record = False
            self.auto = False
            state_changed = True
            control_changed = True
            return control_changed, state_changed

    def control_message(self):
        msg = messages_capnp.Control.new_message(
            timestampCreated=derp.util.get_timestamp(),
            speed=self.speed,
            steer=self.steer)
        return msg

    def state_message(self):
        msg = messages_capnp.Control.new_message(
            timestampCreated=derp.util.get_timestamp(),
            speedOffset=self.speed_offset,
            steerOffset=self.steer_offset,
            auto=self.auto,
            record=self.record);
        return msg
    
    def read(self):
        self.control_message = None
        self.steer_message = None
        try:
            for msg in self.device.read():
                c, s = self.__process(msg)
                if c:
                    self.control_message = self.control_message()
                if s:
                    self.state_message = self.state_message()        
            return True
        except BlockingIOError:
            return True
        except Exception as e:
            print("ERROR Keyboard.read", e)
        return False

    def run(self):
        if not self.read():
            self.__connect()
        if self.control_message:
            self.control_message.timestampPublished = derp.util.get_timestamp()
            self.__publisher.send_multipart([b'control', self.control_message.to_bytes()])
        if self.state_message:
            self.state_message.timestampPublished = derp.util.get_timestamp()
            self.__publisher.send_multipart([b'state', self.state_message.to_bytes()])


def run(config):
    keyboard = Keyboard(config)
    while True:
        keyboard.run()
