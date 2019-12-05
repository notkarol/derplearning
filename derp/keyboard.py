import time
import capnp
import input_capnp
import derp.util

class Keyboard:

    def __init__(self, config):
        self.config = config
        self.device = None
        self.__connect()
        self.speed = 0
        self.steer = 0
        self.offset_speed = 0
        self.offset_steer = 0
        self.record = 0
        self.auto = 0
        self.__context, self.__publisher = derp.util.publisher('input')

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

    def __del__(self):
        if self.device is not None:
            self.device.close()
        self.__publisher.close()
        self.__context.term()

    def __connect(self):
        self.device = derp.util.find_device(self.config['device_names'])
        return self.device is not None

    def __process(self, event):
        if event.code == 0 or event.code == 4 or not event.value:
            return False
        if self.code_map[event.code] == 'arrow_left': self.steer -= 16 / 256
        elif self.code_map[event.code] == 'arrow_right': self.steer += 16 / 256
        elif self.code_map[event.code] == 'arrow_up': self.speed += 4 / 256
        elif self.code_map[event.code] == 'arrow_down': self.speed -= 4 / 256
        elif self.code_map[event.code] == '[': self.offset_steer -= 1 / 256
        elif self.code_map[event.code] == ']': self.offset_steer += 1 / 256
        elif self.code_map[event.code] == '1': self.offset_speed = 24 / 256
        elif self.code_map[event.code] == '2': self.offset_speed = 28 / 256
        elif self.code_map[event.code] == '3': self.offset_speed = 32 / 256
        elif self.code_map[event.code] == '4': self.offset_speed = 40 / 256
        elif self.code_map[event.code] == '5': self.offset_speed = 44 / 256
        elif self.code_map[event.code] == '6': self.offset_speed = 48 / 256
        elif self.code_map[event.code] == '7': self.offset_speed = 52 / 256
        elif self.code_map[event.code] == '8': self.offset_speed = 56 / 256
        elif self.code_map[event.code] == '9': self.offset_speed = 60 / 256
        elif self.code_map[event.code] == '0': self.offset_speed = 64 / 256
        elif self.code_map[event.code] == 'r': self.record = True
        elif self.code_map[event.code] == 'a': self.auto = True
        elif self.code_map[event.code] in ['q', 'escape']:
            self.speed = 0
            self.steer = 0
            self.record = False
            self.auto = False
        return True

    def message(self):
        msg = input_capnp.Input.new_message(
            timestamp=derp.util.get_timestamp(),
            speed=self.speed,
            steer=self.steer,
            offset_speed=self.offset_speed,
            offset_steer=self.offset_steer,
            record=self.record,
            auto=self.auto)
        return msg

    def read(self):
        try:
            event = self.device.read():
            return self.__process( event)
        except BlockingIOError:
            print("Keyboard BLOCKING ERROR")
        except Exception as e:
            print("Keyboard RUN ERROR")
        return None

    def run(self):
        while True:
            send = self.read()
            if send is None:
                self.__connect()
                continue
            if send is False:
                continue
            msg = self.message()
            self.__publisher.send_multipart(['input', msg.to_bytes()])
