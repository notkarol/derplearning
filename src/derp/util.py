import csv
import numpy as np
import os
import PIL.Image
import re
import time
import yaml


def has_image_ext(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.png'])


def load_image(path):
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')                    


def save_image(path, img):
    if type(img) == np.ndarray:
        img = PIL.Image.fromarray((img * 255).astype(np.uint8))
    img.save(path)


def get_name(path):
    return os.path.splitext(os.path.basename(path.rstrip('/')))[0]


def get_record_folder():
    from datetime import datetime
    import socket
    dt = datetime.utcfromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    hn = socket.gethostname()
    return os.path.join(os.environ['DERP_DATA'], "%s-%s" % (dt, hn))


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created [%s]" % path)
        return True
    return


def load_config(paths):

    # Default config has no components
    out = { 'name': 'config', 'components': [ {'state' : {} }] }

    # Don't load blank paths
    if paths is None:
        return out

    # For each path we're given
    for path in paths:

        # if we're given a folder, load it's config.yaml by default
        config_path = os.path.join(path, 'config.yaml') if os.path.isdir(path) else path

        # Open the config file 
        with open(config_path) as f:
            config = yaml.load(f)

            if 'components' in config:
                out['components'].extend(config['components'])
            else:
                if 'name' not in config:
                    config['name'] = get_name(config_path)
                out['components'].append(config)
    return out


def load_class(path, name):
    """ 
    Loads the class "name" at relative path (period separated) "path" and returns it
    """
    from importlib import import_module
    m = import_module(path)
    c = getattr(m, name)
    return c
    

def load_components(config, state):
    out = []

    # Initialize components
    for component in config['components']:
        c = load_class("derp.components." + component['class'].lower(), component['class'])
        obj = c(component)

        # Preset all state keys
        if 'state' in component:
            for key in component['state']:
                val = component['state'][key]

                # Don't set the key if it's already set and the proposed value is None
                # This allows us to have components request fields, but have a master
                # initializer. Useful for servo or car-specific steer_offset
                if key in state and val is None:
                    continue

                state[key] = val
        
        # Discover
        text = 'required' if component['required'] else 'optional'
        print("Connecting to %s %s [%s]" % (text, obj.ready, component.ready))

        # Exit if we're missing a component
        if not component.ready and obj.ready:
            return None

        if obj.ready:
            out.append(obj)
    return out
    

def read_csv(path, floats=True):
    """
    Read through the state file and get our timestamps and recorded values.
    Returns the non-timestamp headers, timestamps as 
    """
    timestamps = []
    states = []
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)[1:]
        for line in reader:
            if not len(line):
                continue
            state = []
            timestamps.append(int(re.sub('\D', '', line[0] ) ) )
            #regex to remove any non-decimal characters from the timestamp so that 
            #it can be read as an int
            for value in line[1:]:
                value = float(value) if floats else value
                state.append(value)
            states.append(state)
    timestamps = np.array(timestamps, dtype=np.uint64)
    if floats:
        states = np.array(states, dtype=np.float)
    return timestamps, headers, states


def find_value(haystack, key, values, interpolate=False):
    """
    Find the nearest value in the sorted haystack to the specified key.
    """

    nearest = 0
    diff = np.abs(haystack - key)
    if interpolate:
        nearest = diff.argsort()[:2]
        return (values[nearest[0]] + values[nearest[1]]) / 2

    nearest = diff.argmin()
    return values[nearest]


def find_matching_file(path, name):
    pattern = re.compile(name)
    for filename in os.listdir(path):
        if pattern.search(filename) is not None:
            return os.path.join(path, filename)
    return None
        

def plot_batch(example, label, name):
    import matplotlib.pyplot as plt
    dim = int(np.sqrt(len(example)))
    fig, axs = plt.subplots(dim, dim, figsize=(dim, dim))
    for i in range(len(example)):
        x = i % dim
        y = int(i // dim)

        # change from CHW to HWC and only show first three channels
        img = np.transpose(example[i].numpy(), (1, 2, 0))[:, :, :3]
        axs[y, x].imshow(img)
        axs[y, x].set_title(" ".join(["%.2f" % x for x in label[i]]))
        
    plt.savefig("%s.png" % name, bbox_inches='tight', dpi=160)
    print("Saved batch [%s]" % name)


def find_device(name, exact=False):
    import evdev
    out = []
    for filename in sorted(evdev.list_devices()):
        device = evdev.InputDevice(filename)
        if ((exact and device.name == name) or
            (not exact and name.lower() in device.name.lower())):
            return device
    return None
    

def get_current_timestamp():
    """ 
    The cononical way to get the timestamp for this program
    """
    return time.now()
