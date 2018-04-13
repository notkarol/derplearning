import csv
from datetime import datetime
import numpy as np
import os
import PIL.Image
import re
import socket
import time
import yaml

def load_image(path):
    """
    Load the RGB version of the image and a PIL image
    """
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')                    

def save_image(path, img):
    """
    Store an image numpy array or PIL image to disk
    """
    if type(img) == np.ndarray:
        img = PIL.Image.fromarray((img * 255).astype(np.uint8))
    img.save(path)


def get_name(path):
    """
    The name of a script is it's filename without the extension
    """
    clean_path = path.rstrip('/')
    bn = os.path.basename(clean_path)
    name, ext = os.path.splitext(bn)
    return name


def get_hostname():
    return socket.gethostname()


def create_record_folder():
    """
    Generate the name of the record folder and created it
    """
    dt = datetime.utcfromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    hn = socket.gethostname()
    path = os.path.join(os.environ['DERP_ROOT'], "data", "%s-%s" % (dt, hn))
    print("Creating", path)
    os.mkdir(path)
    return path


def load_config(config_path):
    """ 
    Loads the vehicle config and all requisite components configs
    """

    # Make sure we have a path to a file
    if os.path.isdir(config_path):
        config_path  = os.path.join(config_path, 'config.yaml')
    
    # First load the car's config
    with open(config_path) as f:
        config = yaml.load(f)

    # Make sure we set the name and path of the config stored
    if 'name' not in config:
        config['name'] = get_name(config_path)
    if 'path' not in config:
        config['path'] = config_path

    # Then load the each component
    dirname = os.path.dirname(config_path)
    for component_config in config['components']:

        # Check if we need to load more parameters from elsewhere
        if 'path' in component_config:
            component_path = os.path.join(os.environ['DERP_ROOT'], 'config', component_config['path'])
            with open(component_path) as f:
                default_component_config = yaml.load(f)

            # Load paramters only if they're not found in default
            for key in default_component_config:
                if key not in component_config:
                    component_config[key] = default_component_config[key]

            # Make sure we have a name for this component
            if 'name' not in component_config:
                component_config['name'] = os.path.basename(os.path.dirname(component_config['path']))

        # Make sure we were able to find a name
        if 'name' not in component_config:
            raise ValueError("load_config: all components must have a name or a path")

        # Make sure we were able to find a class
        if 'class' not in component_config:
            raise ValueError("load_config: all components must have a class in components/")

    # Make sure we also have a state
    if 'state' not in config:
        config['state'] = {}
        
    return config


def load_component(component_config, config):

    # Load the component from its module
    module_name = "derp.components." + component_config['class'].lower()
    class_fn = load_class(module_name, component_config['class'])
    component = class_fn(component_config, config)

    # If we're ready, add it, otherwise make sure it's required
    if not component.ready and component_config['required']:
        raise ValueError("load_components: missing required", component_config['name'])

    print("Loaded component", module_name)
    return component


def load_components(config):
    """
    Load the class of each component by its name and initialize all state keys.
    """
    from derp.state import State
    state = State(config['state'], config)
    components = [state]

    # Initialize components
    for component_config in config['components']:

        # Load the component object
        component = load_component(component_config, config)

        # Skip a non-ready component. Raise an error if it's required as we can't continue
        if not component.ready:
            if component_config['required']:
                raise ValueError("load_components: required component [%s] not available"
                                 % component_config['name'])
            print("load_components: skipping", component_config['name'])
            continue
        
        # if we survived the cull, add the component to 
        components.append(component)

        # Preset all state keys
        if 'state' in component_config:
            for key in component_config['state']:

                # Don't set the key if it's already set and the proposed value is None
                # This allows us to have components request fields, but have a master
                # initializer. Useful for servo or car-specific steer_offset
                val = component_config['state'][key]
                if key not in state or state[key] is None:
                    state[key] = val

    return state, components


def find_component_config(full_config, name):
    """
    Finds the matching component by name of the component and script if needed
    """
    for component_config in full_config['components']:
        if name in component_config['name']:
            return component_config
    

def load_class(path, name):
    """ 
    Loads the class "name" at relative path (period separated) "path" and returns it
    """
    from importlib import import_module
    m = import_module(path)
    c = getattr(m, name)
    return c


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
            timestamps.append(float(line[0]))
            for value in line[1:]:
                try:
                    value = float(value) if floats else value
                except:
                    value = 0
                state.append(value)
            states.append(state)
    timestamps = np.array(timestamps, dtype=np.double)
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


def find_matching_file(path, name_pattern):
    """
    Finds a file that matches the given name regex
    """
    pattern = re.compile(name_pattern)
    if os.path.exists(path):        
        for filename in os.listdir(path):
            if pattern.search(filename) is not None:
                return os.path.join(path, filename)
    return None
