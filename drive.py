#!/usr/bin/env/python3
import argparse
import os
import derp.state
import derp.util

def main(args):

    # Prepare configuration and some supplied arguments
    config_path = os.path.join(os.environ['DERP_ROOT'], 'config', args.config + '.yaml')
    config = derp.util.load_config(config_path)
    if args.model_dir is not None:
        config['model_dir'] = args.model_dir
    state = derp.state.State(config)
    components = derp.util.load_components(config, state)

    # Event loop that runs until state is done
    prev_time = None
    while not state.done():
        
        # Sense Plan Act Record loop
        for component in components:
            component.sense()
        for component in components:
            component.plan()
        for component in components:
            component.act()
        for component in components:
            component.record()
        state.record()
            
        # Print to the screen for verbose mode
        if not args.quiet:
            print("%.3f %s %s | speed %6.3f + %6.3f %i | steer %6.3f + %6.3f %i" %
                  (state['timestamp'],
                   'R' if state['record'] else '_',
                   'A' if state['auto'] else '_',
                   state['speed'], state['offset_speed'], state['use_offset_speed'],
                   state['steer'], state['offset_steer'], state['use_offset_steer']))
                                              

# Load all the arguments and feed them to the main event loader and loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=derp.util.get_hostname(),
                        help="physical config")
    parser.add_argument('--model_dir', type=str, default=None,
                        help="directory to models we wish to run")
    parser.add_argument('--quiet', action='store_true', default=False,
                        help="do not print speed/steer")
    args = parser.parse_args()
    main(args)
