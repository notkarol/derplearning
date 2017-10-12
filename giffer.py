#gif.py 
import sys
import imageio

import yaml
with open("config/line_model.yaml", 'r') as yamlfile:
        cfg = yaml.load(yamlfile)

VALID_EXTENSIONS = ('png', 'jpg')


def create_gif(n_images, soource_directory, output_name, duration):
    images = []
    for dp_i in range(n_images):
        images.append(imageio.imread('%s/%06i.png' % (directory, dp_i) ) )
    output_file = '%s.gif' % ( output_name)
    imageio.mimsave(output_file, images, duration=duration)


if __name__ == "__main__":
    
    #file management stuff
    directory = "%s/ver_%s" % (cfg['dir']['validation'], cfg['dir']['model_name'])
    subdirectory = 'virtual_comparison'
    

    val_count = 255
    duration = .05
    filenames = sys.argv

    create_gif(val_count, '%s/%s' % (directory, subdirectory), '%s/%s' %(directory, subdirectory), .05)