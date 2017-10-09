#gif.py 
import sys
import imageio

VALID_EXTENSIONS = ('png', 'jpg')


def create_gif(n_images, directory, output_name, duration):
    images = []
    for dp_i in range(n_images):
        images.append(imageio.imread('%s/%06i.png' % (directory, dp_i) ) )
    output_file = '%s.gif' % (output_name)
    imageio.mimsave(output_file, images, duration=duration)


if __name__ == "__main__":
    script = sys.argv.pop(0)

    if len(sys.argv) < 2:
        print('Usage: python {} <duration> <path to images separated by space>'.format(script))
        sys.exit(1)

    duration = float(sys.argv.pop(0))
    filenames = sys.argv


    if not all(f.lower().endswith(VALID_EXTENSIONS) for f in filenames):
        print('Only png and jpg files allowed')
        sys.exit(1)

    create_gif(filenames, duration)