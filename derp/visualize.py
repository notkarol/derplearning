import matplotlib.pyplot as plt

def plot_batch(example, label, name):
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
