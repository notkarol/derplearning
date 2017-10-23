import torchvision.transforms as transforms

def clone_train():
    return transforms.Compose([transforms.ColorJitter(brightness=0.5,
                                                      contrast=0.5,
                                                      saturation=0.5,
                                                      hue=0.1),
                               transforms.ToTensor()])

def clone_eval():
    def reflective(x):
        return x[:2]
    return reflective
