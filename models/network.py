import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

class Cycle:
    def __init__(self, G_AB, G_BA, D_A=None, D_B=None, is_train=True, device='GPU:0'):
        self.G_AB = G_AB
        self.G_BA = G_BA
        self.D_A = D_A
        self.D_B = D_B
        self.is_train = is_train
        self.device = device
        self.DataL = None
 
    def load_model(self, G_AB_PATH, G_BA_PATH, D_A_PATH, D_B_PATH):
        self.G_AB.load_state_dict(torch.load(G_AB_PATH))
        self.G_BA.load_state_dict(torch.load(G_BA_PATH))
        if self.is_train:
            self.D_A.load_state_dict(torch.load(D_A_PATH))
            self.D_B.load_state_dict(torch.load(D_B_PATH))

    def load_dataset(self, DATA_PATH, batch_size=8,
                            trans=None,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=None,
                            pin_memory=False):
        dataset = ImageFolder(DATA_PATH, 
                                trans)
        self.DataL = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=pin_memory)

    def set_loss(self):
        pass

    def set_requires_grad(self, nets, requires_grad=False):
        """
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/base_model.py
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        """
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/cycle_gan_model.py
        Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.G_AB, self.G_BA], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.D_A, self.D_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
    
    def one_batch_data_preprocess(self, data):
        print(data[1])

    def train(self, epochs=50, learning_rate=0.01):
        pass

    def save_model(self, G_AB_PATH='./G_AB.pt', G_BA_PATH='./G_BA.pt', D_A_PATH='./D_A.pt', D_B_PATH='./D_B.pt'):
        torch.save(self.G_AB.state_dict(), G_AB_PATH)
        torch.save(self.G_BA.state_dict(), G_BA_PATH)
        if self.is_train:
            torch.save(self.D_A.state_dict(), D_A_PATH)
            torch.save(self.D_B.state_dict(), D_B_PATH)

c = Cycle('A', 'B')
trans = transform = T.Compose([
    # resize
    T.Resize((256,256)),
    # to-tensor
    T.ToTensor(),
    # normalize
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
c.load_dataset('D:\\apple2orange\\train', shuffle=False, trans=trans)
for i, data in enumerate(c.DataL):
    c.one_batch_data_preprocess(data)
    break