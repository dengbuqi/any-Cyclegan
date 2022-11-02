import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import itertools
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
class MyImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        class_to_idx,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root = root,
            transform = transform,
            target_transform = target_transform,
            is_valid_file = is_valid_file,
        )
        if sorted(self.classes) == sorted(class_to_idx.keys()):
            o_idx_to_class = {}
            for key, value in self.class_to_idx.items():
                o_idx_to_class[value] = key
            if self.class_to_idx != class_to_idx:
                for i in range(len(self.imgs)):
                    path, classid = self.imgs[i]
                    self.imgs[i] = (path, class_to_idx[o_idx_to_class[classid]])
            self.class_to_idx = class_to_idx
        else:
            print(f'class_to_idx keys {sorted(class_to_idx.keys())} is not equals to self.classes {sorted(self.classes)}')
class Cycle:
    def __init__(self, G_AB, G_BA, D_A=None, D_B=None, is_train=True, device='CPU'):
        self.G_AB = G_AB.to(device)
        self.G_BA = G_BA.to(device)
        if is_train:
            self.D_A = D_A.to(device)
            self.D_B = D_B.to(device)

        self.real_A = None
        self.real_B = None
        self.fake_A = None
        self.fake_B = None
        self.is_train = is_train
        self.device = device

    def load_model(self, G_AB_PATH, G_BA_PATH, D_A_PATH, D_B_PATH):
        self.G_AB.load_state_dict(torch.load(G_AB_PATH))
        self.G_BA.load_state_dict(torch.load(G_BA_PATH))
        if self.is_train:
            self.D_A.load_state_dict(torch.load(D_A_PATH))
            self.D_B.load_state_dict(torch.load(D_B_PATH))

    def load_train_dataset(self, TRAIN_DATA_PATH,
                            class_to_idx,
                            batch_size=8,
                            train_trans=None,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=None,
                            pin_memory=False):
        self.Train_Data = DataLoader(MyImageFolder(TRAIN_DATA_PATH,
                                class_to_idx=class_to_idx,
                                transform=train_trans),
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=pin_memory)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {}
        for key, value in self.class_to_idx.items():
                self.idx_to_class[value] = key
    
    def load_test_dataset(self, TEST_DATA_PATH,
                            class_to_idx,
                            batch_size=8,
                            test_trans=None,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=None,
                            pin_memory=False):
        self.Test_Data = DataLoader(MyImageFolder(TEST_DATA_PATH,
                                class_to_idx=class_to_idx,
                                transform=test_trans),
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=pin_memory)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {}
        for key, value in self.class_to_idx.items():
                self.idx_to_class[value] = key

    def set_loss(self, 
                criterionGAN = nn.MSELoss(), 
                criterionCycle = torch.nn.L1Loss(), 
                criterionIdt = torch.nn.L1Loss(),
                lambda_identity = 1,
                lambda_A = 0.5,
                lambda_B = 0.5):
        self.criterionGAN = criterionGAN
        self.criterionCycle = criterionCycle
        self.criterionIdt = criterionIdt
        self.lambda_identity = lambda_identity
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
    
    def set_optimizer_and_schedulers(self, lr=0.0002, beta1 = 0.5, gamma=0.9):
        self.lr = lr
        self.beta1 = beta1
        self.gamma = gamma
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.schedulers = [ExponentialLR(op, gamma=gamma) for op in self.optimizers] 

    def set_metrics(self, ):
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
    
    def one_batch_data_preprocess(self, data):
        real_A = []
        real_B = []
        real_Idt_A = []
        real_Idt_B = []
        for img, classid in zip(data[0], data[1]):
            if classid == 1:
                real_A.append(img.to(self.device))
                real_Idt_A.append(classid.to(self.device))
            else:
                real_B.append(img.to(self.device))
                real_Idt_B.append(classid.to(self.device))
        if len(real_A):
            self.real_A =  torch.stack(real_A)
            self.real_Idt_A = torch.stack(real_Idt_A)
        else:
            self.real_A = None
            self.real_Idt_A = None
        if len(real_B):
            self.real_B = torch.stack(real_B)
            self.real_Idt_B = torch.stack(real_Idt_B)
        else:
            self.real_B = None
            self.real_Idt_B = None
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        if self.real_A is not None:
            self.fake_B = self.G_AB(self.real_A)  # G_A(A)
            self.rec_A = self.G_BA(self.fake_B)   # G_B(G_A(A))
        if self.real_B is not None:
            self.fake_A = self.G_BA(self.real_B)  # G_B(B)
            self.rec_B = self.G_AB(self.fake_A)   # G_A(G_B(B))
    
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        if real is not None:
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        else:
            loss_D_real = None
        # Fake
        if fake is not None:
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        else:
            loss_D_fake = None
        # Combined loss and calculate gradients
        if (loss_D_real is not None) and (loss_D_fake is not None):
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        else:
            if loss_D_fake is not None:
                loss_D = loss_D_fake
            if loss_D_real is not None:
                loss_D = loss_D_real
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        # fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.D_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        # fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.D_B, self.real_A, self.fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            if self.real_B is not None:
                self.idt_A = self.G_AB(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            else:
                self.loss_idt_A = 0
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            if self.real_A is not None:
                self.idt_B = self.G_BA(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            else:
                self.loss_idt_B = 0
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        if self.real_A is not None:
            # GAN loss D_A(G_A(A))
            f_B_D_A = self.D_A(self.fake_B)
            self.loss_G_A = self.criterionGAN(f_B_D_A, torch.ones_like(f_B_D_A))
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        else:
            self.loss_G_A = 0
            self.loss_cycle_A = 0

        if self.real_B is not None:
            # GAN loss D_B(G_B(B))
            f_A_D_B = self.D_B(self.fake_A)
            self.loss_G_B = self.criterionGAN(f_A_D_B, torch.ones_like(f_A_D_B))
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        else:
            self.loss_G_B = 0
            self.loss_cycle_B = 0

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

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
    
    def get_current_loss(self):
        return self.loss_D_A, self.loss_D_B, self.loss_G #, self.loss_G_A, self.loss_G_B, self.loss_cycle_A, self.loss_cycle_B, self.loss_idt_A, self.loss_idt_B, 
    
    def train(self, epochs=50):
        for e in range(epochs):
            tbar = tqdm(self.Train_Data)
            for data in tbar:
                self.one_batch_data_preprocess(data)
                self.optimize_parameters()
                loss_D_A, loss_D_B, loss_G = self.get_current_loss()
                tbar.set_description(f'Epochs:{e+1}/{epochs}| loss_D_A={loss_D_A.item():.2f}, loss_D_B={loss_D_B.item():.2f}, loss_G={loss_G.item():.2f}')
            for sd in self.schedulers:
                sd.step()

    def test(self):
        pass

    def save_model(self, G_AB_PATH='./G_AB.pt', G_BA_PATH='./G_BA.pt', D_A_PATH='./D_A.pt', D_B_PATH='./D_B.pt'):
        torch.save(self.G_AB.state_dict(), G_AB_PATH)
        torch.save(self.G_BA.state_dict(), G_BA_PATH)
        if self.is_train:
            torch.save(self.D_A.state_dict(), D_A_PATH)
            torch.save(self.D_B.state_dict(), D_B_PATH)



import torch.nn.functional as F
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120*57*57, 84)
        self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, 120*57*57)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        return torch.sigmoid(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trans = T.Compose([
    # resize
    T.Resize((256,256)),
    # to-tensor
    T.ToTensor(),
    # normalize
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# c = Cycle(Generator(), Generator(), Discriminator(), Discriminator(), device=device)

# c.load_train_dataset('D:\\apple2orange\\train', 
#                 class_to_idx={'B':0, 'A':1},
#                 shuffle=True, 
#                 train_trans=trans)
# c.set_loss()
# c.set_optimizer_and_schedulers()
# c.train(epochs=20)
# c.save_model()

from PIL import Image
img_trans = T.ToPILImage()

apple_image = trans(Image.open('D:\\apple2orange\\train\\A\\n07740461_158.jpg')).to(device)
orange_image = trans(Image.open('D:\\apple2orange\\train\\B\\n07749192_183.jpg')).to(device)

apple_image = apple_image.unsqueeze(0)
orange_image = orange_image.unsqueeze(0)
test_G_AB = Generator()
test_G_BA = Generator()

test_G_AB.load_state_dict(torch.load('.\\G_AB.pt'))
test_G_BA.load_state_dict(torch.load('.\\G_BA.pt'))
test_G_AB.to(device)
test_G_BA.to(device)

fake_orange_image = test_G_AB(apple_image).squeeze()
fake_orange_image = img_trans(fake_orange_image)
fake_orange_image.save('./fake_orange.jpg')

fake_apple_image = test_G_BA(orange_image).squeeze()
fake_apple_image = img_trans(fake_apple_image)
fake_apple_image.save('./fake_apple.jpg')