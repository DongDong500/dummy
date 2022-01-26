import torch
import torch.nn as nn
import torchvision
import pytorch_model_summary

class DUMMY(nn.Module):
    """
    The Dummy model
    """

    def __init__(self) -> None:
        super(DUMMY, self).__init__()

        self.fc1 = nn.Linear(3*1920*1080, 64)
        self.fc2 = nn.Linear(64, 4)


    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    
    inputs = torch.rand(10, 3*1920*1080)
    net = DUMMY()

    outputs = net(inputs)
    print(outputs.size())

    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.data.size())

    print(pytorch_model_summary.summary(net, torch.zeros(100, 3*1920*1080), show_input=True))

    print(net.parameters())

