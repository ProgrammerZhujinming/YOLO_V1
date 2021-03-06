import torch.nn as nn

class YOLO_V1(nn.Module):

    def __init__(self,B=2,Classes_Num=20):
        super(YOLO_V1,self).__init__()
        self.B = B
        self.Classes_Num = Classes_Num

        self.Conv_448 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(2,2)
        )

        self.Conv_112 = nn.Sequential(
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )

        self.Conv_56 = nn.Sequential(
            nn.Conv2d(192, 128, 1, 1, 0),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )

        self.Conv_28 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512,512,1,1,0),
            nn.Conv2d(512,1024,3,1,1),
            nn.MaxPool2d(2, 2)
        )

        self.Conv_14 = nn.Sequential(
            nn.Conv2d(1024,512,1,1,0),
            nn.Conv2d(512,1024,3,1,1),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.Conv2d(1024, 1024, 3, 2, 1),
        )

        self.Conv_7 = nn.Sequential(
            nn.Conv2d(1024,1024,3,1,1),
            nn.Conv2d(1024, 1024, 3, 1, 1)
        )

        self.Fc = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.Linear(4096,7 * 7 * (B*5 + Classes_Num)),
            nn.Sigmoid()
        )



    def forward(self, x):
        # PyTorch的卷积要求的输入是 batch_size * channels * height * width
        x = x.permute(0,3,1,2)
        x = self.Conv_448(x)
        x = self.Conv_112(x)
        x = self.Conv_56(x)
        x = self.Conv_28(x)
        x = self.Conv_14(x)
        x = self.Conv_7(x)
        x = x.view(-1,7*7*1024)
        x = self.Fc(x)
        x = x.view((-1,7,7,(self.B*5 + self.Classes_Num)))
        return x

