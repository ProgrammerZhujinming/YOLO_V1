import torch
from YOLO_V1_DataSet import YoloV1DataSet
dataSet = YoloV1DataSet()
from torch.utils.data import DataLoader
dataLoader = DataLoader(dataSet,batch_size=32,shuffle=True,num_workers=4)

from YOLO_v1_Model import YOLO_V1
Yolo = YOLO_V1().cuda()

from YOLO_V1_LossFunction import  Yolov1_Loss
loss_function = Yolov1_Loss().cuda()

import torch.optim as optim
optimizer = optim.Adam(Yolo.parameters(),lr=1e-4)

loss_sum = 0
for epoch in range(2000*dataSet.Classes):
    for batch_index, batch_train in enumerate(dataLoader):
        train_data = torch.Tensor(batch_train[0]).float().cuda()
        train_data.requires_grad = True
        label_data = torch.Tensor(batch_train[1]).float().cuda()
        loss = loss_function(bounding_boxes=Yolo(train_data),ground_truth=label_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss
        print("batch_index : {} ; batch_loss : {}".format(batch_index,loss.item()))
    if epoch % 100 == 0:
        torch.save(Yolo, './YOLO_V1.pkl')
    print("epoch : {} ; loss : {}".format(epoch,{loss_sum/(epoch+1)}))

