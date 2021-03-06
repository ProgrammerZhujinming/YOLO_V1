import torch.nn as nn
import math
import torch

class Yolov1_Loss(nn.Module):

    def __init__(self, S=7, B=2, Classes=2, l_coord=5, l_noobj=0.5):
        # 有物体的box损失权重设为l_coord,没有物体的box损失权重设置为l_noobj
        super(Yolov1_Loss, self).__init__()
        self.S = S
        self.B = B
        self.Classes = Classes
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def iou(self, bounding_box, ground_box, gridX, gridY, img_size=448, grid_size=64):  # 计算两个box的IoU值
        # predict_box: [centerX, centerY, width, height]
        # ground_box : [centerX/self.grid_cell_size,centerY/self.grid_cell_size,(xmax-xmin)/self.grid_cell_size,(ymax-ymin)/self.grid_cell_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)]
        # 1.  预处理 predict_box  变为  左上X,Y  右下X,Y  两个边界点的坐标 避免浮点误差 先还原成整数
        # 不要共用引用
        predict_box = list([0,0,0,0])
        predict_box[0] = (int)(gridX + bounding_box[0] * grid_size)
        predict_box[1] = (int)(gridY + bounding_box[1] * grid_size)
        predict_box[2] = (int)(bounding_box[2] * img_size)
        predict_box[3] = (int)(bounding_box[3] * img_size)

        # [xmin,ymin,xmax,ymax]
        predict_coord = list([max(0, predict_box[0] - predict_box[2] / 2), max(0, predict_box[1] - predict_box[3] / 2),min(img_size - 1, predict_box[0] + predict_box[2] / 2), min(img_size - 1, predict_box[1] + predict_box[3] / 2)])
        predict_Area = (predict_coord[2] - predict_coord[0]) * (predict_coord[3] - predict_coord[1])

        ground_coord = list([ground_box[5],ground_box[6],ground_box[7],ground_box[8]])
        ground_Area = (ground_coord[2] - ground_coord[0]) * (ground_coord[3] - ground_coord[1])

        # 存储格式 xmin ymin xmax ymax

        # 2.计算交集的面积 左边的大者 右边的小者 上边的大者 下边的小者
        CrossLX = max(predict_coord[0], ground_coord[0])
        CrossRX = min(predict_coord[2], ground_coord[2])
        CrossUY = max(predict_coord[1], ground_coord[1])
        CrossDY = min(predict_coord[3], ground_coord[3])

        if CrossRX < CrossLX or CrossDY < CrossUY: # 没有交集
            return 0

        interSection = (CrossRX - CrossLX + 1) * (CrossDY - CrossUY + 1)
        return interSection / (predict_Area + ground_Area - interSection)

    def forward(self, bounding_boxes, ground_truth): # 输入是 S * S * ( 2 * B + Classes)
        loss_coord = torch.Tensor([0]).requires_grad_()
        loss_confidence = torch.Tensor([0]).requires_grad_()
        loss_classes = torch.Tensor([0]).requires_grad_()
        object_num = 0
        iou_sum = 0
        noobject_num = 0
        for batch in range(len(bounding_boxes)):
            for i in range(self.S):
                for j in range(self.S):
                    exist = {}
                    for t in range(self.B): # 标记还没有分配的box序号
                        exist[t] = 1
                    # 为每一个ground_box找到最大iou的predict_box
                    for k in range(len(ground_truth[batch][i][j])):
                        max_iou = 0
                        max_iou_index = -1
                        if ground_truth[batch][i][j][k][9] == 0: # 面积为0的grount_truth 为了形状相同强行拼接的无用的0-box
                            break
                        # 遍历每一个当前grid cell内部的预测框
                        for t in range(self.B):
                            if exist[t] == 1: # 如果当前框还没有分配
                                predict = bounding_boxes[batch][i][j][5 * t : 5 * t + 4]
                                ground = ground_truth[batch][i][j][k]
                                iou = self.iou(predict,ground,i*64, j*64)
                                if max_iou < iou:
                                    max_iou = iou
                                    max_iou_index = t
                        # 找不到和ground_box有交集的预测框
                        if max_iou_index == -1:
                            continue
                        #正样本的损失计算
                        object_num = object_num + 1
                        iou_sum = iou_sum + max_iou
                        ground_box = ground_truth[batch][i][j][k]
                        predict_box = bounding_boxes[batch][i][j][max_iou_index*5 : max_iou_index*5 + 5]
                        #print(predict_box)
                        loss_coord = loss_coord + self.l_coord * (math.pow(ground_box[0] - predict_box[0],2) + math.pow(ground_box[1] - predict_box[1],2) + math.pow(math.sqrt(ground_box[2])-math.sqrt(predict_box[2]),2) + math.pow(math.sqrt(ground_box[3])-math.sqrt(predict_box[3]),2))
                        loss_confidence = loss_confidence + math.pow(ground_box[4]-predict_box[4],2)
                        ground_class = ground_box[10:]
                        predict_class = bounding_boxes[batch][i][j][self.B * 5:]
                        for i in range(self.Classes):
                            loss_classes = loss_classes + self.l_noobj * math.pow(ground_class[i] - predict_class[i],2)
                        exist[max_iou_index] = 0

                    #所有没有分配到ground_box的预测框都是负样本 负样本只有置信度损失
                    for key in exist:
                        noobject_num = noobject_num + 1
                        predict_box = bounding_boxes[batch][i][j][key*5:key*5+5]
                        loss_confidence = loss_confidence + self.l_noobj * math.pow(predict_box[4],2)
        print("坐标误差:{} 置信度误差:{} 类别损失:{}".format(loss_coord.item(),loss_confidence.item(),loss_classes.item()))
        print("iou:{} confidence:{}".format("nan" if object_num == 0 else (iou_sum/object_num).item(),(loss_confidence/(object_num + noobject_num)).item()))
        return loss_coord + loss_confidence + loss_classes