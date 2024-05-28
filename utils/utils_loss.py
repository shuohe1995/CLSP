import torch
import torch.nn.functional as F
import torch.nn as nn

class partial_loss(nn.Module):
    def __init__(self,confidence,partialY,lw_weight0,lw_weight):
        super().__init__()
        self.confidence = confidence
        self.partialY = partialY
        self.lw_weight0 = lw_weight0
        self.lw_weight = lw_weight
    def forward(self, outputs, index, type='rc'):
        #all_confidence=torch.cat([self.confidence[index],self.confidence[index]])
        all_confidence=self.confidence[index] # single image
        ###rc
        if type=='rc':
            logsm_outputs = F.log_softmax(outputs, dim=1)
            final_outputs = logsm_outputs * all_confidence
            loss = - ((final_outputs).sum(dim=1)).mean()

        elif type=='cc':
            ###cc
            sm_outputs = F.softmax(outputs, dim=1)
            final_outputs = sm_outputs * all_confidence
            loss = - torch.log(final_outputs.sum(dim=1)).mean()
        elif type=='lwc':
            #partialY=torch.cat([self.partialY[index],self.partialY[index]])
            partialY=self.partialY[index]
            onezero = torch.zeros_like(outputs)
            onezero[partialY > 0] = 1
            counter_onezero = 1 - onezero

            sm_outputs = F.softmax(outputs, dim=1)

            sig_loss1 = - torch.log(sm_outputs + 1e-8)
            l1 = all_confidence * onezero * sig_loss1
            average_loss1 = torch.sum(l1) / l1.size(0)

            sig_loss2 = - torch.log(1 - sm_outputs + 1e-8)
            l2 = all_confidence * counter_onezero * sig_loss2
            average_loss2 = torch.sum(l2) / l2.size(0)

            loss = self.lw_weight0 * average_loss1 + self.lw_weight * average_loss2

        return loss

    # def confidence_update(self,outputs1,outputs2,batchY,batch_index):
    #     with torch.no_grad():
    #         sm_outputs = (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1)) / 2
    #         sm_outputs *= batchY
    #         new_batch_confidence = sm_outputs / sm_outputs.sum(dim=1, keepdim=True)
    #         self.confidence[batch_index]=new_batch_confidence
    #     return None

    def confidence_update(self,outputs, batchY,batch_index):
        with torch.no_grad():
            sm_outputs = torch.softmax(outputs, dim=1)
            sm_outputs *= batchY
            new_batch_confidence = sm_outputs / sm_outputs.sum(dim=1, keepdim=True)
            self.confidence[batch_index] = new_batch_confidence
        return None
    def confidence_update_lw(self,batch_outputs, confidence, batch_index):
        with torch.no_grad():
            sm_outputs = F.softmax(batch_outputs, dim=1)
            onezero = torch.zeros_like(sm_outputs)
            onezero[self.partialY[batch_index] > 0] = 1
            counter_onezero = 1 - onezero

            new_weight1 = sm_outputs * onezero
            new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
                confidence.shape[1], 1).transpose(0, 1)
            new_weight2 = sm_outputs * counter_onezero
            new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
                confidence.shape[1], 1).transpose(0, 1)
            new_weight = new_weight1 + new_weight2

            self.confidence[batch_index] = new_weight
            return confidence

