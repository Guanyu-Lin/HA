import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, blocks, layers):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, blocks=blocks, layers=layers)
        self.model.to(device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, real_inc):
        self.model.train()
        # self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        # loss.backward()
        # if self.clip is not None:
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        # self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse, rmse_inc, rmse_non = util.masked_rmse(predict,real, real_inc,0.0)
        # rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse.item(), rmse_inc.item(), rmse_non.item()

    def eval(self, input, real_val, real_inc):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        real_inc_ = torch.unsqueeze(real_inc,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse, rmse_inc, rmse_non = util.masked_rmse(predict,real, real_inc_, 0.0)
        # rmse, rmse_inc, rmse_non = util.masked_rmse(predict,real, real_inc_, 0.0).item()
        return loss.item(),mape,rmse.item(), rmse_inc.item(), rmse_non.item()
