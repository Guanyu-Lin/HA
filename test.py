import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/TSMO',help='data path')
# parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=3,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=583,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,help='')
parser.add_argument('--plotheatmap',type=str,default='False',help='')
parser.add_argument('--blocks',type=int,default=4,help='')
parser.add_argument('--layers',type=int,default=2,help='')

args = parser.parse_args()




def main():
    device = torch.device(args.device)

    # _, _, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    # supports = [torch.tensor(i).to(device) for i in adj_mx]
    supports = None
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]
    # adjinit = None

    if args.aptonly:
        supports = None
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    # engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
    #                 args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
    #                 adjinit, args.blocks,args.layers)
    # model = engine.model
    model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16, blocks=args.blocks, layers=args.layers)
    # model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, blocks=args.blocks, layers=args.layers)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()


    print('model load successfully')


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    real_inc = realy.transpose(1,3)[:,2,:,:]
    # import pdb
    # pdb.set_trace()
    realy = realy.transpose(1,3)[:,0,:,:]
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    amae = []
    amape = []
    armse = []
    armse_inc = []
    armse_non = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        inc = real_inc[:,:,i]
        metrics = util.metric(pred,real, inc)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test RMSE Incident: {:.4f}, Test RMSE Non-incident: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        armse_inc.append(metrics[3])
        armse_non.append(metrics[4])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test RMSE Incident: {:.4f}, Test RMSE Non-incident: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse),np.mean(armse_inc),np.mean(armse_non)))


    if args.plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig("./emb"+ '.pdf')
    link = 0
    y12 = realy[:,link,11].cpu().detach().numpy()
    inc12 = real_inc[:,link,11].cpu().detach().numpy()
    yhat12 = scaler.inverse_transform(yhat[:,link,11]).cpu().detach().numpy()

    y3 = realy[:,link,2].cpu().detach().numpy()
    inc3 = real_inc[:,link,2].cpu().detach().numpy()
    yhat3 = scaler.inverse_transform(yhat[:,link,2]).cpu().detach().numpy()

    df2 = pd.DataFrame({'real12':y12,'inc12':inc12,'pred12':yhat12, 'real3': y3,'inc3':inc3, 'pred3':yhat3})
    df2.to_csv('./wave.csv',index=False)


if __name__ == "__main__":
    main()
