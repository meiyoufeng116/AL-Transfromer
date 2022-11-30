from bisect import bisect
import torch
import torch.nn.functional as F
import torch.nn as nn
from tools.metrics import accuracy
import numpy as np

def evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask, is_mask, response_size=47, treatment_size=2):

    length = data_y.shape[1]
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()

    pred_r, pred_t = model(data_x_r, data_x_t, is_train=True, gen_length=length)
    if is_mask:
        pred_r = torch.mul(pred_r, mask[:, :, :response_size])
        pred_t = torch.mul(pred_t, mask[:, :, -treatment_size:])
    eval_mse_r = loss_mse(data_y[:, :, :response_size], pred_r).item()
    eval_mse_t = loss_mse(data_y[:, :, -treatment_size:], pred_t).item()
    eval_mae_r = loss_mae(data_y[:, :, :response_size], pred_r).item()
    eval_mae_t = loss_mae(data_y[:, :, -treatment_size:], pred_t).item()

    return eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t

def evaluate_flat_lstm(model, data_x_r, data_x_t,data_x ,mask, is_mask, response_size=47, treatment_size=2):

    length = data_x.shape[1]
    all_len=24
    middle_len=12
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()

    pred_r, pred_t = model(data_x_r, data_x_t, is_train=False, gen_length=12)
    if is_mask:
        pred_r = torch.mul(pred_r, mask[:, middle_len:all_len, :response_size])
        pred_t = torch.mul(pred_t, mask[:, middle_len:all_len, -treatment_size:])
    eval_mse_r = loss_mse(data_x[:, middle_len:all_len, :response_size], pred_r[:,middle_len:all_len,:]).item()
    eval_mse_t = loss_mse(data_x[:, middle_len:all_len, -treatment_size:], pred_t[:,middle_len:all_len,:]).item()
    eval_mae_r = loss_mae(data_x[:, middle_len:all_len, :response_size], pred_r[:,middle_len:all_len,:]).item()
    eval_mae_t = loss_mae(data_x[:, middle_len:all_len, -treatment_size:], pred_t[:,middle_len:all_len,:]).item()

    return eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t

def evaluate_simulate(model, data_x_r, data_x_t,data_x ,mask, is_mask, response_size=47, treatment_size=2,Bias=None):
    
    length = data_x.shape[1]
    all_len=24
    half_len=12
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()

    pred_r, pred_t = model(data_x_r, data_x_t,is_train=False, gen_length=12,bias=Bias)
    if is_mask:
        pred_r = torch.mul(pred_r, mask[:, :, :response_size])
        pred_t = torch.mul(pred_t, mask[:, :, -treatment_size:])
    eval_mse_r = loss_mse(data_x[:, half_len:all_len, :response_size], pred_r[:,half_len:all_len,:]).item()
    eval_mse_t = loss_mse(data_x[:, half_len:all_len, -treatment_size:], pred_t[:,half_len:all_len,:]).item()
    eval_mae_r = loss_mae(data_x[:, half_len:all_len, :response_size], pred_r[:,half_len:all_len,:]).item()
    eval_mae_t = loss_mae(data_x[:, half_len:all_len, -treatment_size:], pred_t[:,half_len:all_len,:]).item()
    bias=data_x[:, half_len:all_len, :response_size]-pred_r[:,half_len:all_len,:]

    return eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t,bias,pred_r,pred_t

def evaluate_timesetps(model, data_x_r, data_x_t,data_x ,mask, is_mask, response_size=47, treatment_size=2):
    
    length = data_x.shape[1]
    all_len=36
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()

    pred_r, pred_t = model(data_x_r, data_x_t, is_train=False, gen_length=24)
    if is_mask:
        pred_r = torch.mul(pred_r, mask[:, :, :response_size])
        pred_t = torch.mul(pred_t, mask[:, :, -treatment_size:])
    eval_mse_r = loss_mse(data_x[:, 24:all_len, :response_size], pred_r[:,24:all_len,:]).item()
    eval_mse_t = loss_mse(data_x[:, 24:all_len, -treatment_size:], pred_t[:,24:all_len,:]).item()
    eval_mae_r = loss_mae(data_x[:, 24:all_len, :response_size], pred_r[:,24:all_len,:]).item()
    eval_mae_t = loss_mae(data_x[:, 24:all_len, -treatment_size:], pred_t[:,24:all_len,:]).item()
    # bias=data_x[:, 24:all_len, :response_size]-pred_r[:,24:all_len,:]

    return eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t,pred_r,pred_t




# calculate loss for response and treatment respectively
def evaluate(model, data_x,x_mask, data_y, mask, response_size=47, treatment_size=2):

    
    length = data_y.shape[1]
    all_len=48
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()

    pred = model(data_x,mask,data_x,mask, is_train=False, out_len=all_len)
    pred = torch.mul(pred, mask[:,24:all_len,:])
    eval_mse_r = loss_mse(data_y[:, 24:all_len, :response_size], pred[:, :, :response_size]).item()
    eval_mse_t = loss_mse(data_y[:, 24:all_len, -treatment_size:], pred[:, :, -treatment_size:]).item()
    eval_mae_r = loss_mae(data_y[:, 24:all_len, :response_size], pred[:, :, :response_size]).item()
    eval_mae_t = loss_mae(data_y[:, 24:all_len, -treatment_size:], pred[:, :, -treatment_size:]).item()

    return eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t,pred
def evaluate_simu_t(model, data_x, data_y, mask, response_size=47, treatment_size=2,bias=None):
    
    
    length = data_y.shape[1]
    all_len=48
    middle_len=24
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()

    pred = model(data_x,mask,data_x,mask, is_train=False, out_len=middle_len,bias=bias)
    pred = torch.mul(pred, mask[:,middle_len:all_len,:])
    eval_mse_r = loss_mse(data_y[:, middle_len:all_len, :response_size], pred[:, :, :response_size]).item()
    eval_mse_t = loss_mse(data_y[:, middle_len:all_len, -treatment_size:], pred[:, :, -treatment_size:]).item()
    eval_mae_r = loss_mae(data_y[:, middle_len:all_len, :response_size], pred[:, :, :response_size]).item()
    eval_mae_t = loss_mae(data_y[:, middle_len:all_len, -treatment_size:], pred[:, :, -treatment_size:]).item()
    bias=(data_y[:, middle_len:all_len, :response_size]-pred[:,:, :response_size])
    return eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t,bias,pred

def evaluate_separate_lstm(model_r, model_t, data_x, data_y, mask, is_mask, response_size=47, treatment_size=2):

    length = data_y.shape[1]
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()

    pred_r, _ = model_r(data_x[:, :, :response_size], is_train=False, gen_length=length)
    pred_t, _ = model_t(data_x[:, :, -treatment_size:], is_train=False, gen_length=length)
    if is_mask:
        pred_r = torch.mul(pred_r, mask[:, :, :response_size])
        pred_t = torch.mul(pred_t, mask[:, :, -treatment_size:])
    eval_mse_r = loss_mse(data_y[:, :, :response_size], pred_r).item()
    eval_mse_t = loss_mse(data_y[:, :, -treatment_size:], pred_t).item()
    eval_mae_r = loss_mae(data_y[:, :, :response_size], pred_r).item()
    eval_mae_t = loss_mae(data_y[:, :, -treatment_size:], pred_t).item()

    return eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t



def evaluate_adrnn_cnn(model, C1, C2, data_x_r, data_x_t, data_y, mask, loss_fn, device):  # x (batch, time_step, input_size)
    batch = data_x_r.shape[0]
    length = data_y.shape[1]

    ground_truth = data_y[:, :, 47:].permute(1, 0, 2)
    ground_mask = mask[:, :, 47:].permute(1, 0, 2)

    Pre_y_r = torch.zeros(length, batch, data_x_r.shape[2]).to(device)
    Pre_y_t = torch.zeros(length, batch, data_x_t.shape[2]).to(device)
    Pre_yc = torch.zeros(length, batch, ground_truth.shape[2]).to(device)  # t b input(4)
    for i in range(length):
        y_r, y_t = model(data_x_r, data_x_t)  # y_t b t 4
        # time_step batch_size feature
        C_input = torch.cat((data_x_r.permute(1, 0, 2), data_x_t.permute(1, 0, 2)), 2)
        # 使用前13个time_step进行推断
        C_input = C_input[-13:]  # t[-24:] b input
        # batch_size time_step feature
        C_input = C_input.permute(1, 0, 2)  # t[-24:] b input
        # batch_size time_step 2
        Yc1_pred = C1(C_input)
        # batch_size time_step 2
        Yc2_pred = C2(C_input)
        # batch_size time_step 1
        # index of the max value
        Yc1_pred = Yc1_pred.max(2, keepdim=True)[1]  # F.log_softmax(Yc1_pred)
        Yc2_pred = Yc2_pred.max(2, keepdim=True)[1]  # F.log_softmax(Yc2_pred)
        # time_step batch_size 2
        Yc_pred = torch.cat((Yc1_pred, Yc2_pred), 2).permute(1, 0, 2).float()
        # batch_size 2
        Yc_pred = Yc_pred[-1]  # b 2
        Pre_yc[i] = Yc_pred
        Yc_pred = Yc_pred.to(device)
        # Yc_pred = torch.cat((Yc_pred.type(torch.FloatTensor).cuda(), torch.ones(Yc_pred.shape[0],Yc_pred.shape[1]).cuda()),1)
        # 将当前的预测值添加到已知样本当中
        data_x_r = torch.cat((data_x_r, y_r.permute(1, 0, 2)[-1].view(batch, 1, -1)), 1)
        data_x_t = torch.cat((data_x_t, y_t.permute(1, 0, 2)[-1].view(batch, 1, -1)), 1)
        Pre_y_r[i] = y_r.permute(1, 0, 2)[-1].view(batch, -1)
        # 当前预测的treatment加上CNN惩戒值
        Pre_y_t[i] = torch.mul(y_t.permute(1, 0, 2)[-1].view(batch, -1), Yc_pred)
    # batch_size time_step feature
    Pre_y_r = Pre_y_r.permute(1, 0, 2)
    # batch_size time_step feature
    Pre_y_t = Pre_y_t.permute(1, 0, 2)

    pre_y = torch.cat((Pre_y_r, Pre_y_t), 2)
    pre_y = torch.mul(pre_y, mask)
    eval_loss_r = loss_fn(data_y.permute(2, 1, 0)[:47], pre_y.permute(2, 1, 0)[:47])
    eval_loss_t = loss_fn(data_y.permute(2, 1, 0)[47:], pre_y.permute(2, 1, 0)[47:])
    accu_r = accuracy(ground_truth.permute(2, 1, 0)[0], Pre_yc.permute(2, 1, 0)[0], ground_mask.permute(2, 1, 0)[0])
    accu_t = accuracy(ground_truth.permute(2, 1, 0)[1], Pre_yc.permute(2, 1, 0)[1], ground_mask.permute(2, 1, 0)[1])
    return eval_loss_r, eval_loss_t, accu_r, accu_t


def evaluate_adrnn_cnn_multilabel(rnn, cnn, data_x_r, data_x_t, data_y, mask, is_mask, window, response_size=47, treatment_size=2,
                                  device=torch.device('cpu')):  # x (batch, time_step, input_size)
    batch = data_x_r.shape[0]
    length = data_y.shape[1]
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()
    loss_bce = nn.BCELoss()
    # time_step batch_size feature
    ground_truth = data_y[:, :, -treatment_size:].permute(1, 0, 2)
    # time_step batch_size feature
    ground_mask = mask[:, :, -treatment_size:].permute(1, 0, 2)
    # gen_length batch_size response_size
    pre_y_r = torch.zeros(length, batch, data_x_r.shape[2]).to(device)
    # gen_length batch_size treat_size
    pre_y_t = torch.zeros(length, batch, data_x_t.shape[2]).to(device)
    # gen_length batch_size num_classes
    pre_yc = torch.zeros(length, batch, ground_truth.shape[2]).to(device)

    for i in range(length):
        y_r, y_t = rnn(data_x_r, data_x_t, is_train=False, gen_length=length)  # y_t b t 4
        # time_step batch_size response_size + treat_size
        c_input = torch.cat((data_x_r.permute(1, 0, 2), data_x_t.permute(1, 0, 2)), 2)  # t b input
        # 只使用最近的window个time_step
        c_input = c_input[-window:]  # t[-24:] b input
        # batch_size time_step response_size + treat_size
        c_input = c_input.permute(1, 0, 2)  # t[-24:] b input
        # batch_size time_step num_classes
        yc1_pred = cnn(c_input)

        yc_pred = (yc1_pred > 0.5).long()
        # time_step batch_size num_classes
        yc_pred = yc_pred.permute(1, 0, 2)

        yc_pred = yc_pred[-1]
        # the CNN penalty of current time_step
        pre_yc[i] = yc_pred
        # Yc_pred = torch.cat((Yc_pred.type(torch.FloatTensor).cuda(), torch.ones(Yc_pred.shape[0],Yc_pred.shape[1]).cuda()),1)

        data_x_r = torch.cat((data_x_r, y_r[:, -1:, :]), 1)
        data_x_t = torch.cat((data_x_t, y_t[:, -1:, :]), 1)
        # prediction of response and treatment of current time step
        pre_y_r[i] = y_r[:, -1, :]
        pre_y_t[i] = torch.mul(y_t[:, -1, :], yc_pred.float())

    # batch_size time_step response_size
    pre_y_r = pre_y_r.permute(1, 0, 2)
    # batch_size time_step treat_size
    pre_y_t = pre_y_t.permute(1, 0, 2)
    # batch_size time_step response_size + treat_size
    pre_y = torch.cat((pre_y_r, pre_y_t), 2)
    if is_mask:
        pre_y = torch.mul(pre_y, mask)

    eval_mse_r = loss_mse(data_y[:, :, :response_size], pre_y[:, :, :response_size]).item()
    eval_mse_t = loss_mse(data_y[:, :, -treatment_size:], pre_y[:, :, -treatment_size:]).item()

    eval_mae_r = loss_mae(data_y[:, :, :response_size], pre_y[:, :, :response_size]).item()
    eval_mae_t = loss_mae(data_y[:, :, -treatment_size:], pre_y[:, :, -treatment_size:]).item()
    # accu_r = accuracy(ground_truth.permute(2, 1, 0)[0], pre_yc.permute(2, 1, 0)[0], ground_mask.permute(2, 1, 0)[0])
    # accu_t = accuracy(ground_truth.permute(2, 1, 0)[1], pre_yc.permute(2, 1, 0)[1], ground_mask.permute(2, 1, 0)[1])
    cnn_target = data_y[:, :, -treatment_size:].to(device)
    cnn_target = (cnn_target > 0).float()
    eval_cnn = loss_bce(pre_yc, cnn_target.permute(1, 0, 2)).item()

    return eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t, eval_cnn

def evaluate_adrnn_to_cnn_multiLabel(model, C1, data_x_r, data_x_t, data_y, mask,
                                     loss_fn, device=torch.device("cpu")):  # x (batch, time_step, input_size)

    length = data_y.shape[1]
    ground_truth = data_y[:, :, 47:].permute(1, 0, 2)
    ground_truth = (ground_truth > 0).long()

    ground_mask = mask[:, :, 47:].permute(1, 0, 2)

    pre_y_r, pre_y_t = model(data_x_r, data_x_t, is_train=False, gen_length=length)
    pre_y = torch.cat((pre_y_r, pre_y_t), 2)

    # time_step batch_size response_size + treat_size
    c_input = torch.cat((data_x_r.permute(1, 0, 2), data_x_t.permute(1, 0, 2)), 2)  # t b input

    # 训练值加上预测值
    # time_step batch_size response_size + treat_size
    c_input = torch.cat((c_input, pre_y.permute(1, 0, 2)), 0)

    # 使用训练值中最新的12个time step加上预测值
    c_input = c_input[-(length + 12):-1]  # t[-24:] b input
    c_input = c_input.permute(1, 0, 2)  # t[-24:] b input
    # batch_size time_step num_classes
    yc1_pred = C1(c_input)

    yc_pred = (yc1_pred > 0.2).long()
    # batch_size time_step num_classes
    yc_pred = yc_pred.float().to(device)

    accu_r = accuracy(ground_truth.permute(2, 1, 0)[0], yc_pred.permute(2, 0, 1)[0], ground_mask.permute(2, 1, 0)[0])
    accu_t = accuracy(ground_truth.permute(2, 1, 0)[1], yc_pred.permute(2, 0, 1)[1], ground_mask.permute(2, 1, 0)[1])

    pre_y = torch.mul(pre_y, mask)
    eval_loss_r = loss_fn(data_y[:, :, :47], pre_y[:, :, :47])


    # print("eval_loss_t_old" + str(eval_loss_t_old.item()))
    p = pre_y.permute(2, 1, 0)[47:]
    # print(p.tolist())
    pre_y = torch.mul(pre_y.permute(2, 1, 0)[47:], yc_pred.permute(2, 1, 0))
    eval_loss_t = loss_fn(data_y.permute(2, 1, 0)[47:], pre_y)
    # print("eval_loss_t" + str(eval_loss_t.item()))
    # print(Pre_y.tolist())
    # print(data_y.permute(2,1,0)[47:].tolist())

    # print()
    return eval_loss_r, eval_loss_t, accu_r, accu_t
