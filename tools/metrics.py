import torch
import torch.nn.functional as F


def RMSE(Y, Y_pred):  # b t input
    num = 0
    for y in Y:
        num += y.shape[0]  # batch number
    ground_truth = Y[0]
    predict = Y_pred[0]
    for i, y in enumerate(Y[1:]):
        ground_truth = torch.cat((ground_truth, y), 0)
        predict = torch.cat((predict, Y_pred[i + 1]), 0)
    g_r = ground_truth.permute(2, 1, 0)[:47]
    g_t = ground_truth.permute(2, 1, 0)[47:]
    y_r = predict.permute(2, 1, 0)[:47]
    y_t = predict.permute(2, 1, 0)[47:]
    g_r = torch.unsqueeze(g_r, 0)
    g_t = torch.unsqueeze(g_t, 0)
    y_r = torch.unsqueeze(y_r, 0)
    y_t = torch.unsqueeze(y_t, 0)
    l2_r = F.mse_loss(g_r, y_r)
    l2_t = F.mse_loss(g_t, y_t)
    l1_r = F.l1_loss(g_r, y_r)
    l1_t = F.l1_loss(g_t, y_t)
    return l1_r.item(), l1_t.item(), l2_r.item(), l2_t.item()

def accuracy(ground_truth, Pre_yc, mask):  # P: t  b  input(4)   mask:b t input
    # print(ground_truth.tolist())
    # print(Pre_yc.tolist())
    correct = Pre_yc.eq(ground_truth.view_as(Pre_yc))
    # print(correct.tolist())

    correct = torch.mul(correct.type(torch.cuda.FloatTensor), mask).sum()
    # print(correct.tolist())
    # print()
    base = mask.sum()
    accu = correct.item() / base.item()
    return accu