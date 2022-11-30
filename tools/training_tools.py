import codecs
import imp
import os
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics

from tools.draw_lines import draw_lines
from tools.evaluate_tools import evaluate, evaluate_adrnn_cnn, \
    evaluate_adrnn_cnn_multilabel, evaluate_separate_lstm, evaluate_simulate, evaluate_stacked_lstm,evaluate_flat_lstm
from tools.metrics import RMSE
from tools.utils import EarlyStopping


def train_stacked_lstm_cnn(num_epochs, train_iterator,vaild_iterator,test_iterator, model, is_mask, window, logger, model_save_path, response_size=47, treatment_size=2,
                       device=torch.device("cpu")):
    model = model.to(device)
    logger.info(str(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16)
    loss_mse = nn.MSELoss()
    loss_bce = nn.BCELoss()
    
    start=0
    middle=12
    end=24

    early_stopping = EarlyStopping(verbose=True, logger=logger,patience=32)
    for curr_epoch in range(1, 1 + num_epochs):
        train_loss = 0
        train_loss_r = 0
        train_loss_t = 0
        model = model.train()
        num_batch = 0
        for b, batch in enumerate(tqdm(train_iterator)):
            num_batch += 1

            batch, mask= batch

            train_r = batch[:, :middle, :response_size].to(device)
            train_t = batch[:, :middle, -treatment_size:].to(device)
            y_train = batch[:, middle:end, :].to(device)

            pred_r, pred_t, pred_cnn = model(train_r, train_t,gen_length=12)


            if is_mask:
                mask = mask.to(device)
                pred = torch.mul(pred, mask[:, middle:end, :])

            loss_r = loss_mse(pred_r, y_train[:, :, :response_size])                                #bsz,73,47
            loss_t = loss_mse(pred_t, y_train[:, -pred_t.shape[1]:, -treatment_size:])              #bsz,73,2
            groundtruth = (y_train[:, -pred_cnn.shape[1]:, -treatment_size:] > 0).float()
            loss_cnn = loss_bce(pred_cnn, groundtruth)
            # loss = loss_r + loss_t + loss_cnn
            loss = loss_r + loss_t / (loss_t / loss_r).detach() + loss_cnn / (loss_cnn / loss_r).detach()

            train_loss = train_loss + loss.item()
            train_loss_r += loss_r.item()
            train_loss_t += loss_t.item()

            optimizer.zero_grad()

            loss.backward()
            # nn.utils.clip_grad_norm_(model.r_rnn.parameters(), max_norm=5, norm_type=2)
            # nn.utils.clip_grad_norm_(model.t_rnn.parameters(), max_norm=5, norm_type=2)
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        # scheduler.step()
        train_loss /= num_batch
        train_loss_r /= num_batch
        train_loss_t /= num_batch
        log_info = "epoch {} train_loss {} train_loss_response {} train_loss_treatment {}".format(
            curr_epoch, round(train_loss, 5), round(train_loss_r, 5),
            round(train_loss_t, 5))

        logger.info(log_info)

        # evaluation
        num_batch = 0
        valid_mse_r = 0
        valid_mse_t = 0
        valid_mae_r = 0
        valid_mae_t = 0

        model = model.eval()
        with torch.no_grad():

            for b, batch in enumerate(vaild_iterator):
                num_batch += 1
                data_x, mask= batch

                data_x_r = data_x[:, :, :response_size].to(device)
                data_x_t = data_x[:, :, -treatment_size:].to(device)
                # data_y = data_y.to(device)
                mask = mask.to(device)
                data_x=data_x.to(device)

                # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask,
                #                                                             is_mask,
                #                                                             response_size, treatment_size)
                
                eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_flat_lstm(model, data_x_r, data_x_t, data_x, mask,
                                                                            is_mask,
                                                                            response_size, treatment_size)                

                valid_mse_r += eval_mse_r
                valid_mse_t += eval_mse_t
                valid_mae_r += eval_mae_r
                valid_mae_t += eval_mae_t   
            valid_mse_r /= num_batch
            valid_mse_t /= num_batch
            valid_mae_r /= num_batch
            valid_mae_t /= num_batch
            geometric_mean = np.exp(np.log([valid_mse_r, valid_mse_t, valid_mae_r, valid_mae_t]).mean())
            log_info = "epoch {} geometric_mean {} valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}". \
                format(curr_epoch, round(geometric_mean, 5), round(valid_mse_r, 5), round(valid_mse_t, 5),
                       round(valid_mae_r, 5), round(valid_mae_t, 5))
            early_stopping(geometric_mean, model, model_save_path)
            if early_stopping.early_stop:
                logger.info("early stopping")
                break
            logger.info(log_info)
            
        #test
        num_batch = 0
        valid_mse_r = 0
        valid_mse_t = 0
        valid_mae_r = 0
        valid_mae_t = 0

        model = model.eval()
        with torch.no_grad():

            for b, batch in enumerate(test_iterator):
                num_batch += 1
                data_x, mask= batch

                data_x_r = data_x[:, :, :response_size].to(device)
                data_x_t = data_x[:, :, -treatment_size:].to(device)
                # data_y = data_y.to(device)
                mask = mask.to(device)
                data_x=data_x.to(device)

                # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask,
                #                                                             is_mask,
                #                                                             response_size, treatment_size)
                
                eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_flat_lstm(model, data_x_r, data_x_t, data_x, mask,
                                                                            is_mask,
                                                                            response_size, treatment_size)                

                valid_mse_r += eval_mse_r
                valid_mse_t += eval_mse_t
                valid_mae_r += eval_mae_r
                valid_mae_t += eval_mae_t   
            valid_mse_r /= num_batch
            valid_mse_t /= num_batch
            valid_mae_r /= num_batch
            valid_mae_t /= num_batch
            geometric_mean = np.exp(np.log([valid_mse_r, valid_mse_t, valid_mae_r, valid_mae_t]).mean())
            log_info = "Test epoch {} geometric_mean {} valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}". \
                format(curr_epoch, round(geometric_mean, 5), round(valid_mse_r, 5), round(valid_mse_t, 5),
                       round(valid_mae_r, 5), round(valid_mae_t, 5))
            logger.info(log_info)
            
    # logger.info("\n")
    # logger.info("best_geometric_mean {}\n".format(round(early_stopping.val_loss_min, 5)))

    return model

def train_informer(num_epochs, train_iterator,vaild_iterator,test_iterator, model, is_mask, window, logger, model_save_path, response_size=47, treatment_size=2,
                       device=torch.device("cpu")):
    model = model.to(device)
    logger.info(str(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
    loss_mse = nn.MSELoss()
    loss_bce = nn.BCELoss()
    is_mask=True

    early_stopping = EarlyStopping(verbose=True, logger=logger,patience=15)
    for curr_epoch in range(1, 1 + num_epochs):
        train_loss = 0
        train_loss_r = 0
        train_loss_t = 0
        model = model.train()
        num_batch = 0
        for b, batch in enumerate(tqdm(train_iterator)):
            num_batch += 1

            batch, mask= batch

            train = batch[:, 0:24, :].to(device)
            mask_train=mask[:, 0:24, :].to(device)
            # train_t = batch[:, :-1, -treatment_size:].to(device)
            # mask_t=mask[:, :-1, -treatment_size:].to(device)
            y_train = batch[:, 24:48, :].to(device)

            pred_r = model(train,mask_train,train,mask_train,out_len=train.shape[1])


            if is_mask:
                mask = mask.to(device)
                pred_r = torch.mul(pred_r, mask[:, 24:48, :])

            loss = loss_mse(pred_r, y_train)                                #bsz,73,47
            # loss_t = loss_mse(pred_t, y_train[:, -pred_t.shape[1]:, -treatment_size:])              #bsz,73,2
            # groundtruth = (y_train[:, -pred_cnn.shape[1]:, -treatment_size:] > 0).float()
            # loss_cnn = loss_bce(pred_cnn, groundtruth)
            # loss = loss_r + loss_t + loss_cnn
            # loss = loss_r + loss_t / (loss_t / loss_r).detach() 

            train_loss = train_loss + loss.item()
            # train_loss_r += loss_r.item()
            # train_loss_t += loss_t.item()

            optimizer.zero_grad()

            loss.backward()
            # nn.utils.clip_grad_norm_(model.r_rnn.parameters(), max_norm=5, norm_type=2)
            # nn.utils.clip_grad_norm_(model.t_rnn.parameters(), max_norm=5, norm_type=2)

            optimizer.step()
        scheduler.step()
        train_loss /= num_batch
        train_loss_r /= num_batch
        train_loss_t /= num_batch
        log_info = "epoch {} train_loss {} ".format(
            curr_epoch, round(train_loss, 5))

        logger.info(log_info)

        # evaluation
        num_batch = 0
        valid_mse_r = 0
        valid_mse_t = 0
        valid_mae_r = 0
        valid_mae_t = 0

        model = model.eval()
        with torch.no_grad():

            for b, batch in enumerate(tqdm(vaild_iterator)):
                num_batch += 1
                data_x, mask= batch

                data_x_r = data_x[:, :, :response_size].to(device)
                data_x_t = data_x[:, :, -treatment_size:].to(device)
                data_x = data_x[:,:,:].to(device)
                mask = mask.to(device)
                data_y=data_x[:,:24,:].to(device)

                # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask,
                #                                                             is_mask,
                #                                                             response_size, treatment_size)
                
                eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t,_ = evaluate(model, data_y, mask, data_x, mask,                                                         
                                                                            response_size, treatment_size)                

                valid_mse_r += eval_mse_r
                valid_mse_t += eval_mse_t
                valid_mae_r += eval_mae_r
                valid_mae_t += eval_mae_t   
            valid_mse_r /= num_batch
            valid_mse_t /= num_batch
            valid_mae_r /= num_batch
            valid_mae_t /= num_batch
            geometric_mean = np.exp(np.log([valid_mse_r, valid_mse_t, valid_mae_r, valid_mae_t]).mean())
            log_info = "epoch {} geometric_mean {} valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}". \
                format(curr_epoch, round(geometric_mean, 5), round(valid_mse_r, 5), round(valid_mse_t, 5),
                       round(valid_mae_r, 5), round(valid_mae_t, 5))
            early_stopping(geometric_mean, model, model_save_path)
            if early_stopping.early_stop:
                logger.info("early stopping")
                break
            logger.info(log_info)


        #test
        num_batch = 0
        valid_mse_r = 0
        valid_mse_t = 0
        valid_mae_r = 0
        valid_mae_t = 0

        model = model.eval()
        with torch.no_grad():

            for b, batch in enumerate(test_iterator):
                num_batch += 1
                data_x, mask= batch

                data_x_r = data_x[:, :, :response_size].to(device)
                data_x_t = data_x[:, :, -treatment_size:].to(device)
                # data_y = data_y.to(device)
                mask = mask.to(device)
                data_x=data_x[:,:,:].to(device)
                data_y=data_x[:,:24,:].to(device)
                # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask,
                #                                                             is_mask,
                #                                                             response_size, treatment_size)
                
                eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t,_ = evaluate(model, data_y, mask, data_x, mask,                                                         
                                                                            response_size, treatment_size)                

                valid_mse_r += eval_mse_r
                valid_mse_t += eval_mse_t
                valid_mae_r += eval_mae_r
                valid_mae_t += eval_mae_t  
            valid_mse_r /= num_batch
            valid_mse_t /= num_batch
            valid_mae_r /= num_batch
            valid_mae_t /= num_batch
            geometric_mean = np.exp(np.log([valid_mse_r, valid_mse_t, valid_mae_r, valid_mae_t]).mean())
            log_info = "geometric_mean {} valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}". \
                format( round(geometric_mean, 5), round(valid_mse_r, 5), round(valid_mse_t, 5),
                        round(valid_mae_r, 5), round(valid_mae_t, 5))
        logger.info(log_info)
        logger.info("\n")
        logger.info("best_geometric_mean {}\n".format(round(geometric_mean, 5)))

    return model

def test(test_iterator,model,is_mask, window, logger, model_save_path, response_size=47, treatment_size=2,
                       device=torch.device("cpu")):
    model.load_state_dict(torch.load("trained_models/transformer/2022_07_16_06_42_35/model.pt")) 
    model=model.to(device)
    num_batch = 0
    valid_mse_r = 0
    valid_mse_t = 0
    valid_mae_r = 0
    valid_mae_t = 0

    model = model.eval()
    with torch.no_grad():

        for b, batch in enumerate(tqdm(test_iterator)):
            num_batch += 1
            data_x, mask= batch

            data_x_r = data_x[:, :, :response_size].to(device)
            data_x_t = data_x[:, :, -treatment_size:].to(device)
            # data_y = data_y.to(device)
            mask = mask.to(device)
            data_x=data_x.to(device)

            # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask,
            #                                                             is_mask,
            #                                                             response_size, treatment_size)
            
            eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_flat_lstm(model, data_x_r, data_x_t, data_x, mask,
                                                                        is_mask,
                                                                        response_size, treatment_size)                

            valid_mse_r += eval_mse_r
            valid_mse_t += eval_mse_t
            valid_mae_r += eval_mae_r
            valid_mae_t += eval_mae_t   
        valid_mse_r /= num_batch
        valid_mse_t /= num_batch
        valid_mae_r /= num_batch
        valid_mae_t /= num_batch
        geometric_mean = np.exp(np.log([valid_mse_r, valid_mse_t, valid_mae_r, valid_mae_t]).mean())
        log_info = "test geometric_mean {} valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}". \
            format( round(geometric_mean, 5), round(valid_mse_r, 5), round(valid_mse_t, 5),
                    round(valid_mae_r, 5), round(valid_mae_t, 5))
    logger.info(log_info)
    logger.info("\n")
    logger.info("best_geometric_mean {}\n".format(round(geometric_mean, 5)))
    

    
def test_informer(test_iterator,model,is_mask, window, logger, model_save_path, response_size=47, treatment_size=2,
                       device=torch.device("cpu")):
    model.load_state_dict(torch.load("./trained_models/transformer/2022_08_14_00_02_35/Informer_model.pt")) 
    model=model.to(device)
    num_batch = 0
    valid_mse_r = 0
    valid_mse_t = 0
    valid_mae_r = 0
    valid_mae_t = 0
    TN=0
    TP=0
    FN=0
    FP=0
    pred_t_n=[]
    t_n=[]
    model = model.eval()
    with torch.no_grad():

        for b, batch in enumerate(tqdm(test_iterator)):
            num_batch += 1
            data_x, mask= batch

            data_x_r = data_x[:, :24, :response_size].to(device)
            data_x_t = data_x[:, :24, -treatment_size:].to(device)
            # data_y = data_y.to(device)
            data_y=data_x[:,:24,:].to(device)
            mask = mask.to(device)
            data_x=data_x.to(device)

            # eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t = evaluate_stacked_lstm(model, data_x_r, data_x_t, data_y, mask,
            #                                                             is_mask,
            #                                                             response_size, treatment_size)
            
            eval_mse_r, eval_mse_t, eval_mae_r, eval_mae_t,pred = evaluate(model, data_y, mask, data_x, mask,                                                         
                                                                        response_size, treatment_size)                



            for i in range(batch[0].size(0)):
                treatment=data_x[i, 24:48, -treatment_size:].nonzero()
                pred_t=pred[i,:,-treatment_size:]>=0.05
                treatment_pred=pred_t[:,-treatment_size:].nonzero()
                t_n.append( 0 if treatment.shape[0]==0 else 1)
                pred_t_n.append(0 if treatment_pred.shape[0]==0 else 1)
                if treatment.shape[0]==0 and treatment_pred.shape[0]==0:
                    TN=TN+1
                if treatment.shape[0]!=0 and treatment_pred.shape[0]!=0:
                    TP=TP+1  
                if treatment.shape[0]==0 and treatment_pred.shape[0]!=0:
                    FP=FP+1   
                if treatment.shape[0]!=0 and treatment_pred.shape[0]==0:
                    FN=FN+1  
            
            valid_mse_r += eval_mse_r
            valid_mse_t += eval_mse_t
            valid_mae_r += eval_mae_r
            valid_mae_t += eval_mae_t  
        valid_mse_r /= num_batch
        valid_mse_t /= num_batch
        valid_mae_r /= num_batch
        valid_mae_t /= num_batch
        geometric_mean = np.exp(np.log([valid_mse_r, valid_mse_t, valid_mae_r, valid_mae_t]).mean())
        log_info = "geometric_mean {} valid_mse_response {} valid_mse_treatment {} valid_mae_response {} valid_mae_treatment {}". \
            format( round(geometric_mean, 5), round(valid_mse_r, 5), round(valid_mse_t, 5),
                    round(valid_mae_r, 5), round(valid_mae_t, 5))
    logger.info(log_info)
    logger.info("\n")
    logger.info("best_geometric_mean {}\n".format(round(geometric_mean, 5)))
    Recall=TP/(TP+FN)
    Precision=TP/(TP+FP)
    print("recall:",str(Recall),"     precision: ",str(Precision))
    auc=metrics.roc_auc_score(t_n,pred_t_n)
    print(auc)