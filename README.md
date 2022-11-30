# AL-Transformer: Forecasting Treatment and Response Over Time Using Alternating Sequential Models

This is the repository for the paper AL-Transformer: Forecasting Treatment and Response Over Time Using Alternating Sequential Models. 

The code is early stage version.

## Get Started

1. Create file folder (datasets) and (trained_models/transformer). 
2. Put the normalized_dataset_new.pt files into /datasets
3. Train the model

```    
    python run_h.py 
```

If you want to run the Monte Carlo simulation in AL-Transformer model, just comment 100-101 lines and uncomment 102-103 lines in run_h.py.

```
    # train_stacked_lstm_cnn(args.epoch, train_iterator,vaild_iterator,test_iterator, model, args.mask, window=12, logger=logger, -->
                           model_save_path=os.path.join(model_path, "model.pt"), response_size=47, treatment_size=2, device=device)

    simulation( test_iterator,vaild_iterator, model, args.mask, window=12, logger=logger,
                            model_save_path=os.path.join(model_path, "model.pt"), response_size=47, treatment_size=2, device=device)
```

And using your trained model parameter in tools/simulation.py line 256

```
    model.load_state_dict(torch.load("trained_models/transformer/2022_11_02_14_23_52/model.pt",map_location='cuda:0'))  # change the file in this line.
    model.load_state_dict(torch.load("trained_models/transformer/2022_11_15_01_30_42/model.pt",map_location='cuda:0'))  # 12hours model 
```

If you want to run the baseline model(Informer and Transformer), you can comment line 90 and uncomment 92-93 in run_h.py according to your target model. Then comment the function train_stacked_lstm_cnn in 100-101 line and uncomment function train_informer in lines 108-109

```
    # model = StackedTransformer(response_size=47, treatment_size=2)
    
    model=Informer( enc_in=49, dec_in=49, c_out=49)
    # model=Transformer()

    #    train_stacked_lstm_cnn(args.epoch, train_iterator,vaild_iterator,test_iterator, model, args.mask, window=12, logger=logger,
                           model_save_path=os.path.join(model_path, "model.pt"), response_size=47, treatment_size=2, device=device)


    train_informer(args.epoch, train_iterator,vaild_iterator,test_iterator, model, args.mask, window=12, logger=logger,
                           model_save_path=os.path.join(model_path, "Informer_model.pt"), response_size=47, treatment_size=2, device=device)
      
```

RNN, LSTM, GRU model in the archive/rnn.py, just change the variable module in  1960 line to your target model and run the code.


```
    python archive/rnn.py
```
LSTM-RT model in the archive/new_lstm_rt.py, just run code.

```
    python archive/new_lstm_rt.py
```

the Monte Carlo simulation of RNN, LSTM, GRU and LSTM-RT in simulation_rnn.ipynb, before you run the MC simulation, you should run the train once and get the model.