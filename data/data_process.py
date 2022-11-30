import torch


def dataset_split(data, mask):  # data == tensor   mask == torch    b t input
    # sample, timestep, feature
    train_data = data[:, :50, 3:53]
    train_mask = mask[:, :50, 3:53]
    valid_data = data[:, 50:60, 3:53]
    valid_mask = mask[:, 50:60, 3:53]
    test_data = data[:, 60:, 3:53]
    test_mask = data[:, 60:, 3:53]

    # TODO
    # 舍弃第36个特征
    train_data = train_data[:, :, torch.arange(train_data.size(2)) != 36]
    train_mask = train_mask[:, :, torch.arange(train_mask.size(2)) != 36]
    valid_data = valid_data[:, :, torch.arange(valid_data.size(2)) != 36]
    valid_mask = valid_mask[:, :, torch.arange(valid_mask.size(2)) != 36]
    test_data = test_data[:, :, torch.arange(test_data.size(2)) != 36]
    test_mask = test_mask[:, :, torch.arange(test_mask.size(2)) != 36]

    return train_data, train_mask, valid_data, valid_mask, test_data, test_mask


def dataset_split_h(data, mask):  # data == tensor   mask == torch    b t input
    # sample, timestep, feature
    train_data = data[2682:, :, 3:53]
    train_mask = mask[2682:, :, 3:53]
    valid_data = data[1341:2682, :, 3:53]
    valid_mask = mask[1341:2682, :, 3:53]
    test_data = data[:1341, :, 3:53]
    test_mask = mask[:1341, :, 3:53]

    # TODO
    # 舍弃第36个特征
    train_data = train_data[:, :, torch.arange(train_data.size(2)) != 36]
    train_mask = train_mask[:, :, torch.arange(train_mask.size(2)) != 36]
    valid_data = valid_data[:, :, torch.arange(valid_data.size(2)) != 36]
    valid_mask = valid_mask[:, :, torch.arange(valid_mask.size(2)) != 36]
    test_data = test_data[:, :, torch.arange(test_data.size(2)) != 36]
    test_mask = test_mask[:, :, torch.arange(test_mask.size(2)) != 36]

    return train_data, train_mask, valid_data, valid_mask, test_data, test_mask