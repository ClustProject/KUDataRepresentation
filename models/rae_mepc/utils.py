import pickle
import pandas as pd
from metric import *
import os


def load_data(path):
    with open(path, 'rb') as f:
        data = torch.FloatTensor(pickle.load(f))
        label = data[:, -1]
        data = data[:, :-1]
    return data, label


def calculate_params(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        errors = []
        for i, data in enumerate(dataloader):
            data_x, pred_data_x, _, _ = data
            data_x = data_x.to(device)
            pred_data_x = pred_data_x.to(device)

            _, _, error, _ = model(data_x, pred_data_x, mode="test")
            errors.append(error)
            
        errors = torch.cat(errors, dim=0).view(-1, errors[0].size(-1))  #[seq_len, input_size]

        mean = errors.mean(0)  #[input_size]
        cov = (errors.t()).mm(errors)/errors.size(0) - mean.unsqueeze(1).mm(mean.unsqueeze(1).t())  #[input_size, input_size]
        return mean, cov


def save_mean_cov(mean, cov, mean_cov_save_path):
    mean_cov_dict = {}
    mean_cov_dict['mean'] = mean
    mean_cov_dict['cov'] = cov
    with open(mean_cov_save_path, 'wb') as fw:
        pickle.dump(mean_cov_dict, fw)


def load_mean_cov(mean_cov_save_path):
    with open(mean_cov_save_path, 'rb') as fr:
        mean_cov_dict = pickle.load(fr)

    mean = mean_cov_dict['mean']
    cov = mean_cov_dict['cov']
    return mean, cov


def save_model_weights(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model_weights(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model


def test(args, model, dataloader, mean_cov_save_path):
    print("\nTesting the Model!!!")
    outputs, score, inputs, labels, errors, all_indexes = get_anomaly_score(args, model, dataloader, mean_cov_save_path)

    precision, recall, f_1 = get_precison_recall(score.cpu(), labels.cpu(), 10000, beta=1.0)
    print("F1_score: ", f_1.cpu().data.numpy())

    _, _, roc_auc = CalculateROCAUCMetrics(score, labels)
    _, _, pr_auc = CalculatePrecisionRecallCurve(score, labels)
    print("ROC-AUC:{}, PR-AUC:{}".format(roc_auc, pr_auc))

    inputs, outputs, score, labels, all_indexes = inputs.cpu().data.numpy(), outputs.cpu().data.numpy(), score.cpu().data.numpy(), labels.cpu().data.numpy(), all_indexes.cpu().data.numpy()

    result = pd.DataFrame()
    result['score'] = score
    result['label'] = labels
    result_save_path = 'ours' + mean_cov_save_path.split('/')[-1].split('mean_cov')[-1].split('.pkl')[0] + '.csv'
    result.to_csv(os.path.join('/home/heejeong/Desktop/personal_research/anomaly_detection/proposed/best_result', result_save_path), index=False)

    _, inputs = merge_repeat_records(inputs, all_indexes, mode="mean")
    _, outputs = merge_repeat_records(outputs, all_indexes, mode="mean")
    _, labels = merge_repeat_records(labels, all_indexes, mode="mean")
    return f_1.cpu().data.numpy(), roc_auc, pr_auc


def get_anomaly_score(args, model, dataloader, mean_cov_save_path):
    mean, covariance = load_mean_cov(mean_cov_save_path)

    outputs = []
    labels = []
    datas = []
    errors = []
    all_indexes = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data_x, pred_data_x, data_y, indexes = data
            data_x = data_x.to(args.device)
            pred_data_x = pred_data_x.to(args.device)
            data_y = data_y.to(args.device)
            indexes = indexes.to(args.device)

            _, output, error, _ = model(data_x, pred_data_x, mode="test")

            output_idx = torch.arange(output.size(1)-1, -1, -1).to(args.device).long()
            reverse_output = output.index_select(1, output_idx)
            reverse_error = error.index_select(1, output_idx)

            datas.append(data_x.view(-1, args.ninp))
            labels.append(data_y.view(-1, 1))
            outputs.append(reverse_output.view(-1, args.ninp))
            errors.append(reverse_error.view(-1, args.ninp))

            all_indexes.append(indexes.view(-1, 1))
    
    outputs = torch.cat(outputs, dim=0)
    errors = torch.cat(errors, dim=0)
    labels = torch.cat(labels, dim=0)
    labels = labels.squeeze(dim=1)
    datas = torch.cat(datas, dim=0)
    all_indexes = torch.cat(all_indexes, dim=0)

    xm = (errors - mean)
    cov_eps = covariance + 1e-5 * torch.eye(covariance.size(0)).to(args.device)
    score = xm.mm(cov_eps.inverse()) * xm
    score = score.sum(dim=1)
    return outputs, score, datas, labels, errors, all_indexes


def merge_repeat_records(scores, index, mode="min"):
    n = len(index)
    try:
        k = np.shape(scores)[1]
    except:
        k = 2

    res = np.zeros((n, k + 1))
    for i in range(n):
        res[i, 0] = index[i]
        res[i, 1:] = scores[i]

    res = pd.DataFrame(res)

    cols = {}
    for i in range(0, k + 1):
        cols.update({i: mode})
    res = res.groupby(0).agg(cols)
    res = np.array(res)

    index = res[:, 0]
    scores = res[:, 1:]
    return index, scores
