import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold

from models.ccgru import DualGRU
from models.attention import SelfAttention
from utils import print_metrics, print_avg_metrics, FocalLoss, calculate_alpha

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('embeddings/instagram/embeddings_LaBSE.pickle', 'rb') as f:
# with open('embeddings/vine/embeddings_LaBSE.pickle', 'rb') as f:
    data = pickle.load(f)
    post_emb = data['post_emb']
    comment_emb = data['comment_emb']
    labels = data['labels']
    lengths = data['lengths']

post_emb = post_emb.to(device)
comment_emb = comment_emb.to(device)
labels = labels.to(device)

misclassified_samples = {i: 0 for i in range(len(post_emb))}
misclassified_lengths = {i: 0 for i in range(len(post_emb))}

total_time = 0
train_epochs = 30
fold_num = 5
metrics = {fold: {epoch: {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
                  for epoch in range(train_epochs)}
           for fold in range(fold_num)}
kf = KFold(n_splits=fold_num, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(post_emb.cpu().numpy())):
    print(f"-----Fold {fold + 1}-----")
    all_index = torch.tensor(test_index).to(device)
    train_post_emb, test_post_emb = post_emb[train_index], post_emb[test_index]
    train_comment_emb, test_comment_emb = comment_emb[train_index], comment_emb[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
    train_lengths, test_lengths = lengths[train_index], lengths[test_index]

    # 计算当前fold的alpha值（可选）
    alpha = calculate_alpha(train_labels)
    print(f"Alpha (positive class weight) for fold {fold + 1}: {alpha:.4f}")

    # 初始化Focal Loss，可以选择是否使用alpha
    criterion = FocalLoss(gamma=2.0)  # 不使用alpha
    # criterion = FocalLoss(gamma=2.0, alpha=alpha)  # 使用alpha

    train_dataset = TensorDataset(train_post_emb, train_comment_emb, train_labels, train_lengths)
    test_dataset = TensorDataset(test_post_emb, test_comment_emb, test_labels, test_lengths, all_index)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    attention = SelfAttention(8, 96, 768).to(device)
    classifier = DualGRU(input_dim=768, hidden_dim=768).to(device)
    optimizer = torch.optim.Adam(list(classifier.parameters()) + list(attention.parameters()), lr=0.0001)

    avg_acc, avg_prec, avg_rec, avg_f1, avg_auc = [], [], [], [], []
    best_auc = 0
    best_metrics = []
    for epoch in range(train_epochs):
        start_time = time.time()
        classifier.train()
        train_predictions = []
        train_true_labels = []
        for batch in train_loader:
            input_post, input_comments, input_labels, input_lengths = batch
            optimizer.zero_grad()
            input = attention(torch.concat((input_post, input_comments), dim=1))
            input_post = input[:, 0, :].unsqueeze(1)
            input_comments = input[:, 1:, :]
            output = classifier(input_post, input_comments, input_lengths)
            loss = criterion(output, input_labels.float())
            loss.backward()
            optimizer.step()

            train_predictions.append((output > 0.5).cpu().numpy())
            train_true_labels.append(input_labels.cpu().numpy())

        end_time = time.time()
        total_time += end_time - start_time
        train_predictions = np.concatenate(train_predictions)
        train_true_labels = np.concatenate(train_true_labels)
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.7f}")
        print("Training:".ljust(10), end=' ')
        print_metrics(train_predictions, train_true_labels)

        classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            predictions = []
            true_labels = []
            for batch in test_loader:
                input_post, input_comments, input_labels, input_lengths, all_index = batch
                input = attention(torch.concat((input_post, input_comments), dim=1))
                input_post = input[:, 0, :].unsqueeze(1)
                input_comments = input[:, 1:, :]
                output = classifier(input_post, input_comments, input_lengths)
                predictions.append((output > 0.5).cpu().numpy())
                true_labels.append(input_labels.cpu().numpy())

                for i, (pred, true) in enumerate(zip((output > 0.5).cpu().numpy(), input_labels.cpu().numpy())):
                    if pred != true:
                        misclassified_samples[all_index[i].item()] += 1
                        misclassified_lengths[all_index[i].item()] = input_lengths[i].item()

            predictions = np.concatenate(predictions)
            true_labels = np.concatenate(true_labels)

            print("Testing:".ljust(10), end=' ')
            acc, prec, rec, f1, auc = print_metrics(predictions, true_labels)

            metrics[fold][epoch]['acc'] = acc
            metrics[fold][epoch]['prec'] = prec
            metrics[fold][epoch]['rec'] = rec
            metrics[fold][epoch]['f1'] = f1
            metrics[fold][epoch]['auc'] = auc

print_avg_metrics(train_epochs, metrics, fold_num)

print("Total training time:", total_time)
print("Training time per epoch:", total_time / (train_epochs * fold_num))