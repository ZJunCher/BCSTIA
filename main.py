from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix,roc_curve
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import SubsetRandomSampler
from Model.model import *
from ReadData import *
import argparse
import random
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')



def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(42)

def accuracy(output, labels):
    _, pred = torch.max(output, dim=1)
    # print(output)
    correct = pred.eq(labels)
    # print(pred)
    # print(labels)
    acc_num = correct.sum()

    return acc_num

def stest(model, datasets_test):
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    all_final = []
    pro_all = []
    adj_all = []
    out_logits_all = []
    model.eval()
    for data_test, dti_data_test, label_test in datasets_test:
        batch_num_test = len(data_test)
        data_test = data_test.reshape(batch_num_test, -1, num_nodes, args.patch_size)
        data_test = data_test.to(DEVICE)
        label_test = label_test.long().to(DEVICE)

        output, block_out, out_logits, adj= model(data_test, dti_data_test)
        adj_all.append(adj.cpu().detach().numpy())
        all_final.append(block_out.cpu().detach().numpy())
        out_logits_all.append(out_logits.cpu().detach().numpy())  # 保存out_logits
        # print(output)
        losss = nn.CrossEntropyLoss()(output, label_test)
        eval_loss += float(losss)
        _, pred = torch.max(output, dim=1)
        num_correct = (pred == label_test).sum()
        acc = int(num_correct) / batch_num_test
        eval_acc += acc
        pre = pred.cpu().detach().numpy()
        pre_all.extend(pre)
        label_true = label_test.cpu().detach().numpy()
        labels_all.extend(label_true)
        pro_all.extend(output[:, 1].cpu().detach().numpy())

    # tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
    # sensitivity = tp / (tp + fn)
    # specificity = tn / (tn + fp)
    # eval_acc_epoch = accuracy_score(labels_all, pre_all)
    # precision = precision_score(labels_all, pre_all)
    # recall = recall_score(labels_all, pre_all)
    # f1 = f1_score(labels_all, pre_all)
    # my_auc = roc_auc_score(labels_all, pro_all)
    cm = confusion_matrix(labels_all, pre_all)
    num_classes = cm.shape[0]

    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = 0
        specificity = 0

    eval_acc_epoch = accuracy_score(labels_all, pre_all)

    precision = precision_score(labels_all, pre_all, average='macro')
    recall = recall_score(labels_all, pre_all, average='macro')
    f1 = f1_score(labels_all, pre_all, average='macro')

    # 计算AUC
    try:
        if num_classes == 2:
            my_auc = roc_auc_score(labels_all, pro_all)
        else:
            my_auc = roc_auc_score(labels_all, pro_all, multi_class='ovr')  # 多分类AUC
    except:
        my_auc = 0.0
    # labels_all = np.array(labels_all)
    # pro_all = np.array(pro_all)
    # mask = ~np.isnan(labels_all) & ~np.isnan(pro_all)
    # labels_all_clean = labels_all[mask]
    # pro_all_clean = pro_all[mask]
    # if labels_all_clean.size == 0 or pro_all_clean.size == 0:
    #     print("Error: After removing NaNs, there are no samples left.")
    #     my_auc = 0
    # else:
    #     my_auc = roc_auc_score(labels_all_clean, pro_all_clean)

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, out_logits_all,all_final,adj_all

def save_roc_data(file_path, all_fpr, all_tpr):
    with open(file_path, 'w') as f:
        for fpr, tpr in zip(all_fpr, all_tpr):
            for fp, tp in zip(fpr, tpr):
                f.write(f"{fp}\t{tp}\n")


# main settings
parser = argparse.ArgumentParser(description='GCN')
parser.add_argument('--patch_size', type=int, default=98, help='time length')
parser.add_argument('--batch_size', type=int, default=20, help='batch size')
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=0.003, type=float)
args = parser.parse_args()

dataset, num_nodes, seq_length, num_classes = ADNI_DTI(args)
# dataset, num_nodes, seq_length, num_classes = dianxiandata_DTI(args)
print(num_nodes, seq_length, num_classes)
args.num_nodes = num_nodes
args.num_classes = num_classes
args.seq_length = seq_length

dataset_length = len(dataset)
train_ratio = 1.0
valid_ratio = 0.0
kk = 10
test_acc = []
test_pre = []
test_recall = []
test_f1 = []
test_auc = []
label_ten = []
test_sens = []
test_spec = []
pro_ten = []
i = 0
skf = StratifiedKFold(n_splits=10, shuffle=True)
KF = KFold(n_splits=10, shuffle=True)
labels = [dataset[i][2] for i in range(dataset_length)]
# for flod, (train_and_val_indices, test_indices) in enumerate(skf.split(range(dataset_length), labels)):
for train_and_val_indices, test_indices in KF.split(dataset):
    print("*******{}-flod*********".format(i+1))

    # Further split training and validation indices
    train_size = int(train_ratio * len(train_and_val_indices))
    valid_size = len(train_and_val_indices) - train_size
    train_indices, val_indices = train_and_val_indices[:train_size], train_and_val_indices[train_size:]

    train_num = len(train_indices)
    val_num = len(val_indices)
    test_num = len(test_indices)
    print(train_num, val_num, test_num)

    #  Extract train, validation, and test datasets
    datasets_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=SubsetRandomSampler(train_indices))
    datasets_valid = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=SubsetRandomSampler(val_indices))
    datasets_test = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=SubsetRandomSampler(test_indices))

    model = Model()
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    closs = nn.CrossEntropyLoss()
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    patience = 0

    patiences = 150
    min_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for data, dti_data, label in datasets_train:
            batch_num_train = len(data)
            data = data.reshape(batch_num_train, -1, num_nodes, args.patch_size)
            data = data.to(DEVICE)
            label = label.long().to(DEVICE)
            output, block_out, _, _ = model(data, dti_data)
            batch_loss_train = closs(output, label)
            optimizer.zero_grad()
            total_loss = batch_loss_train
            total_loss.backward()
            optimizer.step()
            acc_num = accuracy(output, label)
            train_acc += acc_num/batch_num_train
            train_loss += batch_loss_train

        losses.append(train_loss / len(datasets_train))
        acces.append(train_acc / len(datasets_train))

        eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, _,_,_ = stest(
            model, datasets_test)

        if eval_acc_epoch > min_acc:
            min_acc = eval_acc_epoch
            torch.save(model.state_dict(), './Save/Model/1_3/latest' + str(i) + '.pth')
            print("Model saved at epoch{}, Best Acc: {}".format(epoch, eval_acc_epoch))
            patience = 0
        else:
            patience += 1

        if patience > patiences or eval_acc_epoch == 1.0 :
            break

        eval_losses.append(eval_loss / len(datasets_test))
        eval_acces.append(eval_acc / len(datasets_test))

        print(
            'i:{},epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},precision : {'
            ':.6f},recall : {:.6f},f1 : {:.6f},my_auc : {:.6f} '
            .format(i, epoch, train_loss / len(datasets_train), train_acc / len(datasets_train),
                    eval_loss / len(datasets_test), eval_acc_epoch, precision, recall, f1, my_auc))

    model_test = Model()
    model_test = model_test.to(DEVICE)
    model_test.load_state_dict(torch.load('./Save/Model/1_3/latest' + str(i) + '.pth'))
    eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all,_,_,_ = stest(
        model_test, datasets_test)

    print("---", eval_acc_epoch)
    test_acc.append(eval_acc_epoch)
    test_pre.append(precision)
    test_recall.append(recall)
    test_f1.append(f1)
    test_auc.append(my_auc)
    test_sens.append(sensitivity)
    test_spec.append(specificity)
    label_ten.extend(labels_all)
    pro_ten.extend(pro_all)

    i = i+1

print("test_acc", test_acc)
print("test_pre", test_pre)
print("test_recall", test_recall)
print("test_f1", test_f1)
print("test_auc", test_auc)
print("test_sens", test_sens)
print("test_spec", test_spec)
avg_acc = sum(test_acc) / kk
avg_pre = sum(test_pre) / kk
avg_recall = sum(test_recall) / kk
avg_f1 = sum(test_f1) / kk
avg_auc = sum(test_auc) / kk
avg_sens = sum(test_sens) / kk
avg_spec = sum(test_spec) / kk
print("*****************************************************")
print('acc', avg_acc)
print('pre', avg_pre)
print('recall', avg_recall)
print('f1', avg_f1)
print('auc', avg_auc)
print("sensitivity", avg_sens)
print("specificity", avg_spec)

acc_std = np.sqrt(np.var(test_acc))
pre_std = np.sqrt(np.var(test_pre))
recall_std = np.sqrt(np.var(test_recall))
f1_std = np.sqrt(np.var(test_f1))
auc_std = np.sqrt(np.var(test_auc))
sens_std = np.sqrt(np.var(test_sens))
spec_std = np.sqrt(np.var(test_spec))
print("*****************************************************")
print("acc_std", acc_std)
print("pre_std", pre_std)
print("recall_std", recall_std)
print("f1_std", f1_std)
print("auc_std", auc_std)
print("sens_std", sens_std)
print("spec_std", spec_std)
print("*****************************************************")

print(label_ten)
print(pro_ten)

for n in range(10):
    full_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    best_model = Model()
    best_model = best_model.to(DEVICE)
    best_model.load_state_dict(torch.load('./Save/Model/1_3/latest'+str(n)+'.pth'))  # Load the best model saved during cross-validation

    # Run the model on the full dataset
    eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, out_logits_all, all_final,adj_all = stest(
        best_model, full_dataloader)
    # Print the final evaluation metrics
    print("N:", n)
    print("Final Eval Acc:", eval_acc_epoch)
    print("Final Precision:", precision)
    print("Final Recall:", recall)
    print("Final F1:", f1)
    print("Final AUC:", my_auc)
    print("Final Sensitivity:", sensitivity)
    print("Final Specificity:", specificity)
    print("---------------------")

