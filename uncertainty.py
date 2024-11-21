import torch
import argparse
from tqdm import tqdm
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.pool import global_mean_pool
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import numpy as np
import pandas as pd

# Assuming graphcpp.dataset is a module you have with CPPDataset defined
from graphcpp.dataset import CPPDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        # Pre-message-passing layer
        self.pre_mp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.PReLU(hidden_channels),
            torch.nn.Dropout(0.5)
        )
        
        # Message-passing layers
        self.mp = torch.nn.Sequential(
            SAGEConv(hidden_channels, hidden_channels, aggr='sum'),
            torch.nn.PReLU(hidden_channels),
            SAGEConv(hidden_channels, hidden_channels, aggr='sum'),
            torch.nn.PReLU(hidden_channels),
            torch.nn.Dropout(0.5)
        )

        # Fingerprint layer
        self.fingerprint = torch.nn.Sequential(
            torch.nn.Linear(2048, hidden_channels),
            torch.nn.PReLU(hidden_channels),
            torch.nn.Dropout(0.5)
        )
        
        # Post-message-passing layer
        self.post_mp = torch.nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x, fp, edge_index, batch):
        # Pre-message-passing
        x = self.pre_mp(x)
        
        # Message-passing
        for layer in self.mp:
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        # global mean pooling
        x = global_mean_pool(x, batch)

        
        # Fingerprint layer
        fp_emb = self.fingerprint(fp)
        x = torch.cat([x, fp_emb], dim=1)
        
        # Post-message-passing
        x = self.post_mp(x)
        
        return x

# Training function
def train(model, dataloader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    times = []
    for data in tqdm(dataloader, desc='Training'):
        start_time = perf_counter_ns()
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.fp, data.edge_index, data.batch)
        loss = criterion(out, data.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        end_time = perf_counter_ns()
        times.append(end_time - start_time)
    scheduler.step()
    return total_loss / len(dataloader), times

# Evaluation function with Monte Carlo Dropout
def test(model, dataloader, n_mc=50):
    model.train()  # Enable dropout during testing
    all_labels = []
    all_mc_preds = []
    all_names = []
    times = []
    for data in tqdm(dataloader, desc='Testing'):
        start_time = perf_counter_ns()
        data = data.to(device)
        mc_preds = []
        for _ in range(n_mc):
            out = model(data.x, data.fp, data.edge_index, data.batch)
            mc_preds.append(out.softmax(dim=-1)[:, 1].detach().cpu().numpy())
        mc_preds = np.array(mc_preds)  # Shape: (n_mc, batch_size)
        mc_preds = np.transpose(mc_preds, (1, 0))  # Shape: (batch_size, n_mc)

        labels = data.y.long().cpu().numpy()
        all_labels.extend(labels)
        all_mc_preds.extend(mc_preds)
        all_names.extend(data.name)  # Assuming data.name exists and contains the names
        end_time = perf_counter_ns()
        times.append(end_time - start_time)
    return np.array(all_labels), np.array(all_mc_preds), all_names, times

if __name__ == '__main__':
    # get epochs, batch size, and fold number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--folder', type=str, default='2024-07-24-16-24-56/70')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_folds', type=int, default=10)
    args = parser.parse_args()

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Example usage
    folder = args.folder
    fp_type = 'topological'
    batch_size = args.batch_size
    epochs = args.epochs
    n_folds = args.n_folds

    print(f"Folder: {folder}")
    print(f"Seed: {args.seed}")
    print(f"Training for {epochs} epochs with batch size {batch_size} and {n_folds}-fold cross-validation")

    # Combine all data splits into one dataset
    train_split = CPPDataset(root=folder, _split='train', fp_type=fp_type).shuffle()
    val_split = CPPDataset(root=folder, _split='val', fp_type=fp_type).shuffle()
    test_split = CPPDataset(root=folder, _split='test', fp_type=fp_type).shuffle()
    dataset = train_split + val_split + test_split

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    
    in_channels = 32
    hidden_channels = 128
    out_channels = 2

    all_test_acc = []
    all_test_roc_auc = []
    all_test_f1 = []
    all_test_mcc = []

    results = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size)

        model = GraphSAGE(in_channels, hidden_channels, out_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = torch.nn.CrossEntropyLoss()

        # print the number of parameters with thousands separators
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Using device: {device}")

        from time import perf_counter_ns

        # take the model with the best validation accuracy
        best_model_state_dict = None
        best_acc = 0

        # Training loop
        for epoch in range(1, epochs+1):
            start_time = perf_counter_ns()
            loss, times = train(model, train_loader, optimizer, criterion, scheduler)
            labels, predictions_mc, names, timestest = test(model, test_loader)
            
            # Get mean predictions for validation accuracy
            mean_preds = predictions_mc.mean(axis=1)
            predictions = (mean_preds > 0.5).astype(int)

            acc = accuracy_score(labels, predictions)
            roc_auc = roc_auc_score(labels, mean_preds)
            mcc = matthews_corrcoef(labels, predictions)
            f1 = f1_score(labels, predictions)

            # current learning rate
            lr = optimizer.param_groups[0]['lr']
            print(f'Fold: {fold + 1}, Epoch: {epoch:03d}, Loss: {loss:.4f}, LR: {lr:.4f}, Time: {(perf_counter_ns() - start_time) / 1e9:.2f}s')
            print(f'Validation Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}')

            # Mean std and CI 95% of time per entry
            avg_time_per_entry = np.mean(times)
            std_time_per_entry = np.std(times)
            lower = np.percentile(times, 2.5)
            upper = np.percentile(times, 97.5)
            print(f"Average time per entry: {avg_time_per_entry:.4f} +- {std_time_per_entry:.4f} seconds")
            print(f"95% CI: [{lower:.4f}, {upper:.4f}] seconds")

            # mean std and CI 95% of time per entry timestest
            avg_time_per_entry_test = np.mean(timestest)
            std_time_per_entry_test = np.std(timestest)
            lower_test = np.percentile(timestest, 2.5)
            upper_test = np.percentile(timestest, 97.5)
            print(f"Test Average time per entry: {avg_time_per_entry_test:.4f} +- {std_time_per_entry_test:.4f} seconds")
            print(f"Test 95% CI: [{lower_test:.4f}, {upper_test:.4f}] seconds")


            if best_model_state_dict is None or acc > best_acc:
                best_model_state_dict = model.state_dict()
                best_acc = acc

        print("="*10)

        # Load the best model
        model.load_state_dict(best_model_state_dict)

        # Test the best model
        test_labels, test_mc_preds, test_names, _ = test(model, test_loader)
        mean_test_preds = test_mc_preds.mean(axis=1)

        upper_bound = np.percentile(test_mc_preds, 97.5, axis=1)
        lower_bound = np.percentile(test_mc_preds, 2.5, axis=1)
        final_predictions = (mean_test_preds > 0.5).astype(int)
        final_acc = accuracy_score(test_labels, final_predictions)
        final_roc_auc = roc_auc_score(test_labels, mean_test_preds)
        final_mcc = matthews_corrcoef(test_labels, final_predictions)
        final_f1 = f1_score(test_labels, final_predictions)
        print(f'Final Test Accuracy: {final_acc:.4f}, ROC AUC: {final_roc_auc:.4f}, F1: {final_f1:.4f}, MCC: {final_mcc:.4f}')

        all_test_acc.append(final_acc)
        all_test_roc_auc.append(final_roc_auc)
        all_test_f1.append(final_f1)
        all_test_mcc.append(final_mcc)

        # Save results
        for name, mean_prob, up_bound, l_bound in zip(test_names, mean_test_preds, upper_bound, lower_bound):
            results.append([name, mean_prob, up_bound, l_bound])

    # Save results to CSV
    df_results = pd.DataFrame(results, columns=['name', 'mean_probability', 'up_bound', 'low_bound'])
    df_results.to_csv('results.csv', index=False)

    # Print average performance over all folds
    print(f'Average Test Accuracy: {np.mean(all_test_acc):.4f}, ROC AUC: {np.mean(all_test_roc_auc):.4f}, F1: {np.mean(all_test_f1):.4f}, MCC: {np.mean(all_test_mcc):.4f}')
