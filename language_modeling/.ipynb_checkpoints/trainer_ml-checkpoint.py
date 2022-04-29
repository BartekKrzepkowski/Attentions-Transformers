import datetime
import gc
import os
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from tensorboard_pytorch import TensorboardPyTorch
from torch.nn.utils.rnn import pack_padded_sequence
from utils import set_seeds, save_model, load_model

class LMTrainer:
    def __init__(self, model, loaders, criterion, optim, scheduler,
                 val_step, is_tensorboard, verbose, device, random_seeds=42):
        self.model = model.to(device)
        self.loaders = loaders
        self.criterion = criterion.to(device)
        self.optim = optim
        self.scheduler = scheduler

        self.val_step = val_step
        self.checkpoint_save_step = 100
        self.verbose = verbose
        self.is_tensorboard = is_tensorboard
        self.device = device
        set_seeds(random_seeds)

    def loop(self, epochs_start, epochs_end, exp_name, checkpoint_save_step=25, load_path=None):
        if load_path:
            load_model(self.model, load_path)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_path = os.path.join(os.getcwd(), f'data/{exp_name}/{date}')
        os.makedirs(f'{self.base_path}/checkpoints')
        if self.is_tensorboard:
            self.writer = TensorboardPyTorch(f'{self.base_path}/tensorboard/{date}', self.device)
            # logging graph structure
            inp = next(iter(self.loaders['train']))[0]
            lengths = torch.LongTensor([1 for _ in range(inp.shape[1])])
            # self.writer.log_graph(self.model, (inp.to(self.device), lengths))
        for epoch in tqdm(range(epochs_start, epochs_end)):
            self.logs = {}

            self.model.train()
            self.run_epoch('train', epoch)

            # self.model.eval()
            # if (epoch + 1) % self.val_step == 0:
            #     with torch.no_grad():
            #         self.run_eval(epoch)

            if (epoch + 1) % checkpoint_save_step == 0:
                self.save_net(f'{self.base_path}/checkpoints/_epoch_{epoch}.pth')

            if self.scheduler:
                self.scheduler.step()
        if self.is_tensorboard:
            self.writer.close()
        gc.collect()

    def run_epoch(self, phase, epoch):
        """Run whole epoch."""
        running_loss = 0.0
        for x_true, y_true, lengths in self.loaders[phase]:
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            y_pred = self.model(x_true, lengths)
            loss, eps = self.criterion(y_pred, y_true)
            if phase == 'train':
                self.optim.zero_grad()
                loss.backward()
                # potrzebne?
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optim.step()

            running_loss += loss.item() * x_true.shape[1]
            # self.writer.update_tensors('y_pred', y_pred)
            # self.writer.update_tensors('y_true', y_true)
            # eps = pack_padded_sequence(input=eps, lengths=lengths, batch_first=False)
            # self.writer.update_tensors('eps', eps.data)

        epoch_loss = running_loss / len(self.loaders[phase].dataset)

        self.logs[f'{phase}_log_loss'] = round(epoch_loss, 4)
        self.writer.log_scalar(f'Loss/{phase}', self.logs[f'{phase}_log_loss'], epoch + 1)
        # self.writer.log_at_epoch_end_val()

        self.writer.log_weight_histogram(self.model, epoch)

        if self.verbose:
            print(self.logs)

    def save_net(self, path):
        """ Saves policy_net parameters as given checkpoint.
        state_dict of current policy_net is stored.
        Args:
            path: path were to store model's parameters.
        """
        save_model(self.model, path)

    def run_eval(self, epoch):
        counter = 0
        i2w = self.loaders['val'].dataset.dataset.Ind2word
        print(f'VALIDATION OF EPOCH {epoch}')
        for x_true, y_true, lengths in self.loaders['val']:
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            x_true_partial = x_true[:3 * x_true.shape[0] // 4]
            idxs = self.model.evaluation(x_true_partial)
            original_sent = [i2w[idx] for idx in x_true.squeeze().cpu().numpy()]
            start_sent = [i2w[idx] for idx in x_true_partial.squeeze().cpu().numpy()]
            words = [i2w[idx] for idx in idxs]
            print('Wyjściowe zdanie to  :: ', ' '.join(original_sent))
            print('Zaczątek zdania to   :: ', ' '.join(start_sent))
            print('Kontynuacja zdania to:: ', ' '.join(words))
            print()
            counter += 1
            if counter > 4:
                break


# if __name__ == "__main__":
#     import torch
#
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     random_seed = 42
#
#     from data import LM_Dataset
#     from utils import loaders_from_dataset, visualize_embeddings, save_model
#
#     dataset = LM_Dataset('datasets/open_subtitles_pl_georgia/datasetSentences.csv')
#     loaders = loaders_from_dataset(dataset, batch_size=32, val_perc_size=0.05)
#
#     from models import LM_Transformer, LMLoss
#
#     epochs = 500
#     model = LM_Transformer(vocab_size=len(dataset.Ind2word), emb_dim=50, rnn=torch.nn.GRU)
#     optim = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
#     params_trainer = {
#         'model': model,
#         'loaders': loaders,
#         'criterion': LMLoss(),
#         'optim': optim,
#         'val_step': 5,
#         'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs),
#         'verbose': True,
#         'device': device,
#         #     'params_clearml': params_clearml,
#         'is_tensorboard': True
#     }
#
#     trainer = LMTrainer(**params_trainer)
#     trainer.loop(0, epochs, exp_name='Attention')




