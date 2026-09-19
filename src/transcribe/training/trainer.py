'''
Training loop with CTC/seq2seq support, validation, and checkpointing.
'''

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    '''
    Handles model training, evaluation, and checkpoint saving.
    Supports both CTC and seq2seq training via config['model_type'].
    '''

    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader=None,
        device='cpu',
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )

        # Loss function (CTC)
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

        # History
        self.train_losses = []
        self.val_losses = []

    def _criterion(self) -> nn.Module:
        '''Cross-entropy for seq2seq training.'''
        return nn.CrossEntropyLoss(ignore_index=0)

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            audios, transcripts, audio_lens, trans_lens = batch
            audios = audios.to(self.device)
            transcripts = transcripts.to(self.device)
            audio_lens = audio_lens.to(self.device)
            trans_lens = trans_lens.to(self.device)

            self.optimizer.zero_grad()
            loss = self._compute_loss(
                audios, transcripts, audio_lens, trans_lens
            )
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(
                    f'Epoch {epoch+1} Step {batch_idx}/{len(self.train_loader)} '
                    f'Loss: {loss.item():.4f}'
                )

        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss

    def _compute_loss(self, audios, transcripts, audio_lens, trans_lens):
        '''Compute loss depending on model type.'''
        if self.config.model_type == 'ctc':
            log_probs, output_lens = self.model(audios, audio_lens)
            # CTC needs transposed log_probs: (seq_len, batch, vocab)
            log_probs_t = log_probs.transpose(0, 1)
            loss = self.ctc_loss(
                log_probs_t,
                transcripts,
                output_lens.to(self.device),
                trans_lens.to(self.device),
            )
            return loss
        else:  # seq2seq
            # Teacher forcing: shift target right, use inputs[:-1] -> outputs
            outputs = self.model(audios, transcripts, audio_lens)
            # outputs: (batch, tgt_len-1, vocab)
            flat_outputs = outputs.reshape(-1, outputs.size(-1))
            flat_targets = transcripts[:, 1:].reshape(-1)

            # Build mask for padding (ignore index 0)
            mask = flat_targets != 0
            loss = self._criterion()(flat_outputs[mask], flat_targets[mask])
            return loss

    def evaluate(self, loader=None) -> dict:
        '''
        Run evaluation on training/val/test data.
        Returns average loss and metrics.
        '''
        self.model.eval()
        loader = loader or self.val_loader
        if loader is None:
            return {'loss': 0.0, 'num_samples': 0}

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                audios, transcripts, audio_lens, trans_lens = batch
                audios = audios.to(self.device)
                transcripts = transcripts.to(self.device)
                audio_lens = audio_lens.to(self.device)
                trans_lens = trans_lens.to(self.device)

                loss = self._compute_loss(
                    audios, transcripts, audio_lens, trans_lens
                )
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss, 'num_samples': len(loader.dataset)}

    def train(self):
        '''
        Full training loop with validation and checkpointing.
        '''
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            train_loss = self.train_one_epoch(epoch)

            # Validation
            val_info = {'loss': 0.0}
            if self.val_loader is not None and (
                self.config.eval_every > 0
                and (epoch + 1) % self.config.eval_every == 0
            ):
                val_info = self.evaluate()
                val_loss = val_info['loss']
                self.val_losses.append(val_loss)
                print(
                    f'Epoch {epoch+1} - Train Loss: {train_loss:.4f} '
                    f'Val Loss: {val_loss:.4f}'
                )
                self.scheduler.step(val_loss)
            else:
                print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}')

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_info['loss'],
                    path=str(checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'),
                )

        # Save final model
        self.save_checkpoint(
            epoch=self.config.epochs - 1,
            val_loss=val_info['loss'],
            path=str(checkpoint_dir / 'final_model.pt'),
        )
        print('Training completed!')

    def save_checkpoint(self, epoch, val_loss, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else 0.0,
            'val_loss': val_loss,
            'model_type': self.config.model_type,
            'config': self.config.__dict__,
        }, path)
        print(f'Saved checkpoint to {path}')
