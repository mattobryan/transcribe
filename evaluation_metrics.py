import numpy as np
from collections import Counter
import editdistance
import torch

class EvaluationMetrics:
    def __init__(self, char_to_idx, idx_to_char):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
    
    def indices_to_text(self, indices):
        """Convert indices back to text"""
        text = ""
        for idx in indices:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
                    text += char
        return text
    
    def calculate_wer(self, reference, hypothesis):
        """Calculate Word Error Rate"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        
        # Calculate edit distance
        distance = editdistance.eval(ref_words, hyp_words)
        wer = distance / len(ref_words)
        
        return wer
    
    def calculate_cer(self, reference, hypothesis):
        """Calculate Character Error Rate"""
        if len(reference) == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0
        
        distance = editdistance.eval(reference, hypothesis)
        cer = distance / len(reference)
        
        return cer
    
    def calculate_bleu(self, references, hypotheses):
        """Calculate BLEU score"""
        from nltk.translate.bleu_score import sentence_bleu
        
        # Convert to proper format
        ref_list = [[ref.split()] for ref in references]
        hyp_list = [hyp.split() for hyp in hypotheses]
        
        bleu_scores = []
        for refs, hyp in zip(ref_list, hyp_list):
            try:
                score = sentence_bleu(refs, hyp)
                bleu_scores.append(score)
            except:
                bleu_scores.append(0.0)
        
        return np.mean(bleu_scores) if bleu_scores else 0.0
    
    def evaluate_model(self, model, test_loader, device):
        """Evaluate model on test set"""
        model.eval()
        all_references = []
        all_hypotheses = []
        
        with torch.no_grad():
            for batch in test_loader:
                audios, transcripts, audio_lens, trans_lens = batch
                audios = audios.to(device)
                
                # Get predictions
                outputs = model(audios)
                predictions = torch.argmax(outputs, dim=-1)
                
                # Convert to text
                for i in range(len(predictions)):
                    pred_text = self.indices_to_text(predictions[i].cpu().numpy())
                    ref_text = self.indices_to_text(transcripts[i].numpy())
                    
                    all_references.append(ref_text)
                    all_hypotheses.append(pred_text)
        
        # Calculate metrics
        wer_scores = [self.calculate_wer(ref, hyp) for ref, hyp in zip(all_references, all_hypotheses)]
        cer_scores = [self.calculate_cer(ref, hyp) for ref, hyp in zip(all_references, all_hypotheses)]
        
        results = {
            'wer': np.mean(wer_scores),
            'cer': np.mean(cer_scores),
            'bleu': self.calculate_bleu(all_references, all_hypotheses),
            'num_samples': len(all_references)
        }
        
        return results

if __name__ == "__main__":
    print("EvaluationMetrics class initialized")
