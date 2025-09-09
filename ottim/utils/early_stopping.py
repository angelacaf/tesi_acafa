import numpy as np
import torch

class EarlyStopping:
    """
    Early stopping per fermare il training quando le metriche non migliorano
    più dopo un certo numero di epoche.
    """
    def __init__(self, patience=2, min_delta=0.01, verbose=True):
        """
        Args:
            patience: Numero di epoche da attendere prima di fermare il training
            min_delta: Cambiamento minimo che viene considerato un miglioramento
            verbose: Se True, stampa messaggi informativi
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_weights = None
        
    def __call__(self, val_score, epoch, logger):
        """
        Verifica se il training deve essere fermato
        
        Args:
            val_score: Punteggio di validazione corrente
            epoch: Epoca corrente
            logger: Logger per i messaggi informativi
        
        Returns:
            bool: True se il modello è migliorato, False altrimenti
        """
        if self.best_score is None:
            self.best_score = val_score
            return True
            
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            if self.verbose:
                logger.info(f"Miglioramento all'epoca {epoch}! Nuovo miglior score: {val_score:.4f}")
            return True
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping: {self.counter}/{self.patience} epoche senza miglioramento (best: {self.best_score:.4f}, current: {val_score:.4f})')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"EarlyStopping: Interruzione training all'epoca {epoch}")
            return False
