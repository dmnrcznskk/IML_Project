import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, Any, Optional, List, Union
import os
from backend.architectures.base_model import BaseModel

class ModelEvaluator:
    """
    Uniwersalna klasa do oceny modeli (zarówno własnych DNN/CNN, jak i Scikit-learn).
    """

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        Args:
            class_names: Słownik mapujący ID klasy na nazwę (np. {0: '20km/h', ...}).
        """
        self.class_names = class_names
        self.target_names = None
        
        if self.class_names:
            max_idx = max(self.class_names.keys())
            # Tworzymy pełną listę nazw indeksowaną od 0 do 42
            self.target_names = [self.class_names.get(i, f"Class {i}") for i in range(max_idx + 1)]

    def _predict_batch(self, model, X: np.ndarray) -> np.ndarray:
        """
        Wewnętrzna metoda obsługująca różnice między Twoim BaseModelem a Scikit-learn.
        """
        if isinstance(model, BaseModel):
            preds = []
            for i in range(len(X)):
                p = model.predict(X[i])
                if isinstance(p, (np.ndarray, list)):
                    preds.append(p[0] if len(p) > 0 else 0)
                else:
                    preds.append(p)
            return np.array(preds)

        else:
            return model.predict(X)

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray, show_plot: bool = True) -> Dict[str, Any]:
        """
        Główna metoda uruchamiająca ewaluację.
        """
        y_pred = self._predict_batch(model, X_test)

        acc = accuracy_score(y_test, y_pred)
        
        all_labels = None
        if self.target_names:
            all_labels = list(range(len(self.target_names)))

        # zero_division=0 sprawia, że dla brakujących klas dostaniesz 0 zamiast błędu/warningu
        report_dict = classification_report(
            y_test, 
            y_pred, 
            target_names=self.target_names, 
            labels=all_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Wypisujemy skrócony raport na konsolę (opcjonalnie można wypisać pełny)
        print(f"\nDokładność (Accuracy): {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=self.target_names, labels=all_labels, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred, labels=all_labels) # Tu też warto dodać labels, żeby macierz miała stały wymiar 43x43
        
        if show_plot:
            self.plot_confusion_matrix(cm)

        return {
            "accuracy": acc,
            "confusion_matrix": cm,
            "report": report_dict
        }

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """
        Rysuje i opcjonalnie zapisuje macierz pomyłek.
        """
        # Filtrujemy puste wiersze/kolumny dla czytelności wykresu? 
        # Nie, zostawmy pełny wymiar, żeby było widać skalę problemu.
        
        plt.figure(figsize=(16, 14)) # Zwiększyłem trochę rozmiar, bo 43 klasy to sporo
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=self.target_names if self.target_names else "auto",
                    yticklabels=self.target_names if self.target_names else "auto")
        plt.xlabel('Przewidziana klasa')
        plt.ylabel('Prawdziwa klasa')
        plt.title('Macierz Pomyłek (Confusion Matrix)')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()