import re
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import time
import os

class TrainingMonitor:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.epochs = []
        self.losses = []
        self.accuracies = []
        
    def read_updates(self):
        """Lit toutes les données du fichier"""
        self.epochs, self.losses, self.accuracies = [], [], []
        
        if os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Pattern pour: Époque X | Loss: X.XXXXX | Accuracy: XX.XXXX%
                    match = re.search(r'Époque\s+(\d+).*?Loss:\s+([0-9.]+).*?Accuracy:\s+([0-9.]+)%', line)
                    if match:
                        try:
                            epoch = int(match.group(1))
                            loss = float(match.group(2))
                            accuracy = float(match.group(3))
                            self.epochs.append(epoch)
                            self.losses.append(loss)
                            self.accuracies.append(accuracy)
                        except ValueError:
                            continue

    def update_plot(self):
        """Met à jour le graphique sans mode interactif"""
        plt.figure(figsize=(12, 8))
        
        # Subplot pour Loss
        plt.subplot(2, 1, 1)
        plt.plot(self.epochs, self.losses, 'b-', label='Loss', linewidth=2)
        plt.ylabel('Loss')
        plt.title('Évolution de la Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Subplot pour Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(self.epochs, self.accuracies, 'r-', label='Accuracy', linewidth=2)
        plt.xlabel('Époque')
        plt.ylabel('Accuracy (%)')
        plt.title('Évolution de l\'Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()  # Important : fermer la figure
        print("📊 Graphique mis à jour : training_progress.png")

    def monitor(self, update_interval=5):
        """Lance la surveillance du fichier"""
        print(f"🎯 Surveillance de {self.log_file_path}...")
        print("Appuyez sur Ctrl+C pour arrêter")
        
        try:
            while True:
                self.read_updates()
                if self.epochs:  # Si on a des données
                    self.update_plot()
                    print(f"✅ Époque {self.epochs[-1]:3d} | Loss: {self.losses[-1]:7.3f} | Acc: {self.accuracies[-1]:5.1f}%")
                else:
                    print("⏳ En attente de données...")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Surveillance arrêtée")
            # Sauvegarde finale
            if self.epochs:
                self.update_plot()
                print(f"📈 Résumé final: {len(self.epochs)} époques")

def simple_file_monitor(log_file_path, update_interval=5):
    """
    Version simplifiée sans classe
    Génère un nouveau PNG à chaque mise à jour
    """
    print(f"🔍 Monitoring: {log_file_path}")
    print("📊 Un nouveau graphique sera généré toutes les 5 secondes")
    print("⏹️  Ctrl+C pour arrêter")
    
    try:
        while True:
            epochs, losses, accuracies = [], [], []
            
            # Lire tout le fichier
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Pattern pour: Époque X | Loss: X.XXXXX | Accuracy: XX.XXXX%
                        match = re.search(r'Époque\s+(\d+).*?Loss:\s+([0-9.]+).*?Accuracy:\s+([0-9.]+)%', line)
                        if match:
                            try:
                                epoch = int(match.group(1))
                                loss = float(match.group(2))
                                accuracy = float(match.group(3))
                                epochs.append(epoch)
                                losses.append(loss)
                                accuracies.append(accuracy)
                            except ValueError:
                                continue
            
            if epochs:
                # Créer le graphique
                plt.figure(figsize=(12, 8))
                
                # Plot Loss
                plt.subplot(2, 1, 1)
                plt.plot(epochs, losses, 'b-', linewidth=2, label='Loss')
                plt.ylabel('Loss')
                plt.title(f'Training Loss (dernier: {losses[-1]:.3f})')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Plot Accuracy
                plt.subplot(2, 1, 2)
                plt.plot(epochs, accuracies, 'r-', linewidth=2, label='Accuracy')
                plt.xlabel('Époque')
                plt.ylabel('Accuracy (%)')
                plt.title(f'Training Accuracy (dernier: {accuracies[-1]:.1f}%)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('training_monitor.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Époque {epochs[-1]:3d} | Loss: {losses[-1]:7.3f} | Acc: {accuracies[-1]:5.1f}% | Graphique mis à jour")
            else:
                print("⏳ En attente de données...")
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n🎯 Monitoring arrêté")
        if epochs:
            print(f"📈 Résumé: {len(epochs)} époques, Loss: {min(losses):.3f}-{max(losses):.3f}, Acc: {min(accuracies):.1f}%-{max(accuracies):.1f}%")

def console_monitor(log_file_path):
    """Version ultra-simple avec affichage console seulement"""
    print("📟 Mode console - Appuyez sur Ctrl+C pour arrêter")
    last_size = 0
    
    try:
        while True:
            if os.path.exists(log_file_path):
                current_size = os.path.getsize(log_file_path)
                if current_size > last_size:
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        f.seek(last_size)
                        new_lines = f.readlines()
                        last_size = current_size
                        
                        for line in new_lines:
                            match = re.search(r'Époque\s+(\d+).*?Loss:\s+([0-9.]+).*?Accuracy:\s+([0-9.]+)%', line)
                            if match:
                                epoch = int(match.group(1))
                                loss = float(match.group(2))
                                acc = float(match.group(3))
                                print(f"📊 Époque {epoch:3d} | Loss: {loss:8.4f} | Accuracy: {acc:6.2f}%")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n⏹️  Arrêté")

if __name__ == "__main__":
    LOG_FILE = "eval.txt"  # Remplacez par votre fichier
    
    # Choisissez une solution :
    
    # Solution 1: Graphique périodique (recommandé)
    simple_file_monitor(LOG_FILE, update_interval=5)
    
    # Solution 2: Avec classe
    # monitor = TrainingMonitor(LOG_FILE)
    # monitor.monitor()
    
    # Solution 3: Console seulement  
    # console_monitor(LOG_FILE)