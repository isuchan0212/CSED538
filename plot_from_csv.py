import matplotlib.pyplot as plt
import csv

# CSV 파일로부터 데이터 읽기
def read_logs(csv_file):
    epochs = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            train_accuracies.append(float(row['train_accuracy']))
            val_accuracies.append(float(row['val_accuracy']))
            test_accuracies.append(float(row['test_accuracy']))

    return epochs, train_losses, val_losses, train_accuracies, val_accuracies, test_accuracies

# 그래프 그리기
def plot_logs(epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Train Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Train and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Train and Validation Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # CSV 파일 경로
    csv_file = 'training_logs.csv'
    
    # CSV로부터 데이터 읽기
    epochs, train_losses, val_losses, train_accuracies, val_accuracies, test_accuracies = read_logs(csv_file)
    
    # 그래프 그리기
    plot_logs(epochs, train_losses, val_losses, train_accuracies, val_accuracies)
