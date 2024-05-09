import matplotlib.pyplot as plt

# Data can be changed depending on the results
epochs = list(range(1, 11))
train_accuracy = [0.4774, 0.7425, 0.7865, 0.8179, 0.8057, 0.8263, 0.8321, 0.8420, 0.8355, 0.8412]
val_accuracy = [0.7485, 0.8136, 0.8225, 0.8314, 0.8195, 0.8284, 0.8343, 0.8314, 0.8225, 0.8402]
train_loss = [1.2225, 0.7381, 0.6123, 0.5518, 0.5384, 0.4906, 0.4765, 0.4589, 0.4738, 0.4535]
val_loss = [0.7700, 0.6102, 0.5547, 0.5294, 0.5137, 0.5016, 0.4938, 0.4914, 0.4867, 0.4864]

# Plotting
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'b^-', label='Training Loss')
plt.plot(epochs, val_loss, 'r^-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()