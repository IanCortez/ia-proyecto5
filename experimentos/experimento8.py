import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math



# Cambiar el dispositivo a usar al GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


batch_size = 32


img_transform = transform.Compose([transform.ToTensor(), transform.Normalize((0.5,),(0.5,))])
dataset = torchvision.datasets.ImageFolder(root="D:/ProgrammingProjects/utec/2021-2/inteligencia_artificial/proyecto5/COVID-19_Radiography_Dataset", 
	transform=img_transform)


# Generar los sets de training, validation y testing
print(len(dataset))
train_set,test_set=torch.utils.data.random_split(dataset,[14815,6350], generator=torch.Generator().manual_seed(0))
test_set,val_set=torch.utils.data.random_split(test_set,[4233,2117], generator=torch.Generator().manual_seed(0))


img, _ = train_set[0]
print(img.shape)


train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)



def show_img(img):
	plt.imshow(img.numpy()[0], cmap='gray')
	plt.show()


print(len(train_set))
print(len(test_set))
print(len(val_set))


img, label = train_set[999]
print(label)
show_img(img)


num_classes = 4
learning_rate =  0.01
num_epochs = 20

	

class CNN(nn.Module):
	def __init__(self, num_classes=4):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=16, kernel_size=10, stride=1, padding=0),
			nn.Dropout2d(0.25),
			nn.ReLU(),
			nn.BatchNorm2d(16)
			)
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=6, stride=1, padding=2),
			nn.Dropout2d(0.25),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.BatchNorm2d(32)
			)
		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
			nn.Dropout2d(0.25),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.BatchNorm2d(64)
			)
		self.layer4 =  nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
			nn.Dropout2d(0.25),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.BatchNorm2d(128)
			)
		self.fc = nn.Linear(37*37*128, num_classes)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		return out


model         = CNN(num_classes).to(device)
loss_fn       = nn.CrossEntropyLoss()
optimizer     = torch.optim.Adam(model.parameters(), lr = learning_rate)
#loss_train    = train(model, optimizer, loss_fn, num_epochs)
#test(model)

print([ e.shape  for e in model.fc.parameters()])

model.fc.weight



def train(model, optimizer, loos_fn, num_epochs):
	loss_vals = []
	running_loss =0.0
	# train the model
	total_step = len(train_loader)

	list_loss= []
	list_time = []
	j=0

	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = images.to(device)
			labels = labels.to(device)
			# forward 
			output = model(images)
			loss   = loss_fn(output, labels)
			# change the params
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			list_loss.append(loss.item())
			list_time.append(j)
			j+=1
				
			if (i+1) % 100 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
		
	print('Finished Training Trainset')
	return list_loss, list_time


error_list, list_time = train(model,optimizer,loss_fn,num_epochs)


print(max(error_list))
plt.plot(list_time, error_list, color="green")
plt.xlabel("Time")
plt.ylabel("Error training")
plt.show()



with torch.no_grad():
	correct = 0
	total = 0

	for images, labels in train_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	train_acc = 100 * correct / total;


	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	test_acc = 100 * correct / total

	for images, labels in val_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	valid_acc = 100 * correct / total

	print('Train Accuracy: {} %'.format(train_acc))
	print('Test Accuracy: {} %'.format(test_acc))
	print('Validation Accuracy: {} %'.format(valid_acc))