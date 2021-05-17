# Object-oriented pytorch and the basis for xt-training

The following schematic captures the core components of the vast majority of deep learning training and inference workflows. The main purpose of this figure is to demonstrate the difference between the abstractable concepts in these workflows (i.e., the arrows/connections) and the configuration relevant to a particular problem (i.e., the classes and their attributes that live in the boxes). There are of course exceptions to this structure, but the vast majority of problems are captured by this workflow, even if we have to group together typically distinct concepts (such as data loaders and text tokenizers for NLP).

The importance of being able to define this workflow is that it immediately suggests a robust set of procedures/functions that can be applied to deep learning in a general manner. A very common thing (read: a very common problem) in ML is to re-write a large amount of very similar training code for each new project. This results in problematic and unnecessary inconsistencies in structure and style between different projects, beyond the obvious wasted time taken to re-write what is basically the same thing over and over.

This is the rationale behind xt-training, a python package authored by Xtract to capture all the common parts of the workflow below. There are of course other packages that aim to do similar things (most call themselves “high-level wrappers” for popular deep learning libraries), and some of them are very good (FastAI in particular is excellent). The main (and I believe necessary) advantage of xt-training over these alternatives is that it does not change the way in which the user interacts with the underlying framework (pytorch). Rather, it supports that interaction by taking care of the tedious but necessary boilerplate code concerned with training/testing loops, logging, and model persistence.

![](assets/pytorch-workflow.jfif)

## Example of evaluation loop

To demonstrate the utility of suitable deep learning code abstractions, I will compare a standard test dataset evaluation in pytorch with and without xt-training:

* **Without xt-training:**

```python
import torch
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

test_dataset = datasets.ImageFolder(
    'path/to/images',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
)

test_loader = DataLoader(dataset, batch_size=32, num_workers=4)

model = models.resnet18(pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

loss_fn = nn.CrossEntropyLoss()

loss_sum = 0
acc_sum = 0
cnt = 0
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
    
        logits = model(x)
        
        loss = loss_fn(y, logits)
        loss = loss.cpu().item()
        
        y_pred = nn.functional.softmax(logits, dim=1).argmax(dim=1)
        acc = (y_pred == y).float().mean()
        acc = acc.cpu().item()
        
        loss_sum = loss_sum + loss
        acc_sum = acc_sum + acc
        cnt += len(y)
        
        print(f'Batch {i}/{len(test_loader)} | loss: {loss}, acc: {acc}')
        print(f'    Running loss: {loss_sum / cnt}, running acc: {acc / cnt}')
```

* **With xt-training:**

```python
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from xt_training import metrics
from xt_training.utils import functional as F

test_dataset = datasets.ImageFolder(
    'path/to/images',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
)

test_loader = DataLoader(dataset, batch_size=32, num_workers=4)

model = models.resnet18(pretrained=True)

loss_fn = nn.CrossEntropyLoss()
eval_metrics = {
    'acc': metrics.Accuracy(),
    'auc': metrics.ROC_AUC(),
    'eps': metrics.EPS(),
}

F.test(
    test_loaders = {'test': test_loader},
    model=model,
    loss_fn=loss_fn,
    eval_metrics=eval_metrics
)
```

The two code snippets above perform the same tasks. The most important point to note is that, other than a single call to `F.test`, the xt-training example contains only problem-specific configuration. On the other hand, a large proportion of the first code snippet consists of implementing boilerplate dataset looping and metric evaluation. This overhead becomes significantly worse for training loops and when a larger number of evaluation metrics are required.

Perhaps the most obvious difference between the two examples is that the first requires more lines of code. This is of course important, but not nearly as important as the difference in the code’s maintainability. Since everything in the second snippet is configuration/architecture specific to the particular problem, it necessarily only exists in one place. In the first snippet, however, the test loop code probably looks very similar to test loops in dozens of other places. Hence, to keep consistent structure and style in our code, we would need to update all those various incarnations of the loop separately, which would almost certainly not happen. The next section describes how the key principles of OOP allow us to abstract deep learning workflows, such as the test loop shown above, without any real loss of flexibility.

## OOP for Deep Learning

One of the most profound differences between Tensorflow and Pytorch is that the latter has been designed specifically for extensibility to be the main method of use. By this I mean the vast majority of pytorch models are defined as classes inheriting from `nn.Module` and, as a result, all pytorch models have a common set of attributes and methods. This represents a tight adherence to standardised best practices in object-oriented programming and, in turn, lends itself to be naturally incorporated into projects with a similar adherence. Furthermore, the establishment of common class structures and method signatures (a method “signature” is the specific definition of inputs and outputs for that method) enables us to develop generalised methods for using them.

## Methods and method signatures

The xt-training package relies on several key concepts in OOP to operate effectively:

* Workflow objects (e.g., data loaders, models, optimisers, schedulers, metrics, etc) for all different problem types have a consistent method *signature*.
* What actually happens in these methods is arbitrary and problem-specific and can be dependent on class attributes.

Together, these two points allow xt-training to be both general enough and flexible enough for a wide variety of deep learning problems from computer vision to NLP and even reinforcement learning.

Below are two examples of model classes that address very different problems but have identical signatures for their forward methods.

* **2D CNN:**

```python
class CVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

* **1D RNN:**

```python
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(32, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 2)
    
    def forward(self, x):
        batch_size = x.shape[0]
        hidden = self.init_hidden(batch_size)
        x, hidden = self.rnn(x, hidden)
        x = x.contiguous().view(-1, 128)
        x = self.fc(x)
        return x
    
    def init_hidden(self, batch_size):
        return torch.zeros(2, 32, 128)
```

Although these classes contain very different attributes and methods, the methods that lie in the path of the standard deep learning workflow (i.e., the `forward` method) have an identical signature: in both cases `x` is a batch of samples returned from a (problem-specific) data loader and the return value is the predictions for that batch. Furthermore, the outputs in both cases form one of the inputs to the loss function and evaluation metrics. As a result, the code surrounding calls to these methods can be standardised. The same can be said for data loaders, optmizers, schedulers, loss functions & evaluation metrics, and logging classes: all the entities in boxes in the flow chart above. This enables for standardisation of the wrapping code implemented in xt-training. For instance:

```python
for x, y in loader:
    x.to(device)
    y.to(device)
    pred = model(x)
    loss = loss_fn(y, pred)
    ...
```

The actual code in xt-training is slightly more complex to account for some edge-cases, but this is the gist. The key point is that the above code works perfectly well with both model definitions above, as well as with any number of loader and loss_fn implementations, provided they follow the required signature pattern.
