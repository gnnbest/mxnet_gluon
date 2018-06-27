
import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import gluon, autograd, nd, init
from mxnet.gluon import loss as gloss, nn

print(sys.path)

batch_size = 256

train_iter, test_iter= gb.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init = init.Normal(sigma=0.01))

loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})

num_epoch = 2

gb.train_cpu(net, train_iter, test_iter, loss, num_epoch, batch_size, None, None, trainer)

filename = 'data/softmax.params'
print(sys.path[0])
net.save_params(sys.path[0] + '/' + filename)







