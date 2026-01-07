import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    """
    A bidirectional LSTM layer with an optional linear layer on top.
    """
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        # input shape: (seq_len, batch, nIn)
        recurrent, _ = self.rnn(input)
        # recurrent shape: (seq_len, batch, nHidden * 2)
        
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        
        output = self.embedding(t_rec)
        # output shape: (T * b, nOut)
        
        output = output.view(T, b, -1)
        # output shape: (seq_len, batch, nOut)
        
        return output

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for OCR.
    """
    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = nn.Sequential()
        
        # --- CNN Backbone (Convolutional Layers) ---
        
        # Input: (nc, 64, W)
        self.cnn.add_module('conv0', nn.Conv2d(nc, 64, 3, 1, 1))
        self.cnn.add_module('relu0', nn.ReLU(True))
        self.cnn.add_module('pool0', nn.MaxPool2d(2, 2))
        # (64, 32, W/2)

        self.cnn.add_module('conv1', nn.Conv2d(64, 128, 3, 1, 1))
        self.cnn.add_module('relu1', nn.ReLU(True))
        self.cnn.add_module('pool1', nn.MaxPool2d(2, 2))
        # (128, 16, W/4)

        self.cnn.add_module('conv2', nn.Conv2d(128, 256, 3, 1, 1))
        self.cnn.add_module('bn2', nn.BatchNorm2d(256))
        self.cnn.add_module('relu2', nn.ReLU(True))
        # (256, 16, W/4)

        self.cnn.add_module('conv3', nn.Conv2d(256, 256, 3, 1, 1))
        self.cnn.add_module('relu3', nn.ReLU(True))
        self.cnn.add_module('pool2', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        # (256, 8, W/4)

        self.cnn.add_module('conv4', nn.Conv2d(256, 512, 3, 1, 1))
        self.cnn.add_module('bn4', nn.BatchNorm2d(512))
        self.cnn.add_module('relu4', nn.ReLU(True))
        # (512, 8, W/4)

        self.cnn.add_module('conv5', nn.Conv2d(512, 512, 3, 1, 1))
        self.cnn.add_module('relu5', nn.ReLU(True))
        self.cnn.add_module('pool3', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        # (512, 4, W/4)
        
        self.cnn.add_module('conv6', nn.Conv2d(512, 512, 2, 1, 0))
        self.cnn.add_module('bn6', nn.BatchNorm2d(512))
        self.cnn.add_module('relu6', nn.ReLU(True))
        # (512, 3, W_feat) - This was the old output, H=3
        
        # --- NEW LAYER TO FIX ASSERTION ERROR ---
        # This layer collapses the height from 3 to 1
        # It uses a (3,1) kernel: 3 high, 1 wide
        self.cnn.add_module('conv7', nn.Conv2d(512, 512, (3,1), (1,1), (0,0)))
        self.cnn.add_module('bn7', nn.BatchNorm2d(512))
        self.cnn.add_module('relu7', nn.ReLU(True))
        # (512, 1, W_feat) - New output, H=1
        # --- END OF FIX ---

        # --- RNN Backbone (Recurrent Layers) ---
        
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # 1. Convolutional features
        # input shape: (batch, channels, height, width)
        conv = self.cnn(input)
        # conv shape: (batch, 512, 1, W_feat)
        
        b, c, h, w = conv.size()
        
        # This assertion will now pass
        assert h == 1, f"The height of conv output must be 1, but got {h}"
        
        # Reshape for RNN: (batch, 512, W_feat) -> (W_feat, batch, 512)
        conv = conv.squeeze(2) # Remove height dim
        conv = conv.permute(2, 0, 1) # (W_feat, batch, 512)
        
        # 2. Recurrent features
        output = self.rnn(conv)
        # output shape: (seq_len, batch, nclass)
        
        # 3. Log softmax for CTC Loss
        output = nn.functional.log_softmax(output, dim=2)
        
        return output

