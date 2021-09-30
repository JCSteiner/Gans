###############################################################################
# Trans_Gan.py
# J. Steiner
# implementation of trans gan from trans gan paper

#%%## LOAD DEPENDENCIES

# imports pytorch dependencies, allowing us to easily work with the pytorch
# neural network layer implementations and use them to create the transformer
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader, dataloader
from torch.autograd import Variable
from torch import autograd

MAX_LEN = 64
EMBED_MIN = -4.5259
EMBED_MAX = 4.5551

#%%## SELF-ATTENTION LAYER
class SelfAttention(nn.Module):

    '''
    Multi-headed self attention from "attention is all you need paper"
    this is the most important module in the transformer. it is an variation
    of dot product self attention from previous papers with the following
    two changes: 
        1. a scaling factor (1/sqrt(dims)) has been implemented to scale
        the outputs so high raw attention values don't push softmax to where it
        has small gradients
        2. instead of computing all dimensions of the attention at once, they
        can be divided into multiple "heads", which is an opportunity to
        compute them in parallel, although this is not yet implemented
    '''
    
    # class constructor
    # default values are value used in original transformer paper
    def __init__(self, input_dims = 512, num_heads = 8):

        # input_dims - the dimensionality of the input into the self attention
        #              layer
        # num_heads  - the number of 'heads' to the self attention we will have
        #              i.e. the number of splits of our dimensions we will have

        # runs the superclass constructor
        super(SelfAttention, self).__init__()

        # stores our parameters for use during the forward pass
        self.input_dims = input_dims
        self.num_heads = num_heads

        # computes the number of dimensions that will be in each head
        self.head_dims = input_dims // num_heads

        # we want to assert that num_heads evenly divides input_dims, if it
        # does not, then we will encounter an error later on, so it's best to
        # just flag this now
        assert (input_dims % num_heads == 0),                                 \
            'Number of heads must evenly divide embedding dimensions'

        # defines the linear layers that will cast our input vectors to values,
        # keys, queries, and a final linear layer to cast the combined heads to
        # the self attention output
        self.to_values  = nn.Linear(self.head_dims, self.head_dims, bias = False)
        self.to_keys    = nn.Linear(self.head_dims, self.head_dims, bias = False)
        self.to_queries = nn.Linear(self.head_dims, self.head_dims, bias = False)
        self.to_output  = nn.Linear(self.input_dims, self.input_dims)

    # the forward pass through the self attention layer
    def forward(self, inp_values, inp_keys, inp_queries, mask = None):

        # gets the batch size from the query input, although it can get found
        # in any of the input vectors, so the choice of queries is arbitrary
        batch_size = inp_queries.shape[0]

        # gets the size of the values, keys, and queries, values and keys will
        # always be the same, although in the case of the encoder, all 3 will
        # be the same
        src_seq_len, trg_seq_len = inp_values.shape[1], inp_queries.shape[1]

        # reshapes the inputted values, keys, and queries to be split into
        # multiple heads
        values  = inp_values.reshape(  batch_size, src_seq_len, self.num_heads, self.head_dims)
        keys    = inp_keys.reshape(    batch_size, src_seq_len, self.num_heads, self.head_dims)
        queries = inp_queries.reshape( batch_size, trg_seq_len, self.num_heads, self.head_dims)

        # passes the values, keys, and queries through their respective linear
        # layers
        values  = self.to_values(values)    # of the shape (batch, src_time, head, feature)
        keys    = self.to_keys(keys)        # of the shape (batch, src_time, head, feature)
        queries =  self.to_queries(queries) # of the shape (batch, trg_time, head, feature)
        # the time dimension referes to the element in the sequence passed in
        # also, remember that the queries come from the target sequence in the case
        # of the decoder, so 

        # this is a way of computing Q * K.transpose() from the paper, this 
        # function (with these parameters) is a way of doing batch matrix
        # mutliplication
        query_key = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])
        # query_key ends up with the shape (batch, head, trg_time, src_time)
        # this shape can be interpreted as a value of the impact of every src
        # sequence value on every target seq value
        # in the case of the encoder, remember that src_seq = trg_seq

        # if the mask has been given as a parameter (since the default is None)
        # this means that we want to mask the query_key intermediate result
        # since we don't want our model to be able to "see in the future"
        if mask is not None:
            # masks values that have not yet occurred (blots out the targets
            # for srcs not yet put in) with a value comperable to -inf
            query_key = query_key.masked_fill(mask == 0, -1.0e20)

        # calculates the attention by scaling the query_key intermediate from
        # 0-1 using softmax along the src_time dimension (the amount of attn)
        # the target should pay to each src as a fraction
        attention = torch.softmax(query_key / (self.input_dims ** 0.5), dim = 3)

        # applies the attention by paying attention to the values
        # multiplies the attention of shape: (batch, head, trg_time, src_time)
        # with the values of shape: (batch, src_time, head, feature)
        # to shape: (batch, trg_time, head, feature)
        attn_to_values = torch.einsum('bhql,blhd->bqhd', [attention, values])
        # folds the heads into a single axis, AKA combines to shape (batch, trg_time, feature)
        attn_to_values = attn_to_values.reshape(batch_size, trg_seq_len, self.input_dims)

        # passes the attn_to_values through one final linear layer
        output = self.to_output(attn_to_values)
        
        # returns the output of the self attention layer
        return output

#%%## ENCODER BLOCK DEFINITION
class Encoder_Block(nn.Module):

    # class constructor
    # default values are values used in original transformer paper
    def __init__(self, input_dims = 256, num_heads = 8, fwd_expand = 4, dropout_prob = 0.0):

        # input_dims   - the dimensionality of the input into the encoder block
        # num_heads    - the number of heads for our self attention
        # fwd_expand   - the amount of expansion of dimensionality in the feed
        #                forward layer as an intermediate step
        # dropout_prob - the probability to pass into our dropout layer

        # runs the superclass constructor
        super(Encoder_Block, self).__init__()

        # defines a self attention layer
        self.attention = SelfAttention(input_dims, num_heads)

        # defines our two layer normalizations
        self.norm1 = nn.LayerNorm(input_dims)
        self.norm2 = nn.LayerNorm(input_dims)

        # defines our 'position-wise feed forward layer' from the transformer
        # paper. it is 2 linear layers with a relu function in between
        # it expands by a factor of fwd_expand (default is 4) and then contracts
        # back to the original size
        self.feed_forward = nn.Sequential(
                            nn.Linear(input_dims, fwd_expand * input_dims),
                            nn.ReLU(),
                            nn.Linear(fwd_expand*input_dims, input_dims))

        # defines a dropout layer (0s our elements in the matrix with probability
        # dropout_prob to prevent overfitting)
        self.dropout = nn.Dropout(dropout_prob)

    # defines a forward pass through the block
    def forward(self, inp_value, inp_key, inp_query, mask = None):

        # inp_value - the vector we are assigning as a value
        # inp_key   - the vector we assign as a key
        # inp_query - the vector we assign as a query
        # maks      - defines what is marked as masked vs un-masked
        
        # passes the inputted values, keys, and queries through the self attn
        x = self.attention(inp_value, inp_key, inp_query, mask)
        # passes the attn through a layer norm and adds the pre-attn input as
        # a 'residual connection'
        x = self.norm1(x + inp_query)
        # passes the normalized x through a dropout layer
        x = self.dropout(x)
        # passes x through the position-wise feed forward network
        x = self.feed_forward(x)
        # passes the x through another layer norm with a residual connection
        x = self.norm2(x) + x
        # passes x through the dropout layer
        x = self.dropout(x)

        # returns x
        return x

#%%## GENERATOR CLASS DEFINITION
class Generator(nn.Module):

    # class constructor
    # defaults are what was used in the trans gan paper
    def __init__(self, noise_dims, hidden_dims = 256, max_len = MAX_LEN//4):

        # noise_dims - the dimensionality of the noise we input into the generator
        # hidden_dims - the hidden dimensionality of the self attention
        # max_len - the length of our positional encoding we use

        # runs the superclass constructor
        super(Generator, self).__init__()

        # stores the hidden dimensions of the network
        self.hidden_dims = hidden_dims

        # linear layer to map a noise vector to a sequence of vectors the size of the hidden dims
        self.noise_transform = nn.Linear(noise_dims, (MAX_LEN//4)*hidden_dims)

        # our encoder block stages, a stage is needed for every upscale + 1
        self.stage_1 = nn.ModuleList([ Encoder_Block(hidden_dims) for _ in range(2) ])
        # our encoder block stages, a stage is needed for every upscale + 1
        self.stage_2 = nn.ModuleList([ Encoder_Block(hidden_dims) for _ in range(2) ])

        # defines an upsample layer, low dimensionality upscales use bicubic
        self.upscale = nn.Upsample(scale_factor=2, mode='bicubic')

        # linear layer to map our final self attention output to the size of a flattened image
        self.output = nn.Sequential( nn.Linear(MAX_LEN*self.hidden_dims, MAX_LEN*256), nn.Hardtanh(EMBED_MIN, EMBED_MAX) )

        # the only interpetation of "learnable positional encoding" referenced in the paper
        self.pe = nn.Parameter(torch.zeros(1, max_len, hidden_dims))

    # forward pass through the network
    def forward(self, x):
        
        # grabs the batch size from the shape of the noise input into the network
        batch_size = x.shape[0]

        # transforms the noise 
        x = self.noise_transform(x)

        # reshapes the noise and adds positional encoding
        x = x.reshape(batch_size, MAX_LEN//4, self.hidden_dims)
        x = x + self.pe[:, :MAX_LEN//4, :self.hidden_dims]

        # loops through the first layer of encoder blocks
        for block in self.stage_1:
            x = block(x, x, x)

        # reshapes to an image so it can be upscaled
        x = x.reshape(batch_size, 4, 4, self.hidden_dims).transpose(1, 3)
        x = self.upscale(x)

        x = x.transpose(1, 3).reshape(batch_size, MAX_LEN, self.hidden_dims)
        # loops through the 2nd layer of encoder blocks
        for block in self.stage_2:
            x = block(x, x, x)

        # completely flattens and casts through the output layer
        x = x.reshape(batch_size, -1)
        x = self.output(x)

        # returns the flattened image output of the generator
        return x

#%%## DISCRIMINATOR CLASS DEFINITION
class Discriminator(nn.Module):

    # class constructor
    # defaults were used in the trans gan paper
    def __init__(self, input_dims, hidden_dims = 256, max_len = MAX_LEN):

        # runs the superclass constructor
        super(Discriminator, self).__init__()

        # stores the hidden dimensions
        self.hidden_dims = hidden_dims

        # linear layer to map an input image to a sequence of vectors the size of the hidden dims
        self.inp_transform = nn.Linear(input_dims, MAX_LEN*hidden_dims)

        # our encoder block stages, a stage is needed for every upscale + 1
        self.stage_1 = nn.ModuleList([ Encoder_Block(hidden_dims) for _ in range(1) ])
        self.stage_2 = nn.ModuleList([ Encoder_Block(hidden_dims) for _ in range(1) ])

        # defines a downsample layer, low dimensionality upscales use average pool
        self.downscale = nn.AvgPool2d(kernel_size=2)

        # linear layer to map our final self attention output to the size of a flattened image
        self.output = nn.Linear((MAX_LEN//4)*self.hidden_dims, 1)

        # the only interpetation of "learnable positional encoding" referenced in the paper
        self.pe = nn.Parameter(torch.zeros(1, max_len, hidden_dims))

    # forward pass through the network
    def forward(self, x):
        
        # grabs the batch size from the input image
        batch_size = x.shape[0]

        # transforms the input image
        x = self.inp_transform(x)

        # reshapes and adds positional encoding
        x = x.reshape(batch_size, MAX_LEN, self.hidden_dims)
        x = x + self.pe[:, :MAX_LEN, :self.hidden_dims]

        # passes x through the first layer of encoder blocks
        for block in self.stage_1:
            x = block(x, x, x)

        # reshapes to an image so it can be upscaled
        x = x = x.reshape(batch_size, int(MAX_LEN ** 0.5) , int(MAX_LEN ** 0.5), self.hidden_dims).transpose(1, 3)
        x = self.downscale(x)

        x = x.transpose(1, 3).reshape(batch_size, MAX_LEN // 4, self.hidden_dims)
        for block in self.stage_2:
            x = block(x, x, x)

        # reshapes and passes through output layer
        x = x.reshape(batch_size, -1)
        x = self.output(x)

        # returns the confidence that the image is real
        return x

#%%## DATA LOADING
from DataLoader import load_prompts
from Embedding_Trainer import NGramEmbedding

# loads in the dataset
dataset, word2Idx, idx2Word = load_prompts('./Data/writingPrompts/valid_wp_source.txt', './Data/writingPrompts/valid_wp_target.txt')

# creates an instance of the embedding layer class and loads its saved state in
embedding = NGramEmbedding(len(word2Idx), 256, 3)
embedding.load_state_dict(torch.load('./Embedding_States/embedder_uncond_dev_v1'))
embedding.eval()

all_embeds = [ embedding.embed(torch.tensor(idx)) for idx in range(len(word2Idx)) ]
# min embedding = min([torch.min(e) for e in all_embeds])
# max embedding = max([torch.max(e) for e in all_embeds])

embed_data = [ (embedding.embed(x), y) for x, y in dataset ]

# batches data
loader = DataLoader(embed_data, batch_size=50, shuffle=True)

# defines device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%## MODEL CREATION

# creates a generator and discriminator object
gen = Generator(256).to(device)
dis = Discriminator(MAX_LEN*256).to(device)

# defines optimizer
g_optim = optim.Adam(gen.parameters(), lr=1e-6, betas=(0.0, 0.99))
d_optim = optim.Adam(dis.parameters(), lr=1e-6, betas=(0.0, 0.99))

# defines a fized mini batch of noise to see progression in the model
fixednoise = torch.randn(32, 256).to(device)

# sets the generator and discriminator to training mode
gen.train()
dis.train()

#%%## MODEL TRAINING
# for 200 epochs
for epoch in range(200):

    # for each batch
    for batch_idx, (real, _) in enumerate(iter(loader)):
        
        # loads in the real data and creates noise
        real = Variable(real.reshape(real.shape[0], -1).to(device), requires_grad = True)
        noise = torch.randn(real.shape[0], 256).to(device)

        # runs real and fake image through discriminator
        dis_real = dis(real).reshape(-1)
        fake = gen(noise)
        dis_fake = dis(fake).reshape(-1)

        # zeros out discriminator gradients
        dis.zero_grad()


        # generates labels for the real and fake images
        real_lbl = Variable(torch.ones_like(dis_real), requires_grad = False)
        fake_lbl = Variable(torch.zeros_like(dis_fake), requires_grad = False)

        # wasterstein loss
        real_grad = autograd.grad(dis_real, real, real_lbl, create_graph=True, retain_graph=True, only_inputs=True)[0]
        fake_grad = autograd.grad(dis_fake, fake, fake_lbl, create_graph=True, retain_graph=True, only_inputs=True)[0]

        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (6 / 2)
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (6 / 2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * 2 / 2

        loss_dis = -torch.mean(dis_real) + torch.mean(dis_fake) + div_gp

        # backward pass through the discriminator
        loss_dis.backward(retain_graph = True)
        d_optim.step()

        # pass through the generator
        output = dis(fake).reshape(-1)
        gen.zero_grad()

        # wasserstein loss through the generator
        loss_gen = -torch.mean(output)
        
        # backward through generator
        loss_gen.backward()
        g_optim.step()

        # if we are at the first batch or a batch number divisible by 0
        if batch_idx == 0:
            
            # prints a status update
            print(f"Epoch [{epoch+1}/{200}] Batch [{batch_idx+1}/{len(loader)}] Loss D: [{loss_dis:.4f}] Loss G: [{loss_gen:.4f}]")

            # without computing gradients
            with torch.no_grad():
                
                # get fake and real images
                fake = gen(fixednoise).reshape(32, MAX_LEN, 256)
                reals = real[:32].reshape(32, MAX_LEN, 256)

                dists_fake = [[[ torch.sum(torch.square(fake[i][j] - all_embeds[k])) for k in range(len(word2Idx))] for j in range(len(fake[i]))] for i in range(len(fake))]
                dists_real = [[[ torch.sum(torch.square(reals[i][j] - all_embeds[k])) for k in range(len(word2Idx))] for j in range(len(reals[i]))] for i in range(len(reals))]

                preds = []
                targs = []
                for b in range(32):
                    pred = ''
                    targ = ''
                    for word in range(MAX_LEN):
                        word_pred = dists_fake[b][word].index(min(dists_fake[b][word]))
                        word_targ = dists_real[b][word].index(min(dists_real[b][word]))
                        pred += idx2Word[word_pred] + ' '
                        targ += idx2Word[word_targ] + ' '
                    pred += '\n'
                    targ += '\n'
                    preds.append(pred)
                    targs.append(targ)

                outfile = open('./Model_Output/output_batch_idx_' + str(batch_idx) + '_epoch_' + str(epoch) +'.txt', 'w')
                outfile.writelines(preds)
                outfile.writelines(targs)
                outfile.close()
                
        # otherwise
        else:
            # still posts a status update
            print(f"Epoch [{epoch+1}/{200}] Batch [{batch_idx}/{len(loader)}]")
        
    # every epoch, save the generator and discriminator states
    torch.save(gen.state_dict(), './Model_States/gen_state_' + str(epoch+1))
    torch.save(dis.state_dict(), './Model_States/dis_state_' + str(epoch+1))