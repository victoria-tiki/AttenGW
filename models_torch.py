import torch
import torch.nn as nn
import torch.nn.functional as F

class SubModule(nn.Module):
    def __init__(self, inp_shape=(4096, 1)):
        super(SubModule, self).__init__()

        self.n_filters = 32
        self.filter_width = 2
        self.dilation_rates = [2**i for i in range(11)] * 3
        
        self.conv1_firstit = nn.Conv1d(1,16, kernel_size=1,  padding='same')
        self.conv1_postfirstit = nn.ModuleList([nn.Conv1d(16,16, kernel_size=1,  padding='same') for dilation_rate in self.dilation_rates])[:-1]
        self.convs_f = nn.ModuleList([nn.Conv1d(16,self.n_filters, kernel_size=self.filter_width, padding='same', dilation=dilation_rate) for dilation_rate in self.dilation_rates])
        self.convs_g = nn.ModuleList([nn.Conv1d(16,self.n_filters, kernel_size=self.filter_width, padding='same', dilation=dilation_rate) for dilation_rate in self.dilation_rates])
        self.conv2 = nn.ModuleList([nn.Conv1d(self.n_filters, 16, kernel_size=1,  padding='same') for dilation_rate in self.dilation_rates])
        #self.skips = []

    def forward(self, x):
        skips = []
        for i, dilation_rate in enumerate(self.dilation_rates):
            
            #in_len = 1 if i == 0 else 16
            conv1=self.conv1_firstit if i==0 else self.conv1_postfirstit[i-1]
            
            x = F.relu(conv1(x))
            x_f = self.convs_f[i](x)
            x_g = self.convs_g[i](x)
            z = F.tanh(x_f) * F.sigmoid(x_g)
            z = F.relu(self.conv2[i](z))
            x = x + z
            skips.append(z)
        out = F.relu(torch.sum(torch.stack(skips), dim=0))
        return out
        


class Attn(nn.Module):
    def __init__(self, dummy, dummy1, convolution_dim=16, num_heads=1):
        super().__init__()
        assert convolution_dim % num_heads == 0, \
            f"convolution_dim ({convolution_dim}) must be divisible by num_heads ({num_heads})."

        self.convolution_dim = convolution_dim
        self.num_heads = num_heads
        self.dim = convolution_dim // num_heads  # Dimension per head

        # Q, K, V Projections
        self.query_proj = nn.Linear(convolution_dim, convolution_dim)
        self.key_proj = nn.Linear(convolution_dim, convolution_dim)
        self.value_proj = nn.Linear(convolution_dim, convolution_dim)

        # Final projection
        self.output_proj = nn.Linear(convolution_dim, convolution_dim)

    def forward(self, target_node, neighbor):
        batch_size, conv_dim, seq_len = target_node.size()

        #projection
        target_node = target_node.permute(0, 2, 1)  # (batch_size, seq_len, convolution_dim)
        neighbor = neighbor.permute(0, 2, 1)       # (batch_size, seq_len, convolution_dim)

        Q = self.query_proj(target_node)  # (batch_size, seq_len, convolution_dim)
        K = self.key_proj(neighbor)       # (batchl_size, seq_len, convolution_dim)
        V = self.value_proj(neighbor)     # (batch_size, seq_len, convolution_dim)

        # multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # Normalize scores
        attended_values = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, dim)

        # concatenate heads/output
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, dim)
        attended_values = attended_values.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, convolution_dim)
        updated_node = self.output_proj(attended_values)  # (batch_size, seq_len, convolution_dim)

        # permute back
        output = updated_node.permute(0, 2, 1)  # (batch_size, convolution_dim, seq_len)
        return output

class full_module(nn.Module):
    def __init__(self):
        super(full_module, self).__init__()
        self.sub_mod_A = SubModule()
        self.sub_mod_B = SubModule()
        self.dim = 16
        self.node_dim = 64
        
        self.conv1d=nn.Conv1d(self.dim,1, 1) #change to dim for attn, node_dim for graph
        self.atten_A=Attn(self.dim,self.node_dim)
        self.atten_B=Attn(self.dim,self.node_dim)

    def forward(self, x):
        x_A = x[:, :, 0].view(-1, 4096, 1).permute(0,2,1)
        x_B = x[:, :, 1].view(-1, 4096, 1).permute(0,2,1)
        #x_C = x[:, :, 2].view(-1, 4096, 1).permute(0,2,1)
        
        x_A = self.sub_mod_A(x_A)
        x_B = self.sub_mod_B(x_B)

        updated_x_A = self.atten_A(x_A, x_B).permute(0,2,1).view(-1, 4096, self.dim, 1) #change to dim for attn, node_dim for graph
        updated_x_B = self.atten_B(x_B, x_A).permute(0,2,1).view(-1, 4096, self.dim, 1)#change to dim for attn, node_dim for graph

        out = torch.cat([updated_x_A, updated_x_B], dim=-1)
        out = torch.max(out, dim=-1).values#[0]

        out = self.conv1d(out.permute(0,2,1)).permute(0,2,1)
        out = F.sigmoid(out)

        return out
