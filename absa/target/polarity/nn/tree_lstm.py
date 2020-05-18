import torch as th
import torch.nn as nn
import dgl


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size: int, h_size: int):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox['h']))
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild) + nodes.data['iou'], 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, device: th.device):
        super(TreeLSTM, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tree_lstm = ChildSumTreeLSTMCell(self.input_dim, self.output_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, rnn_inputs: th.Tensor, graph: dgl.DGLGraph):
        g = graph
        g.register_message_func(self.tree_lstm.message_func)
        g.register_reduce_func(self.tree_lstm.reduce_func)
        g.register_apply_node_func(self.tree_lstm.apply_node_func)
        g.ndata['iou'] = self.tree_lstm.W_iou(rnn_inputs)
        g.ndata['h'] = self.linear(rnn_inputs)
        g.ndata['c'] = th.autograd.Variable(th.zeros(
            (graph.number_of_nodes(), self.output_dim), dtype=th.float32),
                                            requires_grad=False).to(self.device)
        dgl.prop_nodes_topo(g)

        return g.ndata.pop('h')
