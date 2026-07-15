import torch
import torch.nn as nn

class SpatioTemporalGRUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalGRUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2

        # conv_x outputs for r, z, n for input x_t
        # conv_h outputs for r, z, n for hidden h_t
        # conv_m outputs for r, z, n for memory m_t (optional)

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 6, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            # output conv similar to conv_o for gating output
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )

            self.conv_n = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Conv2d(in_channel, num_hidden * 6, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
            self.conv_h = nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
            self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
            self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
            self.conv_n = nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        # We keep c_t for interface compatibility but won't update it (GRU doesn't use it)
        # The main states we update are h_t and m_t

        # Compute gates for input
        x_concat = self.conv_x(x_t)      # shape: [B, 3*num_hidden, W, W]
        h_concat = self.conv_h(h_t)      # shape: [B, 3*num_hidden, W, W]
        m_concat = self.conv_m(m_t)      # shape: [B, 3*num_hidden, W, W]

        # Split for reset (r), update (z), new candidate (n)
        i_r, i_z, i_n, i_r_m, i_z_m, i_n_m = torch.split(x_concat, self.num_hidden, dim=1)
        h_r, h_z, h_n = torch.split(h_concat, self.num_hidden, dim=1)
        m_r, m_z, m_n = torch.split(m_concat, self.num_hidden, dim=1)

        # Reset gate
        r_t = torch.sigmoid(i_r + h_r + m_r)
        # Update gate
        z_t = torch.sigmoid(i_z + h_z + m_z)
        # New candidate hidden state
        n_t = torch.tanh(i_n + self.conv_n(r_t * h_n) + m_n)

        # Update hidden state
        h_new = (1 - z_t) * n_t + z_t * h_t

        # Update memory state m similarly (optional, to keep same structure)
        # Let's treat m_t like a second GRU hidden state updated the same way but independent.
        r_t_m = torch.sigmoid(i_r_m + m_r)  # reuse gates for m update, or define separate gates if you want more complexity
        z_t_m = torch.sigmoid(i_z_m + m_z)
        n_t_m = torch.tanh(i_n_m + r_t_m * m_n)
        m_new = (1 - z_t_m) * n_t_m + z_t_m * m_t

        # Concatenate h_new and m_new and pass through conv_o and conv_last to get final output h_t
        mem = torch.cat((h_new, m_new), dim=1)
        o_t = torch.sigmoid(self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        # c_t is unused in GRU but kept for interface compatibility
        return h_new, c_t, m_new
