import networks
import torch
import torch.nn as nn

class DDFSeg(nn.Module):
    def __init__(self, opts):
        super(DDFSeg, self).__init__()

        self.opts = opts
        self.parts = opts.parts
        # etc....

    def initialize_model(self):
    # initialize with gaussian weights (????)
        self.disA.apply(networks.gaussian_weights_init)
    
    # Set scheduler??
        
    def setgpu(self, gpu):
        self.gpu = gpu
        self.parts.cuda(self.gpu)
        # etc

    def forward(self, x):
        # TO DO
        return x

    def update_parts(self):
        # TO DO
        self.parts_optimizer.zero_grad()
        loss = self.backward()
        self.loss_item = loss.item()
        self.parts_optimizer.step()

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir)
        # etc
        
    
    def save(self, filename, ep, total_it):
        state = {
                'disA': self.disA.state_dict(),
                'disA2': self.disA2.state_dict(),
                'disB': self.disB.state_dict(),
                'disB2': self.disB2.state_dict(),
                'disContent': self.disContent.state_dict(),
                'enc_c': self.enc_c.state_dict(),
                'enc_a': self.enc_a.state_dict(),
                'gen': self.gen.state_dict(),
                'disA_opt': self.disA_opt.state_dict(),
                'disA2_opt': self.disA2_opt.state_dict(),
                'disB_opt': self.disB_opt.state_dict(),
                'disB2_opt': self.disB2_opt.state_dict(),
                'disContent_opt': self.disContent_opt.state_dict(),
                'enc_c_opt': self.enc_c_opt.state_dict(),
                'enc_a_opt': self.enc_a_opt.state_dict(),
                'gen_opt': self.gen_opt.state_dict(),
                'ep': ep,
                'total_it': total_it
                }
        torch.save(state, filename)


