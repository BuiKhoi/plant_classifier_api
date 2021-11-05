from .model import Backbone, MobileFaceNet, l2_norm
import torch
from torchvision import transforms as trans

class face_learner(object):
    def __init__(self, conf, threshold):
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        
        self.threshold = threshold
    
    def load_state(self, fixed_str):
        # print("Loading model from", fixed_str)
        self.model.load_state_dict(torch.load(fixed_str))
    
    def infer(self, conf, image, target_embs, tta=False):
        '''
        image : PIL Image
        target_embs : [n, 512] computed embeddings of spieces in databank
        names : recorded names of spieces in databank
        tta : test time augmentation (hfilp, that's all)
        '''
        self.model.eval()
        embs = []
        if tta:
            mirror = trans.functional.hflip(image)
            emb = self.model(conf.test_transform(image).to(conf.device).unsqueeze(0))
            emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
            embs.append(l2_norm(emb + emb_mirror))
        else:                        
            embs.append(self.model(conf.test_transform(image).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        target_embs = torch.tensor(target_embs).to(conf.device)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        # print(minimum)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum
