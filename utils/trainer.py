from .model import Backbone, MobileFaceNet, l2_norm
import torch
from torchvision import transforms as trans

class face_learner(object):
    def __init__(self, conf, threshold):
        self.user_config = conf
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        
        self.threshold = threshold
    
    def load_state(self, fixed_str):
        # print("Loading model from", fixed_str)
        loaded = torch.load(fixed_str, map_location=self.user_config.device)
        self.model.load_state_dict(loaded)
    
    def infer(self, conf, image, target_embs, tta, labels):
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
        sorted_idx = torch.argsort(dist, dim=1)
        
        return extract_result(sorted_idx, labels, target_embs, source_embs, conf.similarity_limit)

def extract_result(sorted_idxs, labels, target_embds, source_embd, limit=5):
    pred_labels = []
    similarities = []
    for idx in sorted_idxs[0]:
        if pred_labels == [] or (labels[idx][0] not in pred_labels):
            pred_labels.append(labels[idx][0])
            similarities.append(torch.mm(target_embds[torch.unsqueeze(idx, 0)], source_embd.transpose(0,1)).detach().cpu().numpy()[0][0])
        
        if len(pred_labels) == limit:
            break
    return pred_labels, [round(float(s), 4) for s in similarities]
