import numpy as np
from torch.utils.data import Dataset
import kaldiio


class MyDataset(Dataset):
    def __getitem__(self, index):
        key, recipe = self.pair_list[index]
        # recipe is in format "noisy_feat:clean_feat"
        assert len(recipe.split(':')) == 2
        noisy_feat_path, clean_feat_path = recipe.split(':')
        noisy_feat = kaldiio.load_mat(noisy_feat_path)
        clean_feat = kaldiio.load_mat(clean_feat_path)
        # feat in (T, D)
        dim = clean_feat.shape[1]
        noisy = np.zeros((self.seg_length, dim), dtype=np.float32)
        clean = np.zeros((self.seg_length, dim), dtype=np.float32)
        assert noisy_feat.shape[0] != clean_feat.shape[0], \
            f"The numbers of clean and noisy features do not match ({noisy_feat.shape[0]} != {clean_feat.shape[0]})."
        if clean.shape[0] <= self.seg_length:
            noisy[:noisy_feat.shape[0], :] = noisy_feat
            clean[:clean_feat.shape[0], :] = clean_feat
            lens = clean_feat.shape[0]
        else:
            speech_offset = np.random.randint(0, clean_feat.shape[0] - self.seg_length)
            noisy = noisy_feat[speech_offset: speech_offset+self.seg_length, :]
            clean = clean_feat[speech_offset: speech_offset+self.seg_length, :]
            lens = self.seg_length

        one_item = {"noisy": noisy, "speech": clean, "length": np.array([lens], np.int32), "utt_id": key,
                    "utt_rcp": recipe}
        return one_item

    def __len__(self):
        return len(self.pair_list)

    @staticmethod
    def read_utter_from_scp(recipe_path):
        with open(recipe_path, 'r') as f:
            return list(map(lambda x: x.strip().split(' '), f.readlines()))

    def __init__(self, recipe_path, seg_length):
        self.scp_path = recipe_path
        self.seg_length = seg_length
        self.pair_list = self.read_utter_from_scp(self.scp_path)
