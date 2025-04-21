class Recommender():
    def __init__(self):
        
        return
    
    
    def train(self):
        
        return
    
    def predict(self):
        
        return
    
    def user_item2idx(self, df, user_clm, item_clm):
        # user_clm = "user_id", item_clm = "item_id"
        self.user_clm = user_clm
        self.item_clm = item_clm
        user_ids = df[user_clm].unique()
        item_ids = df[item_clm].unique()
        self.user2idx = {uid: i for i, uid in enumerate(user_ids)}
        self.item2idx = {iid: i for i, iid in enumerate(item_ids)}
        self.idx2item = {i: iid for iid, i in self.item2idx.items()}
        self.idx2user = {i: iid for iid, i in self.user2idx.items()}

        df["user_idx"] = df[user_clm].map(self.user2idx)
        df["item_idx"] = df[item_clm].map(self.item2idx)
        
        return df