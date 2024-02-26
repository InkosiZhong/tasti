import torch
import torchvision
import tasti
import numpy as np
from tqdm.autonotebook import tqdm
import pickle
import os, shutil
from torch.utils.tensorboard import SummaryWriter

class Index:
    def __init__(self, config):
        self.config = config
        if self.config.do_training:
            self.target_dnn_cache = tasti.DNNOutputCache(
                self.get_target_dnn(),
                self.get_target_dnn_dataset(train_or_test='train'),
                self.target_dnn_callback
            )
            self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='train')
        else:
            self.target_dnn_cache = None
        self.seed = self.config.seed
        self.rand = np.random.RandomState(seed=self.seed)
        
    def override_target_dnn_cache(self, target_dnn_cache, train_or_test='train'):
        '''
        This allows you to override tasti.utils.DNNOutputCache if you already have the target dnn
        outputs available somewhere. Returning a list or another 1-D indexable element will work.
        '''
        return target_dnn_cache
        
    def is_close(self, a, b):
        '''
        Define your notion of "closeness" as described in the paper between records "a" and "b".
        Return a Boolean.
        '''
        raise NotImplementedError
        
    def get_target_dnn_dataset(self, train_or_test='train'):
        '''
        Define your target_dnn_dataset under the condition of "train_or_test".
        Return a torch.utils.data.Dataset object.
        '''
        raise NotImplementedError
    
    def get_embedding_dnn_dataset(self, train_or_test='train'):
        '''
        Define your embedding_dnn_dataset under the condition of "train_or_test".
        Return a torch.utils.data.Dataset object.
        '''
        raise NotImplementedError
        
    def get_target_dnn(self):
        '''
        Define your Target DNN.
        Return a torch.nn.Module.
        '''
        raise NotImplementedError
        
    def get_embedding_dnn(self):
        '''
        Define your Embeding DNN.
        Return a torch.nn.Module.
        '''
        raise NotImplementedError
        
    def get_pretrained_embedding_dnn(self):
        '''
        Define your Embeding DNN.
        Return a torch.nn.Module.
        '''
        return self.get_pretrained_embedding_dnn()
        
    def target_dnn_callback(self, target_dnn_output):
        '''
        Often times, you want to process the output of your target dnn into something nicer.
        This function is called everytime a target dnn output is computed and allows you to process it.
        If it is not defined, it will simply return the input.
        '''
        return target_dnn_output

    def do_mining(self):
        '''
        The mining step of constructing a TASTI. We will use an embedding dnn to compute embeddings
        of the entire dataset. Then, we will use FPFRandomBucketter to choose "distinct" datapoints
        that can be useful for triplet training.
        '''
        print('[Stage] Index Mining')
        if self.config.do_mining:
            model = self.get_pretrained_embedding_dnn()
            try:
                model.cuda()
                model.eval()
            except:
                pass

            dataset = self.get_embedding_dnn_dataset(train_or_test='train')
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0, # not support
                pin_memory=False
            )
            
            embeddings = []
            for batch in tqdm(dataloader, desc='Embedding DNN'):
                batch = batch.cuda()
                with torch.no_grad():
                    output = model(batch).cpu()
                embeddings.append(output)  
            del dataloader
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.numpy()
            
            bucketter = tasti.bucketters.FPFRandomBucketter(self.config.nb_train, self.seed)
            reps, _, _ = bucketter.bucket(embeddings, self.config.max_k)
            self.training_idxs = reps
            np.save(os.path.join(self.config.cache_root, 'training_idxs.npy'), self.training_idxs)
        else:
            '''self.training_idxs = self.rand.choice(
                    len(self.get_embedding_dnn_dataset(train=True)),
                    size=self.config.nb_train,
                    replace=False
            )'''
            self.training_idxs = np.load(os.path.join(self.config.cache_root, 'training_idxs.npy'))
            
    def do_training(self):
        '''
        Fine-tuning the embedding dnn via triplet loss. 
        '''
        print('[Stage] Index Training')
        if self.config.do_training:
            model = self.get_target_dnn()
            model.eval()
            model.cuda()
            
            for idx in tqdm(self.training_idxs, desc='Target DNN'):
                self.target_dnn_cache[idx]
            
            dataset = self.get_embedding_dnn_dataset(train_or_test='train')
            triplet_dataset = tasti.data.TripletDataset(
                dataset=dataset,
                target_dnn_cache=self.target_dnn_cache,
                list_of_idxs=self.training_idxs,
                is_close_fn=self.is_close,
                length=self.config.nb_training_its
            )
            dataloader = torch.utils.data.DataLoader(
                triplet_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0, # not support
                pin_memory=False
            )
            
            model = self.get_embedding_dnn()
            model.train()
            model.cuda()
            loss_fn = tasti.TripletLoss(self.config.train_margin)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.train_lr)
            
            tb_path = os.path.join(self.config.cache_root, 'logger')
            if os.path.exists(tb_path):
                shutil.rmtree(tb_path)
            os.makedirs(tb_path)
            writer = SummaryWriter(tb_path)
            
            for i, (anchor, positive, negative) in enumerate(dataloader):
                anchor = anchor.cuda(non_blocking=True)
                positive = positive.cuda(non_blocking=True)
                negative = negative.cuda(non_blocking=True)
                
                e_a = model(anchor)
                e_p = model(positive)
                e_n = model(negative)
                
                optimizer.zero_grad()
                loss = loss_fn(e_a, e_p, e_n)
                print(f'[Training] iter={i:05d}/{len(dataloader):05d} loss={loss}')
                writer.add_scalar('Train Loss', loss, i)
                loss.backward()
                optimizer.step()
            torch.save(model.state_dict(), os.path.join(self.config.cache_root, 'model.pt'))
            self.embedding_dnn = model
        elif os.path.exists(os.path.join(self.config.cache_root, 'model.pt')):
            model = self.get_embedding_dnn()
            model.cuda()
            checkpoint = torch.load(os.path.join(self.config.cache_root, 'model.pt'))
            model.load_state_dict(checkpoint)
            self.embedding_dnn = model
            print('loaded model.pt from cache')
        else:
            self.embedding_dnn = self.get_pretrained_embedding_dnn()
            
    def do_infer(self):
        '''
        With our fine-tuned embedding dnn, we now compute embeddings for the entire dataset.
        '''
        print('[Stage] Index Inferring')
        del self.target_dnn_cache
        self.target_dnn_cache = tasti.DNNOutputCache(
            self.get_target_dnn(),
            self.get_target_dnn_dataset(train_or_test='test'),
            self.target_dnn_callback
        )
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='test')

        if self.config.do_infer:
            model = self.embedding_dnn
            model.eval()
            model.cuda()
            dataset = self.get_embedding_dnn_dataset(train_or_test='test')
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0, # not support
                pin_memory=False
            )

            embeddings = []
            for batch in tqdm(dataloader, desc='Inference'):
                try:
                    batch = batch.cuda()
                except:
                    pass
                with torch.no_grad():
                    output = model(batch).cpu()
                embeddings.append(output)  
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.numpy()

            np.save(os.path.join(self.config.cache_root, 'embeddings.npy'), embeddings)
            
            self.embeddings = embeddings
        else:
            self.embeddings = np.load(os.path.join(self.config.cache_root, 'embeddings.npy'))
        
    def do_bucketting(self):
        '''
        Given our embeddings, cluster them and store the reps, topk_reps, and topk_dists to finalize our TASTI.
        '''
        print('[Stage] Index Bucketting')
        if self.config.do_bucketting:
            bucketter = tasti.bucketters.FPFRandomBucketter(self.config.nb_buckets, self.seed)
            self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k)
            
            np.save(os.path.join(self.config.cache_root, 'reps.npy'), self.reps)
            np.save(os.path.join(self.config.cache_root, 'topk_reps.npy'), self.topk_reps)
            np.save(os.path.join(self.config.cache_root, 'topk_dists.npy'), self.topk_dists)
        else:
            self.reps = np.load(os.path.join(self.config.cache_root, 'reps.npy'))
            self.topk_reps = np.load(os.path.join(self.config.cache_root, 'topk_reps.npy'))
            self.topk_dists = np.load(os.path.join(self.config.cache_root, 'topk_dists.npy'))
            
    def crack(self):
        print('[Stage] Index Cracking')
        cache = self.target_dnn_cache.cache
        cached_idxs = []
        for idx in range(len(cache)):
            if cache[idx] != None:
                cached_idxs.append(idx)        
        cached_idxs = np.array(cached_idxs)
        bucketter = tasti.bucketters.CrackingBucketter(self.config.nb_buckets)
        self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k, cached_idxs)
        np.save('./cache/reps.npy', self.reps)
        np.save('./cache/topk_reps.npy', self.topk_reps)
        np.save('./cache/topk_dists.npy', self.topk_dists)

    def init(self):
        torch.cuda.empty_cache()
        self.do_mining()
        torch.cuda.empty_cache()
        self.do_training()
        torch.cuda.empty_cache()
        self.do_infer()
        torch.cuda.empty_cache()
        self.do_bucketting()
        torch.cuda.empty_cache()

        is_online = isinstance(self.target_dnn_cache, tasti.DNNOutputCache)
        modified = self.config.do_mining or self.config.do_training or \
            self.config.do_infer or self.config.do_bucketting
        path_to_dnn_cache = os.path.join(self.config.cache_root, 'reps_dnn_cache.pkl')
        if is_online and not modified and os.path.exists(path_to_dnn_cache):
            with open(path_to_dnn_cache, 'rb') as f:
                reps_dnn_cache = pickle.loads(f.read())
            print('load reps_dnn_cache.pkl')
        else:
            reps_dnn_cache = None
        cache_to_save = {}
        for rep in tqdm(self.reps, desc='Target DNN Invocations'):
            if reps_dnn_cache is not None:
                self.target_dnn_cache[rep] = reps_dnn_cache[rep]
                continue
            value = self.target_dnn_cache[rep]
            if is_online and reps_dnn_cache is None:
                cache_to_save[rep] = value
        if is_online and reps_dnn_cache is None:
            with open(path_to_dnn_cache, 'wb') as f:
                str = pickle.dumps(cache_to_save)
                f.write(str)