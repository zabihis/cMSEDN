import os
from collections import Counter
import dgl
from dgl.data import DGLDataset, save_graphs, load_graphs
import numpy as np
from dgl.data.utils import save_info, load_info
from dgl.nn.pytorch import EdgeWeightNorm
from gensim.models import Word2Vec
from tqdm import tqdm
import torch as t
from sklearn.neighbors import KernelDensity
from itertools import product


class seqProcessing(DGLDataset):
    """
        url : str
            The url to download the original dataset.
        raw_dir : str
            Specifies the directory where the downloaded data is stored or where the downloaded data is stored. Default: ~/.dgl/
        save_dir : str
            The directory where the finished dataset will be saved. Default: the value specified by raw_dir
        force_reload : bool
            If or not to re-import the dataset. Default: False
        verbose : bool
            Whether to print progress information.
        """
    
    def __init__(self,
                 url=None,  # Kept for parent class but unused here
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 shared_lab2id=None,
                 params=None):
        self.shared_lab2id = shared_lab2id  
        self.params=params
        super(seqProcessing, self).__init__(name='nodeEmbedding',
                                            url=url,
                                            raw_dir=raw_dir,
                                            save_dir=save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose
                                            )      
        
    def process(self):
        self.k=self.params.k
        raw_dir=self.raw_dir
        self.id2kmers, self.kmers2id = self.kmers_gen(self.params.k)
        print(f'- Loading the raw_data: {raw_dir}')

        ## loading dataset
        seqNum=0
        with open(raw_dir, 'r') as f:
                    self.data = []
                    next(f)
                    for i in f:
                        self.data.append(i.strip('\n').split(','))
                        seqNum+=1
        self.rawLab = [i[1] for i in self.data] 
        self.kmer_RNA = [[i[0][j:j + self.k] for j in range(len(i[0]) - self.k + 1)] for i in self.data]  
        self.seqs = [i[0] for i in self.data] 
        self.seqLens = [len(i[0]) for i in self.data]
        

        #Getting the mapping variables for class label and label id...
        if self.shared_lab2id is not None:
            self.lab2id = self.shared_lab2id
            self.id2lab = [None] * len(self.lab2id)
            for lab, idx in self.lab2id.items():
                self.id2lab[idx] = lab
        else:
            self.lab2id = {}
            self.id2lab = []
            for lab in self.rawLab:
                if lab not in self.lab2id:
                    self.lab2id[lab] = len(self.lab2id)
                    self.id2lab.append(lab)

        classNum = len(self.lab2id)
        self.labels = t.tensor([self.lab2id[i] for i in self.rawLab])
        
        # Get the mapping variables for kmers in seqs and kmers_id
        self.idSeq=[]
        for one_kmer_rna in tqdm(self.kmer_RNA):
            one_idseq=[]
            for one_kmer in one_kmer_rna:
                if ("O" not in one_kmer and "N" not in one_kmer):
                    one_idseq.append(self.kmers2id[one_kmer])
            self.idSeq.append(one_idseq)

        self.kmersNum = len(self.id2kmers)

        # Get the ids of RNAsequence and label
        self.labels = t.tensor([self.lab2id[i] for i in self.rawLab])
        self.params.set_n_classes(classNum)
        
        print(" ------- Dataset info ------- ")
        print(f'- Number of sequence is {seqNum}')
        print(f'- Number of class is {classNum}')
        print(f'- Sequence length: {min(self.seqLens)} - {max(self.seqLens)}')
        
        classStat = Counter(self.rawLab)
        print(classStat) 
        print(f'Number of Kmer is {self.kmersNum}  for k={self.k}')
        print(" ---------------------------- \n")
       
        # ===================================================================
        # Calls vectorize() to generate node features
        # =================================================================== 
        if self.params.node_embedding_method != "cMSEDN":
            self.vectorize(method=self.params.node_embedding_method, feaSize=self.params.d,\
                            loadCache=self.shared_lab2id is not None)

        # ===================================================================
        # Construct and save graph
        # ===================================================================        
        print('\n--- Construct and save the graph...---')
        print('- Model:' + self.params.gnn_model_name)
        print(f'- kmer: {self.params.k}')
        print(f'- node_embedding_method: {self.params.node_embedding_method}')

        self.graphs = []
        print('\n* Seq to graph...')
        for eachSeqIndex, eachSeq in enumerate(tqdm(self.idSeq)):
            eachSeqATGC=self.seqs[eachSeqIndex]
            # ------------------------------------------------------
            # remap kmer ids to local graph ids
            # ------------------------------------------------------
            newidSeq = []
            old2new = {}
            new2old = {}
            count = 0
            for eachid in eachSeq:
                if eachid not in old2new:
                    old2new[eachid] = count
                    new2old[count] = eachid
                    count += 1
                newidSeq.append(old2new[eachid])
            num_nodes = len(old2new)

            # =======================================
            # 1. ADJACENCY EDGES (ORIGINAL DE BRUIJN)
            # =======================================
            counter_uv = Counter(list(zip(newidSeq[:-1], newidSeq[1:])))
            adjacency_edges = list(counter_uv.keys())
            adjacency_weights = [float(v) for v in counter_uv.values()]
                   
            # ===================
            # 3. MERGE EDGE SETS
            # ===================
            all_edges = []
            all_weights = []

            ## adjacency edges
            for e, w in zip(adjacency_edges, adjacency_weights):
                all_edges.append(e)
                all_weights.append(w)

            # ==================
            # 4. BUILD GRAPH
            # ==================
            src = [e[0] for e in all_edges]
            dst = [e[1] for e in all_edges]

            graph = dgl.graph((src, dst), num_nodes=num_nodes)
            weight = t.FloatTensor(all_weights)
            norm = EdgeWeightNorm(norm='both')
            norm_weight = norm(graph, weight)
            graph.edata['weight'] = norm_weight

            # --------------------------------------
            # 4.1 Calc Contextual Node Features (cMSEDN): 
            # #---------------------------------------
            if self.params.node_embedding_method == "cMSEDN":
                L = len(eachSeqATGC)
                bandwidthList = [0.5, 1.5, 3, 4.5]
                # Compute MSEDN for the full sequence: (bases=4, len=L, S=4)
                eachSeqMSEDN = self.MSEDN_enc(eachSeqATGC, L, bandwidthList, fs=1)

                kmer_sum = {}
                kmer_count = {}
                feat_dim = 4 * self.k * 4
                kmerEmb_contextual = np.zeros((self.kmersNum, feat_dim), dtype=np.float32)

                for i in range(L - self.k + 1):
                    curr_kmer = eachSeqATGC[i:i+self.k]
                    slice_msedn = eachSeqMSEDN[:, i:i+self.k, :]  
                    if curr_kmer not in kmer_sum:
                        kmer_sum[curr_kmer] = slice_msedn.copy()  
                        kmer_count[curr_kmer] = 1
                    else:
                        kmer_sum[curr_kmer] += slice_msedn
                        kmer_count[curr_kmer] += 1

                for kmer, count in kmer_count.items():
                    if kmer in self.kmers2id:
                        avg_slice = kmer_sum[kmer] / count
                        avg_vec = avg_slice.T.flatten().astype(np.float32)  # flatten
                        kmerEmb_contextual[self.kmers2id[kmer]] = avg_vec

                # Extract features in local node order (old2new maps local -> global)
                node_features = kmerEmb_contextual[list(old2new.keys())]

            else:
                # Fallback to pre‑computed embeddings
                node_features = self.vector['embedding'][list(old2new.keys())]

            #---------------------------------------
            # 4.2 Assign to graph
            # #---------------------------------------
            graph.ndata['attr'] = t.tensor(node_features, dtype=t.float32)
            self.graphs.append(graph)

#=============================================================
    def kmers_gen(self, k):
        kmers2id, id2kmers = {}, []
        """Return a list of all possible k-mers over {A, T, G, C}."""
        id2kmers=[''.join(p) for p in product('ATGC', repeat=k)]
        for i in range(len(id2kmers)):
            kmers2id[id2kmers[i]]=i
        return id2kmers, kmers2id
    

    def __getitem__(self, idx):
        # Get a sample corresponding to it by idx
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        # Number of data samples
        return len(self.graphs)

    def save(self):
        # Save the processed data to `self.save_path`
        save_graphs(self.save_dir + ".bin", self.graphs, {'labels': self.labels})
        # Save additional information in the Python dictionary
        info_path = self.save_dir + "_info.pkl"
        info = {'kmers': self.k, 'kmers2id': self.kmers2id, 'id2kmers': self.id2kmers, 'lab2id': self.lab2id,
                'id2lab': self.id2lab}
        save_info(info_path, info)

    def load(self):
        # Import processed data from `self.save_path`
        self.graphs, label_dict = load_graphs(self.save_dir + ".bin")
        self.labels = label_dict['labels']
        info_path = self.save_dir + "_info.pkl"
        info = load_info(info_path)
        self.k, self.kmers2id, self.id2kmers, self.lab2id, self.id2lab = info['kmers'], info['kmers2id'], info[
            'id2kmers'], info['lab2id'], info['id2lab']

    def has_cache(self):
        # Check if there is processed data in `self.save_path`
        graph_path = self.save_dir + ".bin"
        info_path = self.save_dir + "_info.pkl"
        return os.path.exists(graph_path) and os.path.exists(info_path)
    

    def EDN_calc(self,idxPoints, bandwidth, kernel, xspace):
            if len(idxPoints) > 0:
                kde_model = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(idxPoints)
                EDN_sig = np.exp(kde_model.score_samples(xspace)) * len(idxPoints)
            else:
                EDN_sig = np.zeros((xspace.shape[0]))
            mx=EDN_sig.max()
            if mx>1:
                EDN_sig=EDN_sig/mx
            return EDN_sig


        #EDN encoder MultiScale 
    def MSEDN_enc(self, one_seq, seqLenLimit, bandwidthList, fs=1):
        if not isinstance(one_seq, str):
            raise Exception(f"!! Input sequence is {type(one_seq)}. it MUST be str type! ")
        kernel = "cosine"
        Nscale = len(bandwidthList)
        xspace = np.linspace(0, seqLenLimit-1, fs*seqLenLimit)[:, np.newaxis]
        seqEncoded = np.zeros((4, fs*seqLenLimit, Nscale), dtype='float32')
        one_seq = one_seq.upper().replace("U", "T")
        for iscale, bandwidth in enumerate(bandwidthList):
            sigs = []
            for base in ['A', 'T', 'G', 'C']:
                idx = np.array(list(self.findstr(one_seq, base)))[:, np.newaxis]
                sigs.append(self.EDN_calc(idx, bandwidth, kernel, xspace))
            seqEncoded[:, :, iscale] = np.array(sigs)
        return seqEncoded


    def findstr(self, str, ch):
        for i, ltr in enumerate(str):
            if ltr == ch:
                yield i

    # MultiScale EDN  function 
    def MSEDN_vecfun(self, one_seq):
        seqLenLimit=len(one_seq)  
        bandwidthList = [0.5, 1.5, 3, 4.5]
        fs=1
        fmat = self.MSEDN_enc(one_seq, seqLenLimit, bandwidthList, fs)
        fvec=fmat.T.flatten()
        return fvec


    def minmax_norm(self, vec):
        min_val = np.min(vec)
        max_val = np.max(vec)
        # Avoid division by zero if all values are equal
        if max_val - min_val == 0:
            return np.zeros_like(vec)  # or return vec if you prefer
        return (vec - min_val) / (max_val - min_val)
        

    def vectorize(self, method=None, feaSize=None, window=5, sg=1,
                  workers=8, loadCache=True):
        self.vector = {}
        print('\n***Executing vectorize function***')
        embeddings_file_path=f'checkpoints/Node_feature/kmerEmbeddings_{method}_k{self.k}_d{feaSize}.npy'
        if os.path.exists(embeddings_file_path) and loadCache:
            print(f'Found a cache file for embedding method: {method} in checkpoints/Node_feature/...')
            input_embedding_data = np.load(embeddings_file_path, allow_pickle=True).item()
            self.id2kmers = input_embedding_data['id2kmers']
            self.kmers2id = input_embedding_data['kmers2id']
            self.vector['embedding'] = input_embedding_data['embedding']
            print(f"Input embeddings shape: {input_embedding_data['embedding'].shape}")
            print(f'Loaded cache from '+embeddings_file_path+'!')
            return #if embedding file exiss, function returns here
        else:
            print(f'No cached checkpoints/Node_feature for {method}.')
            print('calc the Node features...')
        
        if method == 'onehot':
            print('--------------------------------------------------')
            print(f'<< Using {method} implementation with vector size: {feaSize} >>')                  
            # Create node-level one-hot features
            num_kmers = len(self.kmers2id)  
            embedding_matrix = np.eye(num_kmers, dtype=np.float32)
            self.vector['embedding'] = embedding_matrix
        
        elif method == 'sMSEDN':
            print('--------------------------------------------------')
            print(f'<< Using {method} implementation with vector size: {feaSize} >>')                  
            # Create embedding matrix for all k-mers
            embedding_matrix = np.zeros((self.kmersNum, feaSize), dtype=np.float32)
            # Get embedding for each k-mer
            for i in range(self.kmersNum): 
                fvec = self.MSEDN_vecfun(self.id2kmers[i])
                embedding_matrix[i] = fvec
            self.vector['embedding'] = embedding_matrix
        
        elif method == 'word2vec':
            print('--------------------------------------------------')
            print(f'<< Using {method} implementation with vector size: {feaSize} >>')                  
            doc = [i + ['<EOS>'] for i in self.kmer_RNA]
            self.id2kmers.append('<EOS>')
            self.kmers2id['<EOS>'] = self.kmersNum
            self.kmersNum+=1
            model = Word2Vec(doc, min_count=0, window=window, vector_size=feaSize, workers=workers, sg=sg, seed=self.params.seed)
            word2vec = np.zeros((self.kmersNum, feaSize), dtype=np.float32)
            for i in range(self.kmersNum):
                word2vec[i] = model.wv[self.id2kmers[i]]
            self.vector['embedding'] = word2vec
        
        # Combined DNABERT2 and NTv2_100m since they have identical code
        elif method in ['DNABERT2', 'NTv2']:
            print('--------------------------------------------------')
            print(f'<< Using {method} implementation with vector size: {feaSize} >>') 

            LLM_embedding_cache_path=f'checkpoints/kmerEmbeddings_'+\
                            f'{self.params.node_embedding_method}_k{self.params.k}.npy'
            embedding_data = np.load(LLM_embedding_cache_path, allow_pickle=True).item()
            kmerEmbedding_dic = embedding_data.get('embedding', {})
            print(f'loaded {method} cache file: size {len(kmerEmbedding_dic.keys())}')
            
            # Create embedding matrix for all k-mers
            embedding_matrix = np.zeros((self.kmersNum, self.params.llmDim), dtype=np.float32)
            # Get embedding for each k-mer
            for i, kmer in enumerate(self.id2kmers): 
                kmer_emb= kmerEmbedding_dic[kmer]
                kmer_emb=self.minmax_norm(kmer_emb)
                embedding_matrix[i]=kmer_emb
            from sklearn.decomposition import PCA
            pca = PCA(n_components=int(self.params.d))
            embedding_matrix = pca.fit_transform(embedding_matrix)
            self.vector['embedding'] = embedding_matrix
        
        else:
            raise ValueError(f"!! {method} embedding not defined.")

        print(f"Node Feature/Embedding generated. Matrix dim:{self.vector['embedding'].shape}")
        # Save k-mer embeding vectors
        embeding_data= {
            'id2kmers': self.id2kmers,
            'kmers2id': self.kmers2id,
            'embedding': self.vector['embedding']}
        np.save(embeddings_file_path, embeding_data, allow_pickle=True)
        print(f'Node Feature/Embedding Matrix saved!')
        return