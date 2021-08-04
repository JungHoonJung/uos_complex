import subprocess, os, gzip, h5py
import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import pkg_resources


####################### google drive download
import requests

def download_file_from_google_drive(id, destination, filesize=576435294):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
 
        return None
 
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
 
        with open(destination, "wb") as f:
            with tqdm(total = filesize, unit='B', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(CHUNK_SIZE)
 
    URL = "https://docs.google.com/uc?export=download"
 
    session = requests.Session()
 
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
 
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
 
    save_response_content(response, destination)    

githuburl = [
    'https://github.com/arbenson/ScHoLP-Data',
    #'https://github.com/manhtuando97/KDD-20-Hypergraph'
    ]

def unzipping(path):
    for file in os.listdir(path):
        if os.path.splitext(os.path.join(path, file))[1] =='.gz':
            with gzip.open(os.path.join(path, file), 'rb') as f:
                with open( os.path.splitext(os.path.join(path, file))[0],'w', encoding='utf8') as g:
                    g.write(f.read().decode('utf8'))


def get_github_dataset(path ='.', git_clone = True, unzip = False,  make_hdf5= ''):
    '''get hypergraph dataset of some papers through github.
    
    Parameters
    ------------
    path : <path_like_string>
        The original data will be downloaded into giben `path`
    git_clone : `bool`
        data download option. e.g. `git clone` 
    unzip : `bool`
        If `True`, some data which is compressed as gzip will be decompressed as txt into same directory.
    make_hdf5 : <path_like_string>
        If null string, this option will be ignored. Else, all dataset will be processed as hdf5 format.
    '''
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path,'ScHoLP-Data'), exist_ok=True)
    #os.makedirs(os.path.join(path,'KDD-20-Hypergraph'), exist_ok=True)
    if git_clone:
        res = subprocess.run(args = ['git','clone','https://github.com/arbenson/ScHoLP-Data', os.path.join(path, 'ScHoLP-Data')], 
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(res.stdout.decode())
        #res = subprocess.run(args = ['git','clone','https://github.com/manhtuando97/KDD-20-Hypergraph', os.path.join(path,'KDD-20-Hypergraph')], 
        #                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #print(res.stdout.decode())

    if unzip:
        dataset = os.path.join(path, 'ScHoLP-Data')
        for i in os.listdir(dataset):
            if os.path.isdir(os.path.join(dataset, i)):
                print('unzip : ', os.path.join(dataset, i))
                unzip(os.path.join(dataset, i))
    if make_hdf5:
        #KDD = os.path.join(path, 'KDD-20-Hypergraph','Datasets')
        SHLP = os.path.join(path, 'ScHoLP-Data')
        #for i in os.listdir(KDD):
        #    KDD_make_hdf5(os.path.join(KDD, i),make_hdf5)
        for i in os.listdir(SHLP):
            if i[0] == '.' :continue
            if os.path.isdir(os.path.join(SHLP, i)):
                SHLP_make_hdf5(os.path.join(SHLP, i), make_hdf5)


def KDD_make_hdf5(source_path = '.', target_path = 'data'):
    print('Making HDF5 File of ', f"'{os.path.splitext(os.path.basename(source_path))[0]}'", ' : ', end='')
    os.makedirs(target_path, exist_ok=True)
    data = {}
    nodeset = {}
    source =open(source_path, 'r')
    for line in source:
        nodes = line.rstrip().split()
        previous = data.get(len(nodes), [])
        dline = list(map(int, nodes))
        for node in dline:
            nodeset[node] = 1
        previous.append(dline)
        data[len(nodes)]  = previous
    
    hdf = h5py.File(os.path.join(target_path, os.path.splitext(os.path.basename(source_path))[0]+'.hdf5'), 'w')
    hdf.attrs['total_nodes'] = len(nodeset)
    datag = hdf.create_group('hyperedges')
    for k in data:
        datag.create_dataset(str(k),data=np.array(data[k]), compression='lzf')
    hdf.close()
    print('Done.')

def SHLP_make_hdf5(source_path = '.', target_path = 'data'):
    os.makedirs(target_path, exist_ok=True)
    basename = os.path.basename(source_path)
    print('Making HDF5 File of ', f"'{basename}'", ' : ', end='')
    hyperfiles = ['nverts','simplices', 'times']
    node_labels = os.path.join(source_path, basename+'-node-labels.txt.gz')
    exist_nodelabel = os.path.exists(node_labels)
    nodeset = {}
    if exist_nodelabel:
        with gzip.open(node_labels, 'rb') as f:
            #with open(os.path.join(target_path, basename+'-node-labels.txt'),'w',encoding='utf8') as g:
            txtline = f.read().decode('utf8')
            #g.write()
            txtlines = txtline.split('\n')
            maxline = 0
            labels = []
            for line in txtlines[:-1]:
                line = (' '.join(line.split()[1:])).encode('utf8')
                if len(line)>maxline:
                    maxline = len(line)
                labels.append(line)
            node_label = np.array(labels,dtype=f'S{maxline}')
        
    data = {}
    timedata = {}
    with gzip.open(os.path.join(source_path, basename+'-nverts.txt.gz'), 'rb') as nverts:
        with gzip.open(os.path.join(source_path, basename+'-simplices.txt.gz'), 'rb') as simplices:
            with gzip.open(os.path.join(source_path, basename+'-times.txt.gz'), 'rb') as times:
                verts = nverts.read().decode().split()
                simps = simplices.read().decode().split()
                time = times.read().decode().split()
                lineno = 0
                for i,t in zip(verts, time):
                    k = int(i)
                    previous = data.get(k, [])
                    hedge = list(map(int, simps[lineno:k+lineno]))
                    previous.append(hedge)
                    for node in hedge:
                        nodeset[node] = 1
                    data[k]  = previous
                    previous = timedata.get(k, [])
                    previous.append(int(t))
                    timedata[k]  = previous
                    lineno += k
    
    
    hdf = h5py.File(os.path.join(target_path, basename+'.hdf5'), 'w')
    if exist_nodelabel:
        hdf.attrs['total_nodes'] = node_label.shape[0]
        hdf.create_dataset('node_labels', data = node_label)
    else:
        hdf.attrs['total_nodes'] = len(nodeset)
    datag = hdf.create_group('hyperedges')
    for k in data:
        datag.create_dataset(str(k),data=np.array(data[k]), compression='lzf')
    timeg = hdf.create_group('time')
    for k in data:
        timeg.create_dataset(str(k),data=np.array(timedata[k]), compression='lzf')
    hdf.close()
    print('Done.')

def merge_hdf5(path, output_name = 'dataset.hdf5'):
    os.makedirs(os.path.split(os.path.join(path, output_name))[0], exist_ok=True)
    mhdf  = h5py.File(os.path.join(path, output_name), 'w')
    for file  in os.listdir(path):
        fname, ext = os.path.splitext(file)
        if ext == '.hdf5':
            with h5py.File(os.path.join(path, file),'r') as h:
                hdfg = mhdf.create_group(f'{fname}')   
                hdfg.attrs['total_nodes'] = h.attrs['total_nodes']
                if h.get('node_labels', False):
                    hdfg.create_dataset('node_labels', data = h['node_labels'][:], compression='lzf')
                hdfd = hdfg.create_group('hyperedges')
                for data in h['hyperedges']:
                    hdfd.create_dataset(data, data = h['hyperedges'][data][:], compression='lzf')
                if h.get('time', False):
                    hdft = hdfg.create_group('time')
                    for time in h['time']:
                        hdft.create_dataset(time, data = h['time'][time][:], compression='lzf')
    mhdf.close()


class HGData_: ## controls a single h5groups 
    def __init__(self, obj, readonly = False):
        self.hdf5 = obj
    
    def lengths(self):
        length = {}
        for i in self.hdf5:
            length[int(i)] = self.hdf5[i].shape[0]
        return length

    def __repr__(self):
        return f'<HyperGraph Data "{self.hdf5.name[1:]}" (mode {self.mode})>'

    def get_all_data(self):
        data = {}
        for i in self.hdf5:
            data[int(i)] = self.hdf5[i][:]
        return data

    def __iter__(self):
        for i in self.hdf5:
            yield i
    
    def __getitem__(self, value):
        if isinstance(value, int):
            return self.hdf5[str(value)][:]

        elif isinstance(value, tuple):
            data =[]
            for i in value:
                data.append(self.hdf5[str(i)][:])
            return data
        else:
            return self.hdf5[value]


class HGData: ## data + time is consist of whole dataset
    def __init__(self, obj, readonly = False):
        self.mode = 'r' if readonly else 'r+'
        self._standalone = False
        if isinstance(obj, h5py.Group) or isinstance(obj, h5py.File):
            self.hdf5 = obj
        else:
            self.hdf5 = h5py.File(obj, self.mode)
            self._standalone = True

        self.name = self.hdf5.name[1:]
        self.data = HGData_(self.hdf5['hyperedges'])
        self.has_time = False
        if self.hdf5.get('time',False):
            self.has_time = True
            self.time = HGData_(self.hdf5['time'])
        self._node_labels = {}
        self.total_nodes = self.hdf5.attrs['total_nodes']
    
    @property
    def nodes(self):
        return np.arange(self.total_nodes)

    def lengths(self):
        return self.data.lengths()

    def __repr__(self):
        if self._standalone:
            return f'<HyperGraph Data hdf5 file "{os.path.basename(self.hdf5.filename)}" (mode {self.mode})>'
        else:
            return f'<HyperGraph Data "{self.hdf5.name[1:]}" (mode {self.mode})>'

    @property
    def node_labels(self):
        return self._node_labels

    @node_labels.getter
    def node_labels(self):
        if self._node_labels == {}:
            if self.hdf5.get('node_labels', False):
                self._node_labels = {i : j.decode('utf8') for i,j in enumerate(self.hdf5['node_labels'][:])}
            else:
                self._node_labels = None
        return self._node_labels

    def get_all_data(self):
        return self.data.get_all_data() if not self.has_time else self.data.get_all_data(), self.time.get_all_data() 
    
    @property
    def network(self):
        edges = self.data[2]
        self._Graph = nx.from_edgelist(edges)
        if self.node_labels is not None:
            self._Graph.add_nodes_from([(node, {'label' : self.node_labels[node]}) for node in self.node_labels])
        return self._Graph

    def __iter__(self):
        for i in self.data:
            yield i

    def __call__(self, data = False, time = False):
        for node in self.data:
            res = int(node)
            if data or time:
                res = [int(node)]
            if data:
                res.append(self.data[node][:])
            if time:
                res.append(self.time[node][:])
            yield res

    def __getitem__(self, value):
        if isinstance(value, int) or isinstance(value, tuple):
            return self.data[value]
        else:
            return self.hdf5[value]

    def get_nerve_complex(self):    
        """Get nerve complex

        Returns:
            [type]: [description]
        """        
        k = [int(ks) for ks in self]
        k.sort(reverse= True)
        facets = {}
        simps = set()
        for i in tqdm(k):
            for j in self[i]:
                simp = tuple(sorted(j))
                simpset = set(simp)
                if simp in simps:  #overlap check
                    continue
                simps.add(simp)

                facet  = False
                nfacet  = False
                for node in simp: 
                    faces = facets.get(node, [])
                    if not faces or facet:
                        faces.append(simpset)
                        facet = True
                        facets[node] = faces
                    else:
                        for face in faces:
                            if len(simpset - face) == 0:
                                nfacet = True
                            break
                    if nfacet:
                        break
        print(self.name)
        return facets

    def get_clique_complex(self):    
        facets = {}
        simps = set()
        for j in tqdm(nx.find_cliques(self.network)):
            simp = tuple(sorted(j))
            simpset = set(simp)
            if simp in simps:  #overlap check
                continue
            simps.add(simp)

            facet  = False
            nfacet  = False
            for node in simp: 
                faces = facets.get(node, [])
                if not faces or facet:
                    faces.append(simpset)
                    facet = True
                    facets[node] = faces
                else:
                    for face in faces:
                        if len(simpset - face) == 0:
                            nfacet = True
                        break
                if nfacet:
                    break
        return facets

class HGDataset: 
    def __init__(self, file, readonly = False):
        self.mode = 'r' if readonly else 'r+'
        self.hdf5 = h5py.File(file, self.mode)
        for datum in self.hdf5:
            setattr(self, datum.replace('-','_'), HGData(self.hdf5[datum], self.mode))
    
    def __repr__(self):
        return f'<HyperGraph Dataset hdf5 file "{os.path.basename(self.hdf5.filename)}" (mode {self.mode})>'

    def ls(self):
        for groups in self.hdf5:
            print(groups.replace('-','_'))
        return

    def __iter__(self):
        for datum in self.hdf5:
            yield getattr(self, datum.replace('-','_'))

    @classmethod
    def from_github(cls, path, output_name = "HGDatasets.hdf5", readonly = True):
        get_github_dataset(path, True, False, path)
        merge_hdf5(path, output_name)
        return cls(os.path.join(path, output_name), readonly)

    @classmethod
    def from_google_drive(cls, path = '.', readonly = True):
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, 'HGDataset.hdf5')
        if not os.path.exists(os.path.join(path, 'HGDataset.hdf5')):
            download_file_from_google_drive('1DDGvonWKGQ8LIB22lKp2z511pymcXktr', filename)
        return cls(filename, readonly)
    
    def __getitem__(self, value):
        data_list = [data.replace('-','_') for data in self.hdf5]
        for data in data_list:
            if data.startswith(value):
                print(data, " is found.")
                return getattr(self, data)
        raise ValueError('No dataset matched.')
                

    def close(self):
        self.hdf5.close()
        return 

    def __del__(self):
        self.hdf5.close()
        return



################   static method #####################

def from_github(path, output_name = "HGDatasets.hdf5", readonly = True):
    get_github_dataset(path, True, False, path)
    merge_hdf5(path, output_name)
    return HGDataset(os.path.join(path, output_name), readonly)

def from_google_drive(path = '.', readonly = True):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, 'HGDataset.hdf5')
    if not os.path.exists(os.path.join(path, 'HGDataset.hdf5')):
        download_file_from_google_drive('1DDGvonWKGQ8LIB22lKp2z511pymcXktr', filename)
    return HGDataset(filename, readonly)

def get_pacs():
    return HGData(pkg_resources.resource_filename('uos_complex', 'data/APS_pacs/pacs.hdf5'))