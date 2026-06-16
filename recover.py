import h5py, hdf5plugin
with h5py.File('I(q)@L=50_test.h5', 'r') as f:
    grp = f['(Na,Co,Ag,Pb,Mo,Fe)_monoatomic_clusters']
    k = list(grp.keys())[0]
    mol = grp[k]
    print(k, {ds: mol[ds].shape for ds in mol.keys()})
