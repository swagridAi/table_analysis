import h5py
import numpy as np
import os

class DiskBasedCooccurrenceMatrix:
    """Store large co-occurrence matrices on disk using HDF5."""
    
    def __init__(self, filepath, mode='w'):
        """
        Initialize a disk-based co-occurrence matrix.
        
        Args:
            filepath (str): Path to HDF5 file
            mode (str): File mode ('w' for write, 'r' for read)
        """
        self.filepath = filepath
        self.file = h5py.File(filepath, mode)
        
        if mode == 'w':
            # Create datasets
            self.file.create_group('metadata')
        
        # Load metadata if reading
        if mode == 'r':
            self.all_elements = list(self.file['metadata']['elements'][()])
            self.elem_to_idx = {elem: i for i, elem in enumerate(self.all_elements)}
    
    def store_matrix(self, co_occurrence):
        """
        Store co-occurrence data on disk.
        
        Args:
            co_occurrence (dict): Dictionary mapping (elem1, elem2) tuples to counts
        """
        # Get all unique elements
        all_elements = sorted(set(elem for pair in co_occurrence.keys() for elem in pair))
        self.all_elements = all_elements
        self.elem_to_idx = {elem: i for i, elem in enumerate(all_elements)}
        
        # Store elements
        dt = h5py.special_dtype(vlen=str)
        elements_dataset = self.file.create_dataset('metadata/elements', 
                                                  (len(all_elements),), 
                                                  dtype=dt)
        for i, elem in enumerate(all_elements):
            elements_dataset[i] = elem
        
        # Create sparse representation
        rows = []
        cols = []
        data = []
        
        for (elem1, elem2), count in co_occurrence.items():
            i = self.elem_to_idx[elem1]
            j = self.elem_to_idx[elem2]
            rows.append(i)
            cols.append(j)
            data.append(count)
            # Also add symmetric entry
            rows.append(j)
            cols.append(i)
            data.append(count)
        
        # Store sparse representation
        self.file.create_dataset('matrix/rows', data=np.array(rows, dtype=np.int32))
        self.file.create_dataset('matrix/cols', data=np.array(cols, dtype=np.int32))
        self.file.create_dataset('matrix/data', data=np.array(data, dtype=np.float32))
        self.file.attrs['matrix_size'] = len(all_elements)
    
    def get_cooccurrence(self, elem1, elem2):
        """Get co-occurrence count for a pair of elements."""
        if elem1 not in self.elem_to_idx or elem2 not in self.elem_to_idx:
            return 0
        
        i = self.elem_to_idx[elem1]
        j = self.elem_to_idx[elem2]
        
        # Look up in the sparse matrix
        rows = self.file['matrix/rows'][()]
        cols = self.file['matrix/cols'][()]
        data = self.file['matrix/data'][()]
        
        for idx, (r, c) in enumerate(zip(rows, cols)):
            if (r == i and c == j) or (r == j and c == i):
                return data[idx]
        
        return 0
    
    def close(self):
        """Close the HDF5 file."""
        self.file.close()