import sys,os
import pandas as pd
import numpy as np

def split(f_in, f_type, in_dir):
    df = pd.read_hdf(in_dir+"/"+f_in, key="table")
    num_chunks = int(len(df)/5000+1)
    chunk_size = 5000
    print(f"creating {num_chunks} chunks")
    # Split the DataFrame and save each chunk to a separate HDF5 file
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        # print(i)
        chunk = df.iloc[start_idx:end_idx]
        chunk.to_hdf(f'{in_dir}/{f_type}_chunk_{i+1}.h5', key='table', mode='w')
    print(f"done!")
        
def main(args):

        f_in = str(sys.argv[1])
        f_type = str(sys.argv[2])
        in_dir = str(sys.argv[3])
        split(f_in, f_type, in_dir)
        return 

if __name__ == "__main__":
    main(sys.argv)   
