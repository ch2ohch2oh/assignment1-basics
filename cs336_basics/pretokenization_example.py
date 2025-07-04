import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_tokens: list[bytes]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    for s in split_special_tokens:
        if not isinstance(s, bytes):
            raise ValueError("All special tokens must be bytestrings")

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            for tok in split_special_tokens:
                found_at = mini_chunk.find(tok)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
            if found_at != -1:
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

## Usage
if __name__ == "__main__":    
    num_processes = 16  # Example number of processes to use
    with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, ["<|endoftext|>".encode("utf-8")])
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            print(f"Chunk from {start} to {end}: {chunk[:100]}...")