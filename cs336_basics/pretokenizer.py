from collections import defaultdict
import os
import time
import regex as re
from concurrent.futures import ProcessPoolExecutor


def find_chunk_boundaries(file_path: str, desired_num_chunks: int, special_tokens: list[bytes]) -> list[int]:
    """
    Chunk the file into parts that starts with `split_special_tokens`.
    May return fewer chunks if the boundaries end up overlapping.
    """
    for s in special_tokens:
        if not isinstance(s, bytes):
            raise ValueError("All special tokens must be bytestrings")

    try:
        file_size = os.path.getsize(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_size == 0:
        return [0]

    special_token_pattern = re.compile(b"|".join(re.escape(tok) for tok in special_tokens))

    # Initial guesses for chunk boundary locations, uniformly spaced.
    # The first boundary is always 0 and the last is file_size.
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # Use a 'with' statement to ensure the file is properly closed.
    with open(file_path, "rb") as f:
        # Adjust each boundary (except the first and last) to align with a special token.
        for i in range(1, len(chunk_boundaries) - 1):
            search_pos = chunk_boundaries[i]

            # Search forward from the initial guess until a token is found or EOF.
            while search_pos < file_size:
                f.seek(search_pos)
                # Read a reasonably sized buffer to search for the token.
                buffer = f.read(mini_chunk_size)
                if not buffer:  # Reached end of file
                    chunk_boundaries[i] = file_size
                    del chunk_boundaries[i + 1 :]  # Remove any remaining boundaries
                    break

                match = special_token_pattern.search(buffer)
                if match:
                    # Found a token, align boundary and move to the next one.
                    chunk_boundaries[i] = search_pos + match.start()
                    break

                # No token in this buffer, advance search position.
                search_pos += len(buffer)

    # De-duplicate and sort boundaries. This handles cases where chunking
    # logic might make two boundaries identical (e.g., on small files).
    return sorted(set(chunk_boundaries))


def print_chunk_boundaries_preview(file_path: str, boundaries: list[int]) -> None:
    """
    Print a preview of the chunk boundaries in the file.
    """
    with open(file_path, "rb") as file:
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            file.seek(start)
            chunk = file.read(end - start)
            print(f"Chunk {i}: {start} to {end}, size: {len(chunk)} bytes")
            print(chunk[:100].decode("utf-8", errors="ignore"), "...\n")  # Print first 100 chars


def pretokenize_chunk(
    file_path: str, boundaries: tuple[int, int], special_tokens: list[bytes], mini_chunk_size: int = 4096
) -> dict[bytes, int]:
    """
    Tokenize a chunk of the file and return a dictionary of token counts. Discards special tokens.
    """
    vocab = defaultdict(int)
    start, end = boundaries
    pretoken_pattern = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    special_token_pattern = re.compile(b"|".join(re.escape(tok) for tok in special_tokens))

    with open(file_path, "rb") as file:
        file.seek(start)
        text_bytes = file.read(end - start)

    for stripped_text in special_token_pattern.split(text_bytes):
        # Tokenize each clean segment without special tokens
        for token_match in pretoken_pattern.finditer(stripped_text):
            vocab[token_match.group(0)] += 1
    return vocab


def print_vocab(vocab: dict[bytes, int], topn: int = 20) -> None:
    """
    Print the vocabulary counts in a readable format.
    """
    print("Vocabulary size:", len(vocab))
    zero_count_tokens = [token for token, count in vocab.items() if count == 0]
    print("Tokens with zero count:", len(zero_count_tokens))
    print(f"Top {topn} tokens:")
    for token, count in sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:topn]:
        print(f"{token} => {count}")


def _pretokenize(file_path: str, special_tokens: list[bytes], num_processes: int = 1) -> dict[bytes, int]:
    """
    Tokenize a chunk of the file and return a dictionary of token counts. Discards special tokens. Naive version.
    """
    vocab = defaultdict(int)
    pretoken_pattern = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    special_token_pattern = re.compile(b"|".join([re.escape(tok) for tok in special_tokens]))

    with open(file_path, "rb") as file:
        text_bytes = file.read()

    # !!! It is NOT correct to use `re.sub` here, because the tokens before and after special tokens may be joined together **incorrectly**.
    for stripped_text in special_token_pattern.split(text_bytes):
        # Tokenize each clean segment without special tokens
        for token_match in pretoken_pattern.finditer(stripped_text):
            vocab[token_match.group(0)] += 1
    return vocab


def pretokenize(file_path: str, special_tokens: list[bytes], num_processes: int = -1) -> dict[bytes, int]:
    """
    Find the vocabulary from a file using the specified special tokens.
    Uses multiprocessing for parallel chunk processing.
    """
    if num_processes < 1:
        num_processes = os.cpu_count() or 1

    boundaries = find_chunk_boundaries(file_path, num_processes, special_tokens)

    args = [(file_path, (boundaries[i], boundaries[i + 1]), special_tokens) for i in range(len(boundaries) - 1)]

    vocab = defaultdict(int)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        result = list(executor.map(_wrapper_find_chunk_vocab, args))

    for chunk_vocab in result:
        for token, count in chunk_vocab.items():
            vocab[token] += count
    return vocab


# Wrapper to make multiprocessing safe
def _wrapper_find_chunk_vocab(args):
    return pretokenize_chunk(*args)


## Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-tokenize a file into vocabulary counts.")
    parser.add_argument(
        "-n", "--num_processes", type=int, default=1, help="Number of processes to use for tokenization."
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = [b"<|endoftext|>"]
    chunk_boundaries = find_chunk_boundaries(file_path, num_processes, special_tokens)
    # print_chunk_boundaries_preview(file_path, chunk_boundaries)
    start = time.time()
    vocab = pretokenize(file_path, special_tokens, num_processes=num_processes)
    end = time.time()
    print(f"Tokenization completed in {end - start:.2f} seconds using {num_processes} processes.")
    print_vocab(vocab, topn=10)
