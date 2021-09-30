###############################################################################
# DataLoader.py
# J. Steiner

#%%## LOAD DEPENDENCIES
# imports the pytorch library, pretty much just used so we can convert lists
# to tensors pre-emptively
import torch

#%%## DATA LOADING FUNCTIONS
# this function is for loading data from a single text file, where the sequence
# is each line that comes in 
def load(file_path):

    # file_path - the file path we are loading the text from

    # a list that will store the data set by the end of the function
    data = []

    # initializes the word2Idx dictionary, and adds a padding token first
    word2Idx = dict()
    word2Idx['<PAD>'] = 0
    # initializes a idx2Word dictionary so that way we can easily convert back
    idx2Word = dict()
    idx2Word[0] = '<PAD>'

    # loads in the file from the parameter file path
    in_File = open(file_path)

    # loops through each line in the loaded in file
    for line in in_File:

        # initializes an empty list that will store the word idx's for each
        # line
        idx_list = []

        # strips off the new line character for each line
        line = line.strip()

        # if the line has anything left after the new line token was removed
        # meaning it's not just a carriage return
        if len(line) > 0:
            
            # puts spaces around punctuation
            line = line.replace("'", " ' ")
            line = line.replace('"', ' " ')
            line = line.replace('.', ' . ')
            line = line.replace(',', ' , ')
            line = line.replace('!', ' ! ')
            line = line.replace('?', ' ? ')
            
            # stores the line, split across spaces, with a start of sequence
            # and an end of sequence token a the start and end of each sequence
            # respectively
            word_list = line.split()

            # for each word in the word list
            for word in word_list[:16]:
                
                # if the word is not in the word index
                if word not in word2Idx:
                    # adds the word and index to the word2Idx and idx2Word
                    # dictionaries
                    word2Idx[word] = len(word2Idx)
                    idx2Word[len(word2Idx)-1] = word

                # appends the word index to the index list
                idx_list.append(word2Idx[word])

            # appends the idx list to the data list
            data.append(idx_list)

    # finds the longest sequence length in the data (used to pad all others to 
    # that longest length)
    max_seq_len = 16#max([len(seq) for seq in data])

    # stores the padded sequence and has -1 as a dummy label
    data = [ (torch.tensor(seq + [0] * (max_seq_len - len(seq))), -1) for seq in data]

    # returns the data list, word2Idx dict and idx2Word dict
    return data, word2Idx, idx2Word

# this function is for loading data from a category of writing prompts
def load_prompts(source_file_path, target_file_path):

    # source_file - the file path we are loading the text from for the prompts 
    # target_file - the file path we are loading the text from for the stories

    # a list that will store the data set by the end of the function
    data = []

    # initializes the word2Idx dictionary, and adds a padding token first
    word2Idx = dict()
    word2Idx['<PAD>'] = 0
    # initializes a idx2Word dictionary so that way we can easily convert back
    idx2Word = dict()
    idx2Word[0] = '<PAD>'

    # loads in the file from the parameter file path
    infile_source = open(source_file_path, encoding='utf-8')
    infile_target = open(target_file_path, encoding='utf-8')
    
    seqs_source = []
    seqs_target = []

    max_length = 500
    ctr = 1

    # loops through each line in the loaded in file
    for source_line, target_line in zip(infile_source, infile_target):

        if ctr > max_length:
            break

        # initializes an empty list that will store the word idx's for each
        # line
        idx_list_source = []
        idx_list_target = []

        # strips off the new line character for each line
        source_line = source_line.strip()
        target_line = target_line.strip()
            
        # puts spaces around punctuation
        source_line = source_line.replace('"', ' " ')
        source_line = source_line.replace('.', ' . ')
        source_line = source_line.replace(',', ' , ')
        source_line = source_line.replace('!', ' ! ')
        source_line = source_line.replace('?', ' ? ')
        target_line = target_line.replace('"', ' " ')
        target_line = target_line.replace('.', ' . ')
        target_line = target_line.replace(',', ' , ')
        target_line = target_line.replace('!', ' ! ')
        target_line = target_line.replace('?', ' ? ')
        
        # stores the line, split across spaces, with a start of sequence
        # and an end of sequence token a the start and end of each sequence
        # respectively
        word_list_source = source_line.split()
        word_list_target = target_line.split()

        # for each word in the word list
        for word in word_list_source[:64]:
            
            # if the word is not in the word index
            if word not in word2Idx:
                # adds the word and index to the word2Idx and idx2Word
                # dictionaries
                word2Idx[word] = len(word2Idx)
                idx2Word[len(word2Idx)-1] = word

            # appends the word index to the index list
            idx_list_source.append(word2Idx[word])

        seqs_source.append(idx_list_source)

        # # for each word in the word list
        # for word in word_list_target:
            
        #     # if the word is not in the word index
        #     if word not in word2Idx:
        #         # adds the word and index to the word2Idx and idx2Word
        #         # dictionaries
        #         word2Idx[word] = len(word2Idx)
        #         idx2Word[len(word2Idx)-1] = word

        #     # appends the word index to the index list
        #     idx_list_target.append(word2Idx[word])

        seqs_target.append(idx_list_target)

        ctr += 1

    infile_source.close()
    infile_target.close()

    # finds the longest sequence length in the data (used to pad all others to 
    # that longest length)
    max_seq_len_source = max([len(seq) for seq in seqs_source])
    max_seq_len_target = max([len(seq) for seq in seqs_target])

    # stores the padded sequence and has -1 as a dummy label
    data = [ (torch.tensor(seq_source + [0] * (max_seq_len_source - len(seq_source))),\
              torch.tensor(seq_target + [0] * (max_seq_len_target - len(seq_target)))) \
              for seq_source, seq_target in zip(seqs_source, seqs_target)]

    # returns the data list, word2Idx dict and idx2Word dict
    return data, word2Idx, idx2Word