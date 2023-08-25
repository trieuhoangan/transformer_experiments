import argparse
import numpy as np
# 2 strategies:
# 1.) accept multiple appearance of a word in a sentence
# 2.) consider multiple appearance of a word in 1 sentence as 1
def load_dictionary_from_file(dictionary_file):
    dictionary = {}
    try:
        with open(dictionary_file,'r',encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            true_line = line.replace("\n","")
            key = true_line.split("<!!!!!!>")[0]
            value = true_line.split("<!!!!!!>")[1]
            dictionary[key] = int(value)
        return dictionary
    except:
        return {}
def print_dictionary_to_file(output_file,dictionary):
    with open(output_file,'w',encoding="utf-8") as f:
        for pair in dictionary:
            line = "{}<!!!!!!>{}\n".format(pair,dictionary[pair])
            f.write(line)
def count_appearance(words,dictionary):
    num_word = len(words)
    for i in range(num_word):
        first_word = words[i]
        for j in range(i+1,num_word):
            second_word = words[j]
            pair1,pair2,time = check_pair_in_dict(first_word,second_word,dictionary)
            dictionary[pair1] = time+1
            dictionary[pair2] = time+1
        return dictionary
def extract_pair_from_input_file(input_file,dictionary,contain_doc_dictionary):
    with open(input_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        real_line = line.replace("\n","")
        true_line = " ".join(real_line.split())
        raw_words = true_line.split(" ")
        words = []
        unique_words = []
        for word in raw_words:
                if word not in words:
                    unique_words.append(word)
        words = raw_words
        dictionary = count_appearance(words,dictionary)
        contain_doc_dictionary = count_appearance(unique_words,contain_doc_dictionary)
        
    return dictionary,contain_doc_dictionary,len(lines)
        
def check_pair_in_dict(first_word,second_word,dictionary):
    pair1 = "{}$${}".format(first_word,second_word)
    pair2 = "{}$${}".format(second_word,first_word)
    if pair1 not in dictionary and pair2 not in dictionary:
        return pair1,pair2,0
    else:
        # if pair1 in dictionary:
        #     return pair1,dictionary[pair1]
        # if pair2 in dictionary:
        #     return pair2,dictionary[pair2]
        return pair1,pair2,dictionary[pair2]
def apply_td_idf(dictionary,contain_doc_dictionary,total_doc):
    for pair in dictionary:
        TF = dictionary[pair]
        IDF = np.log((total_doc+1)/(contain_doc_dictionary[pair]+1)) +1
        TF_IDF = TF * IDF 
        dictionary[pair] = TF_IDF
    return dictionary
def normalize_dictionary(dictionary):
    sum=0
    for pair in dictionary:
        sum = sum + dictionary[pair]
    # sum = sum/2
    for pair in dictionary:
        # dictionary[pair] = 2*dictionary[pair]/len(dictionary.keys())
        dictionary[pair] = 2*dictionary[pair]/sum
    return dictionary
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some args.')
    parser.add_argument('--train-file', type=str, default=None,
                    help='input source text file path')
    parser.add_argument('--test-file', type=str, default=None,
                    help='input source text file path')
    parser.add_argument('--valid-file', type=str, default=None,
                    help='input source text file path')
    parser.add_argument('--output-file', type=str, default=None,
                    help='ouput dictionary file path')
    parser.add_argument('--output-counter', type=str, default=None,
                    help='ouput dictionary file path')
    args = parser.parse_args()
    dictionary = load_dictionary_from_file(args.output_file)
    contain_doc_dictionary = load_dictionary_from_file(args.output_counter)
    total_doc = 0
    if args.train_file is not None:
        dictionary,contain_doc_dictionary,num_doc = extract_pair_from_input_file(args.train_file,dictionary,contain_doc_dictionary)
        total_doc = total_doc +num_doc
    if args.test_file is not None:
        dictionary,contain_doc_dictionary,num_doc = extract_pair_from_input_file(args.test_file,dictionary,contain_doc_dictionary)
        total_doc = total_doc +num_doc
    if args.valid_file is not None:
        dictionary,contain_doc_dictionary,num_doc = extract_pair_from_input_file(args.valid_file,dictionary,contain_doc_dictionary)
        total_doc = total_doc +num_doc

    dictionary = normalize_dictionary(dictionary)
    dictionary = apply_td_idf(dictionary,contain_doc_dictionary,total_doc)
    print_dictionary_to_file(args.output_file,dictionary)