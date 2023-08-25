import phonlp
dir = '/home/s1910446/bin_model/'
# phonlp.download(save_dir=dir)
# parsing_model = phonlp.load(save_dir=dir)
# tokens = parsing_model.annotate(output_type='conll',text="Xin đọc về bút hiệu Nguyễn Ái Quốc tại Nguyễn Ái Quốc (bút hiệu) .")
# print(tokens[3][0])
base_dir = "/home/s1910446/pascal/data/iwsltdeen/corpus3"
# base_dir = "/home/s1910446/pascal/data/iwslten2vi"
tag_root_dir = base_dir+"/sub_tags"
train_file = tag_root_dir+"/train.en"
valid_file = tag_root_dir+"/valid.en"
test_file = tag_root_dir+"/test.en"

output_train = tag_root_dir+"/prime_indices_train.en"
output_valid = tag_root_dir+"/prime_indices_valid.en"
output_test = tag_root_dir+"/prime_indices_test.en"
dict_file = tag_root_dir+"/dict.en"

def logerror(sent,tokens):
    error_log = tag_root_dir+"/error.txt"
    with open(error_log,"a",encoding='utf-8') as f:
        len_sent = len(sent.split(" "))
        len_tok = len(tokens)
        f.write(str(len_sent)+"  "+sent+"\n")
        f.write(str(len_tok)+"  "+str(tokens)+"\n")
def parsing(tags_list,filename):
    lines = []

    with open(filename,'r',encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        tags = line.replace("\n","").split(" ")
        for tag in tags:
            if tag not in tags_list:
                tags_list[tag]=1
            else:
                tags_list[tag]=tags_list[tag]+1
    print(len(tags_list))
    return tags_list
def find_max(tags):
    max = 0 
    tag_max = ""
    for tag in tags:
        if tags[tag] > max:
            max = tags[tag]
            tag_max = tag
    return tag_max
def sort_tag(tags):
    newtag = {}
    while len(tags) >0:
        maxtag = find_max(tags)
        newtag[maxtag] = tags.pop(maxtag)
    return newtag
def log_dict(tags_list,file):
    newtags = sort_tag(tags_list)
    with open(file,'w') as f:
        for tag in newtags:
            f.write("{} : {}\n".format(tag,newtags[tag]))
def indexing_dict(tags):
    '''
    function to transform index of each relation into indices
    the index of each relation depends on its appearance frequent
    relation with higher appearance frequent will get higher index
    there are 2 exception: 
        "ROOT" will take the index that is double the number of relation
        "punct" will take index 0
    '''
    new_tag={}
    max_length=len(tags)
    i = max_length
    for tag in tags:
        new_tag[tag] = i
        i = i - 1
    new_tag["ROOT"] = max_length*2
    new_tag["root"] = max_length*2
    new_tag["punct"] = 0
    for tag in new_tag:
        if tag!="punct" and new_tag[tag]==0:
            new_tag[tag]=1
    return new_tag
def rewrite_tag(tag_id,filename,output):
    lines = []
    new_lines = []
    with open(filename,'r',encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        tags = line.replace("\n","").split(" ")
        new_tag = [str(tag_id[tag]) for tag in tags]

        new_lines.append(" ".join(new_tag))
    with open(output,'w') as f:
        for line in new_lines:
            f.write(line+"\n")
def is_prime(n):
    if n<2:
        return False
    if n<4:
        return True
    else:
        for i in range(2,int(n/2)):
            if n%i ==0:
                return False
        return True
def get_prime_list(n):
    prime_list = []
    i = 2
    while len(prime_list)<n:
        if is_prime(i):
            prime_list.append(i)
        i = i +1
    return prime_list
def indexing_primal_dict(tags):
    new_tag={}
    max_length=len(tags)
    prime_list = get_prime_list(max_length+2)
    print(prime_list)
    i = max_length
    for tag in tags:
        new_tag[tag] = prime_list[i]
        i = i - 1
    new_tag["ROOT"] = prime_list[max_length+1]
    new_tag["root"] = prime_list[max_length+1]
    new_tag["punct"] = prime_list[0]
    
    return new_tag
if __name__ == "__main__":
    tags_list = {}
    tags_list = parsing(tags_list,test_file)
    tags_list =parsing(tags_list,train_file)
    tags_list =parsing(tags_list,valid_file)
    # # log_dict(dict_file)
    # new_tags = indexing_dict(tags_list)
    new_tags = indexing_primal_dict(tags_list)
    # print(new_tags)
    rewrite_tag(new_tags,train_file,output_train)
    rewrite_tag(new_tags,test_file,output_test)
    rewrite_tag(new_tags,valid_file,output_valid)
    # print(get_prime_list(60))