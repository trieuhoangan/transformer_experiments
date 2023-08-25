import phonlp
dir = '/home/s1910446/bin_model/'
# phonlp.download(save_dir=dir)
import traceback
import sys
parsing_model = phonlp.load(save_dir=dir)
# tokens = parsing_model.annotate(output_type='conll',text="Xin đọc về bút hiệu Nguyễn Ái Quốc tại Nguyễn Ái Quốc (bút hiệu) .")
# print(tokens[3][0])
base_dir = "/home/s1910446/pascal/data/iwslten2vi/"
data_path = base_dir+ "corpus/"
train_file = data_path+"clear_train.tok.bpe.10000.vi"
valid_file = data_path+"clear_valid.tok.bpe.10000.vi"
test_file = data_path+"clear_test.tok.bpe.10000.vi"
tag_root_dir = base_dir+"tags_root"
output_train = tag_root_dir+"/train.vi"
output_valid = tag_root_dir+"/valid.vi"
output_test = tag_root_dir+"/test.vi"

def logerror(sent,tokens):
    error_log = tag_root_dir+"/error.txt"
    with open(error_log,"a",encoding='utf-8') as f:
        len_sent = len(sent.split(" "))
        len_tok = len(tokens)
        f.write(str(len_sent)+"  "+sent+"\n")
        f.write(str(len_tok)+"  "+str(tokens)+"\n")
def parsing(filename,output):
    lines = []
    with open(filename,'r',encoding="utf-8") as f:
        lines = f.readlines()
    tags = []
    for line in lines:
        sent = line.replace("\n","")
        sent = line.replace("@@","")
        sent = " ".join(sent.split())
        try:
            tokens = parsing_model.annotate(output_type='conll',text=sent)[3][0]
            sent_tag = []
            if len(tokens)!=len(sent.split(" ")):
                logerror(sent,tokens)
            for tok in tokens:
                # print(tok[0])
                tok_tag = int(tok[0])-1
                if tok_tag <0:
                    tok_tag = tokens.index(tok)
                sent_tag.append(tok_tag)
            tags.append(sent_tag)
        except:
            print(traceback.format_exc())
            sent_words = sent.split(' ')
            line_words = " ".join(sent.split()).split(' ')
            # if len(sent_words) != len(line_words):
            sent_tag = [i for i in range(len(line_words))]
            tags.append(sent_tag)
    with open(output,'w') as f:
        for sent_tag in tags:
            tag = [str(float(i)) for i in sent_tag]
            line = " ".join(tag)
            f.write(line+"\n")



if __name__ == "__main__":
    parsing(test_file,output_test)
    parsing(train_file,output_train)
    parsing(valid_file,output_valid)