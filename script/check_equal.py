base_dir = "/home/s1910446/pascal/data/iwslten2vi/"
data_path = base_dir+ "corpus/"
train_file = data_path+"clear_train.tok.bpe.10000.vi"
valid_file = data_path+"clear_valid.tok.bpe.10000.vi"
test_file = data_path+"clear_test.tok.bpe.10000.vi"
tag_root_dir = base_dir+"tags_root"
output_train = tag_root_dir+"/train.vi"
output_valid = tag_root_dir+"/valid.vi"
output_test = tag_root_dir+"/test.vi"
error_file = tag_root_dir+"/error2.txt"
def check_all_equal(source_file,tag_file):
    with open(source_file,'r',encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(tag_file,'r',encoding='utf-8') as f:
        tag_lines = f.readlines()    
    assert len(src_lines) == len(tag_lines)
    for i in range(0,len(src_lines)):
        src_line = src_lines[i].replace("\n","")
        src_line = " ".join(src_line.split())
        tag_line = tag_lines[i].replace("\n","")
        if len(tag_line.split(" "))!=len(src_line.split(" ")):
            with open(error_file,"a",encoding='utf-8') as f:
                len_sent = len(src_line.split(" "))
                len_tok = len(tag_line.split(" "))
                f.write(str(len_sent)+"  "+src_line+"\n")
                f.write(str(len_tok)+"  "+tag_line+"\n")
if __name__ =="__main__":
    check_all_equal(train_file,output_train)