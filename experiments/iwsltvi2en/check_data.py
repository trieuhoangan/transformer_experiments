base_dir = "/home/s1910446/pascal/data/iwslten2vi/"
data_path = base_dir+ "corpus/"
train_file = data_path+"clear_train.tok.bpe.10000.vi"
valid_file = data_path+"clear_valid.tok.bpe.10000.vi"
test_file = data_path+"clear_test.tok.bpe.10000.vi"
tag_root_dir = base_dir+"tags_root"
output_train = tag_root_dir+"/train.vi"
output_valid = tag_root_dir+"/valid.vi"
output_test = tag_root_dir+"/test.vi"

if __name__=="__main__":
    with open(train_file,'r',encoding="utf-8") as f:
        src_lines = f.readlines()
    with open(output_train,'r',encoding="utf-8") as f:
        tag_lines = f.readlines()
    for i in range(len(src_lines)):
        src_line = src_lines[i]
        tag_line = tag_lines[i]
        if len(src_line.split(" "))!=len(tag_line.split(" ")):
            print(src_line)