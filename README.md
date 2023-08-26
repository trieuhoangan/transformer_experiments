- **requirements**
    - fairseq 1.10
    - pytorch
    - Standford CoreNLP parser (for LISA and Sub_tag)

- **preprocessing data:**
   - vi-en: 
       1. get dependency parsing by phonlp model by running: 
            - python3 script/prepare_data.py
            - python3 script/prepare_sub_tag.py
       2. parse style of relation of dependency into indices:
            - python3 script/transform_sub_tags.py
   - en-vi/en-de/de-en:
       1. get dependency parsing by phonlp model by running: 
            - bash script/prepare_tag_root.sh ( change paths in file ) 
            - python3 script/corenlp_get_sub_tags.py ( change paths in file ) 
       2. parse style of relation of dependency into indices:
            - python script/transform_sub_tags.py ( change paths in file ) 
- **prepare data before training**
    - bash binarize_data.sh
    - bash binarize_tag_root.sh (for LISA and subtag)
    - bash binarize_sub_tag.sh (for subtag)
- **script to run transformer model (change the correct path to run):**
    - ```bash run_transformer.sh```
    - for the experiment which does not have this file. Please modify from the file in transformer folder Or can run 
    - bash new_train.sh
- **script to eval transformer model (change correct path to run):**
    - bash new_eval.sh



