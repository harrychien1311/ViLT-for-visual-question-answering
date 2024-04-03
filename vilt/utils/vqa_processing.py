import json
import os
import nltk
from nltk.stem.snowball import *
from tqdm import *
from collections import Counter, OrderedDict
import string
nltk.download('punkt')

def create_vocab(root_dir):
    #root_dir = os.environ['root']
    splits = ['train']

    ## question
    q_counter = Counter()
    n_sample = 0
    maxlen = 0
    for split in splits:
        dataset = json.load(open(root_dir + f'{split}.json'))
        for one_data in tqdm(dataset):
            n_sample += 1
            question = one_data['question']
            question = question.lower()
            tokens = nltk.word_tokenize(question)
            token_len = len(tokens)
            maxlen = max([maxlen,token_len])
            q_counter.update(tokens)
    print('number of sample = ' + str(n_sample))
    print('max len = ' + str(maxlen))
    q_word_counts = [x for x in q_counter.items()]
    q_word_counts.sort(key=lambda x: x[1], reverse=True)
    json.dump(q_word_counts, open('q_word_counts.json', "w"), indent=2)

    ### build vocabulary based on question
    vocab = [x[0] for x in q_word_counts if x[1] >= 0]
    unk_word = '<UNK>'
    vocab = [unk_word] + vocab
    vocab = OrderedDict(zip(vocab,range(len(vocab))))
    json.dump(vocab, open('word2vocab_id.json', 'w'), indent=2)

    ## answer
    ans_counter = Counter()
    for split in splits:
        dataset = json.load(open(root_dir + f'{split}.json'))
        for annotation in tqdm(dataset):
            for answer in annotation['answers']:
                answer = answer['answer'].lower()
                ans_counter.update([answer]) # don't forget the [], counter.update input a list
    ans_counts = [x for x in ans_counter.items()]
    ans_counts.sort(key=lambda x: x[1], reverse=True)
    json.dump(ans_counts, open('ans_counts.json', "w"), indent=2)

    ### build answer candidates
    output_num = 3000
    n_totoal = sum([x[1] for x in ans_counts])
    ans_counts = ans_counts[:output_num]
    n_cover = sum([x[1] for x in ans_counts])
    print(f"we keep top{len(ans_counts)}most answers")
    print ("coverage: %d/%d (%.4f)"%(n_cover, n_totoal, 1.0 * n_cover / n_totoal))
    ans_list = [x[0] for x in ans_counts]
    ans_dict = OrderedDict(zip(ans_list,range(len(ans_list))))
    json.dump(ans_dict, open('answer2answer_id.json', 'w'), indent=2)

def encode_sentence(sentence, vocab):
    unk_word = '<UNK>'
    tokens = nltk.word_tokenize(sentence.lower())
    tokens_id = [vocab.get(x, vocab[unk_word]) for x in tokens]
    return tokens_id

def data_formating(root_dir):
    #root_dir = os.environ['root']
    splits = ['train', 'val', 'test']
    all_imgs = os.listdir(root_dir+'train')+os.listdir(root_dir+'val')+os.listdir(root_dir+'test')
    all_imgs.sort()
    json.dump(all_imgs, open('all_imgs.json','w'))

    vocab = json.load(open('word2vocab_id.json'))
    answer2answer_id = json.load(open('answer2answer_id.json'))
    all_imgs = json.load(open('all_imgs.json'))
    image_id2image_feat_id = {img:idx for idx, img in enumerate(all_imgs)}

    for split in ['train', 'val']:
        ## load annotation file and question file
        dataset = json.load(open(root_dir + '%s.json'%split))

        ## encode QA
        image_id = []
        image_feat_id = [] # this is the line number of the image feat matrix
        questions = []
        answer_label = []
        answerable = []
        
        for one_data in tqdm(dataset):
            ans_counter = Counter([x['answer'] for x in one_data['answers']])
            ans = ans_counter.most_common(1)[0][0]
            a_label = answer2answer_id.get(ans, -1)
            if split[0] == 'train' and a_label == -1:
                continue
                
            i_id = one_data['image']
            i_feat_id = image_id2image_feat_id[str(i_id)]
            question = one_data['question'] # remove the '?' at the end
            
            image_id.append(i_id)
            image_feat_id.append(i_feat_id)
            questions.append(question)
            answer_label.append(a_label)
            answerable.append(one_data['answerable'])

            all_data = {'image_id': image_id, 'image_feat_id': image_feat_id, 'answerable':answerable, 
                    'question': questions, 'answer': answer_label}
        json_file = r'%s.json'%split
        json.dump(all_data, open(json_file, 'w'))
    dataset = json.load(open(root_dir + 'test.json'))

    ## encode QA
    image_id = []
    image_feat_id = [] # this is the line number of the image feat matrix
    questions = []
    # answer_label = []
    # answerable = []
    
    for one_data in tqdm(dataset):
        # ans_counter = Counter([x['answer'] for x in one_data['answers']])
        # ans = ans_counter.most_common(1)[0][0]
        # a_label = answer2answer_id.get(ans, -1)
        # # if split[0] == 'train' and a_label == -1:
        # #     continue
            
        i_id = one_data['image']
        i_feat_id = image_id2image_feat_id[str(i_id)]
        question = one_data['question'] # remove the '?' at the end
        
        image_id.append(i_id)
        image_feat_id.append(i_feat_id)
        questions.append(question)
        # answer_label.append(a_label)
        # answerable.append(one_data['answerable'])

        all_data = {'image_id': image_id, 'image_feat_id': image_feat_id, 
                'question': questions}
    json_file = 'test_encode_annotation.json'
    json.dump(all_data, open(json_file, 'w'))
if __name__=="__main__":
    create_vocab("/home/chien/ViLT/root/")
    data_formating("/home/chien/ViLT/root/")
    
