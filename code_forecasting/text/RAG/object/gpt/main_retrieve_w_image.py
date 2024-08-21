import os
import json
import argparse
import json
import os

import random
from tqdm import tqdm
import pandas as pd
import re
from openai import OpenAI
from llama_index.core import ServiceContext, Settings
from llama_index.llms.openai import OpenAI as llama_index_OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex,Document
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

import nest_asyncio
nest_asyncio.apply()
#from llama_index import StorageContext, load_index_from_storage, Document
import os
from time import sleep

ROOT_PATH = None

def extract_words(value):
    words_pattern = re.compile(r'\b\w{2,}\b')
    words = words_pattern.findall(value)
    if len(words) == 0:
        return None
    else:
        result_string = ' '.join(words)
        return result_string

def parse_outputs(outputs):
    match = re.match(r'([A-Z])\b', outputs)
    if match:
        parse_result = [match.group(0), extract_words(outputs)]
    else:
        parse_result = [None, outputs]
    return parse_result

def generate_prompt_rules(task):
    if task == 'reasoning':
        rules = [
        "1. The atomic event is the basic unit describing a specific event, typically presented in the form of a quadruple (S, R, O, T), where S represents the subject, R represents the relation, O represents the object, and T represents the relative time.\n",
        "2. When formulating the ultimate prediction, the preeminent factor to be meticulously weighed and scrutinized is the [Key Events]. Complementing this paramount consideration is the [Related events], which, though ancillary in nature, serves as a valuable adjunct, furnishing pertinent contextual details and auxiliary insights to fortify the predictive analysis.\n"
        "3. Given a query of (S, R, T) in the future and the list of historical events until t, event forecasting aims to predict the missing object.",
        ]
        prompt_rules = "You are an assistant to perform event forecasting with the following rules:\n" + ''.join(rules)
    
    return prompt_rules

def generate_reasoning_input(row, complement_summary_k, complement_summary_r):
    # complement_summary = sorted(complement_summary, key=lambda item: item[0],reverse=True)
    subject = row['Subject']
    relation = row['Relation']
    time = row['timid']
    correct_object = row['Object']
    candidates = eval(row['Candidates'])
    options = candidates + [correct_object]
    random.shuffle(options)
    option_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapped_options = [[option_letters[i],option]  for i, option in enumerate(options)]

    news_text_k = []
    for time in complement_summary_k.keys():
        for md5 in complement_summary_k[time].keys():
            news_text_i = "[Date]" + time + ":\n"+"* "+ "\n* ".join(complement_summary_k[time][md5])
            news_text_k.append(news_text_i)

    news_text_r = []
    for time in complement_summary_r.keys():
        for md5 in complement_summary_r[time].keys():
            news_text_i = "[Date]" + time + ":\n"+"* "+ "\n* ".join(complement_summary_r[time][md5])
            news_text_r.append(news_text_i)

    input = (
            "[Query]: ({}, {}, {})\n"
            "[Key News Text]:\n{}\n"
            "[Relevant News Text]:\n{}\n"
            "[Options]:\n{}\n"
            "You must only generate the letter of the correct option without any explanation."
        ).format(subject, relation, time, "\n".join(news_text_k), "\n".join(news_text_r),
                "\n".join([f"{option[0]}: {option[1]}" for option in mapped_options]))
    for option in mapped_options:
        if option[1] == correct_object:
            correct_object_letter = option[0]
            break
    return input,[correct_object_letter, correct_object],mapped_options


def run_llm(prompt,prompt_rules, temperature, max_tokens, opeani_api_keys, engine):
    # openai.api_key = opeani_api_keys
    client = OpenAI(api_key=opeani_api_keys,)
    messages = [{"role":"system","content":prompt_rules}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    response = client.chat.completions.create(
                    model=engine,
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed = 12345,
                    # top_p=0.5,
                    frequency_penalty=0,
                    presence_penalty=0)
    result = response.choices[0].message.content

    return result


class LinkDataset(object):
    def __init__(self):
        with open(f'{ROOT_PATH}/datasets_forecasting/text/function_json/doc_list.json', 'r') as file:
            self.doc = json.load(file)
        with open(f'{ROOT_PATH}/datasets_forecasting/text/function_json/sum_gemini_all.json', 'r') as file:
            self.doc_sum = json.load(file)
        with open(f'{ROOT_PATH}/datasets_forecasting/text/function_json/Md52timid.json','r') as f:
            self.Md52timid = json.load(f)
        self.df_all = pd.read_csv(f"{ROOT_PATH}/datasets_forecasting/text/event/all_dataset.csv",sep="\t",dtype={"Relation_id": str})

        with open(f'{ROOT_PATH}/datasets_forecasting/text/function_json/classification_gemini_all.json','r') as f:
            self.classification = json.load(f)
        with open(f'{ROOT_PATH}/datasets_forecasting/text/function_json/aligned_gemini_all.json','r') as f:
            self.aligned = json.load(f)
        with open(f'{ROOT_PATH}/datasets_forecasting/text/function_json/complementary_gemini_all.json','r') as f:
            self.complementary = json.load(f)

    def retrieve_text(self, subject, ceid, timid, histlen, opt):
        event_temporal_exploration = self.df_all.loc[(self.df_all['ce_id'] == ceid) &(self.df_all['timid'] < timid) & (self.df_all['timid'] >= timid-histlen)]
        Md5_list = event_temporal_exploration['Md5'].unique().tolist()
        documents = []
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

        nodes_key = []
        nodes_related = []

        for Md5 in Md5_list:
            sum_i = self.doc_sum[Md5][1:]

            sum_i_key = []
            sum_i_related = []

            num_list = []
            compl_list = []

            nodes_key_i = []
            nodes_related_i = []

            for image_i in self.classification[Md5].keys():
                if self.classification[Md5][image_i] == "aligned":
                    num_i = self.aligned[Md5][image_i]
                    if num_i not in num_list:
                        num_list.append(num_i)
                elif self.classification[Md5][image_i] == "complementary":
                    content_compl_i = self.complementary[Md5][image_i]
                    compl_list.append(content_compl_i.strip("* ").strip('\n'))

            for num_i_sum in range(len(sum_i)):
                if num_i_sum + 1 in num_list:
                    sum_i_key.append(sum_i[num_i_sum].strip('\n'))
                else:
                    sum_i_related.append(sum_i[num_i_sum].strip('\n'))
            
            sum_i_related.extend(compl_list)
            
            for index_i, text_i in enumerate(sum_i_key):
                text_i = text_i.strip('\n')
                node_i = TextNode(text=text_i, id_=f"{Md5}_{index_i}", metadata={'timid':self.Md52timid[Md5][0],'Md5':Md5,'title':self.doc[Md5]["Title"]})
                nodes_key_i.append(node_i)

            add_num = len(sum_i_key)

            for index_i, text_i in enumerate(sum_i_related):
                text_i = text_i.strip('\n')
                node_i = TextNode(text=text_i, id_=f"{Md5}_{index_i+add_num}", metadata={'timid':self.Md52timid[Md5][0],'Md5':Md5,'title':self.doc[Md5]["Title"]})
                nodes_related_i.append(node_i)

            if len(nodes_key_i) != 1:
                for index_n, node_i in enumerate(nodes_key_i):
                    if index_n == 0:
                        node_i.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                            node_id=nodes_key_i[index_n+1].node_id
                        )
                    elif index_n == len(nodes_key_i)-1:
                        node_i.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                            node_id=nodes_key_i[index_n-1].node_id
                        )
                    else:
                        node_i.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                            node_id=nodes_key_i[index_n+1].node_id
                        )
                        node_i.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                            node_id=nodes_key_i[index_n-1].node_id
                        )

            if len(nodes_related_i) != 1:
                for index_n, node_i in enumerate(nodes_related_i):
                    if index_n == 0:
                        node_i.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                            node_id=nodes_related_i[index_n+1].node_id
                        )
                    elif index_n == len(nodes_related_i)-1:
                        node_i.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                            node_id=nodes_related_i[index_n-1].node_id
                        )
                    else:
                        node_i.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                            node_id=nodes_related_i[index_n+1].node_id
                        )
                        node_i.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                            node_id=nodes_related_i[index_n-1].node_id
                        )

            nodes_key.extend(nodes_key_i)
            nodes_related.extend(nodes_related_i)

        index_key = VectorStoreIndex(nodes_key, use_async=True)

        retriever_key = VectorIndexRetriever(
            index=index_key,
            similarity_top_k= opt.key_number,
        )
        
        response_key = retriever_key.retrieve(subject)
        related_number = 50 - len(response_key)

        index_related = VectorStoreIndex(nodes_related, use_async=True)

        retriever_related = VectorIndexRetriever(
            index=index_related,
            similarity_top_k= related_number,
        )

        response_related = retriever_related.retrieve(subject)

        return response_key, response_related

def main(args):
    
    print(f"args.data_path:{args.data_path}")
    print(f"args.max_tokens:{args.max_tokens}")
    print(f"args.temperature:{args.temperature}")
    print(f"args.engine:{args.engine}")
    print(f"args.histlen:{args.histlen}")

    os.environ["OPENAI_API_KEY"] = "your api key"

    df = pd.read_csv(args.data_path, sep="\t",dtype={"Relation_id": str})

    database = LinkDataset()
    total_num = 0
    true_num = 0
    true_num_group = [0]*len(df['groupid'].unique().tolist())
    all_result = []
    wrong_event = []

    with tqdm(total=df.shape[0], desc="Event Forecasting") as pbar:
        for index_event, row in df.iterrows():
            group_num = row['groupid']
            id = row['ID']
            response = []
            complement_summary = []
            histlen = args.histlen

            # retrieve_text
            retrieve_result_k, retrieve_result_r = database.retrieve_text(row['Subject'],row['ce_id'],row['timid'],histlen, args)

            # filter text
            text_with_timid_k = {r.node_id:(r.metadata['timid'],r.text, r.metadata['Md5'])for r in retrieve_result_k}
            text_with_timid_r = {r.node_id:(r.metadata['timid'],r.text, r.metadata['Md5'])for r in retrieve_result_r}

            text_sort_by_timid_k = sorted(text_with_timid_k.values(), key=lambda item: item[0])
            text_sort_by_timid_r = sorted(text_with_timid_r.values(), key=lambda item: item[0])

            complement_summary_k = {}

            for text_sort_by_timid_i in text_sort_by_timid_k:
                if f"{text_sort_by_timid_i[0]}" not in complement_summary_k:
                    complement_summary_k[f"{text_sort_by_timid_i[0]}"] = {}
                    if f"{text_sort_by_timid_i[2]}" not in complement_summary_k[f"{text_sort_by_timid_i[0]}"]:
                        complement_summary_k[f"{text_sort_by_timid_i[0]}"][f"{text_sort_by_timid_i[2]}"] = [text_sort_by_timid_i[1]]
                        
                else:
                    if f"{text_sort_by_timid_i[2]}" not in complement_summary_k[f"{text_sort_by_timid_i[0]}"]:
                        complement_summary_k[f"{text_sort_by_timid_i[0]}"][f"{text_sort_by_timid_i[2]}"] = [text_sort_by_timid_i[1]]
                    else:
                        complement_summary_k[f"{text_sort_by_timid_i[0]}"][f"{text_sort_by_timid_i[2]}"].append(text_sort_by_timid_i[1])

            complement_summary_r = {}

            for text_sort_by_timid_i in text_sort_by_timid_r:
                if f"{text_sort_by_timid_i[0]}" not in complement_summary_r:
                    complement_summary_r[f"{text_sort_by_timid_i[0]}"] = {}
                    if f"{text_sort_by_timid_i[2]}" not in complement_summary_r[f"{text_sort_by_timid_i[0]}"]:
                        complement_summary_r[f"{text_sort_by_timid_i[0]}"][f"{text_sort_by_timid_i[2]}"] = [text_sort_by_timid_i[1]]
                        
                else:
                    if f"{text_sort_by_timid_i[2]}" not in complement_summary_r[f"{text_sort_by_timid_i[0]}"]:
                        complement_summary_r[f"{text_sort_by_timid_i[0]}"][f"{text_sort_by_timid_i[2]}"] = [text_sort_by_timid_i[1]]
                    else:
                        complement_summary_r[f"{text_sort_by_timid_i[0]}"][f"{text_sort_by_timid_i[2]}"].append(text_sort_by_timid_i[1])
            
            # event forecasting
            prompt_reasoning,correct_option,mapped_options = generate_reasoning_input(row, complement_summary_k, complement_summary_r)
            prompt_reasoning_rules = generate_prompt_rules('reasoning')
            result = run_llm(prompt_reasoning, prompt_reasoning_rules, args.temperature, args.max_tokens, args.opeani_api_keys, args.engine)
            sleep(1)

            if result == None:
                pbar.update(1)
                wrong_event.append(index_event)
                continue
        
            response = parse_outputs(result)

            total_num += 1
            pbar.update(1)
            if any(item in correct_option for item in response):
                true_num +=1
                true_num_group[group_num] +=1
    
            all_result.append(
                {
                    "correct_option": correct_option,
                    "predict": result,
                    "mapped_options": mapped_options,
                    "group_num":group_num,
                    "ID": id,
                    "index": index_event
                }
            )

            if total_num%200 == 0:
                print(f"current acc:{true_num/total_num}")

        group_dict = df['groupid'].value_counts().to_dict()
        print(f"total_num:{total_num} == {sum(list(group_dict.values()))}")

        precision = true_num/total_num
        print(f"precision:{precision}")
        for i in range(len(true_num_group)):
            precision_group = true_num_group[i]/group_dict[i]
            print(f"precision{i}:{precision_group}")

        with open("./result_RAG_text_w_image.json", "w") as f: 
            json.dump(all_result, f, ensure_ascii=False, indent=2)
        print('save in ./result_RAG_text_w_image.json')

        wrong_ev = {}
        wrong_ev['wrong_ev_index'] = wrong_event
        with open("./wrong_event_w_image.json", "w") as f: 
            json.dump(wrong_ev, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="your root path")
    parser.add_argument("--data_path", type=str, default="datasets_forecasting/text/event/final_test.csv")
    parser.add_argument("--opeani_api_keys", type=str, default="your api key")
    parser.add_argument("--engine", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0)
    # parser.add_argument("--repetition_penalty", type=float, default=0.4)
    parser.add_argument("--max_tokens", type=int, default=128)

    parser.add_argument("--histlen", type=int, default=30)
    parser.add_argument("--key_number", type=int, default=20)

    args = parser.parse_args()

    ROOT_PATH = args.root_path
    args.data_path = os.path.join(ROOT_PATH, args.data_path)

    main(args)



    


