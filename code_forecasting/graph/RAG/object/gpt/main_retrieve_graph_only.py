import os
import json
import torch
import argparse
import json
import os

import random
from tqdm import tqdm
import pandas as pd
import re
from openai import OpenAI

from time import sleep

# os.environ["CUDA_VISIBLE_DEVICES"]="2"

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


def read_dictionary(filename):
    d = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[1])] = line[0]
    return d

date2id = read_dictionary(f"{ROOT_PATH}/datasets_forecasting/graph/event/date2id.txt")
id2date = {v:k for k,v in date2id.items()}

def absolute2relative(events):
    events_relative = []
    if len(events)>0:
        for event in events:
            timid = id2date[event[3]]
            event = list(event)
            event[3] = timid
            events_relative.append(event)
    return events_relative

def generate_prompt_rules(task):
    if  task == 'reasoning':
        rules = [
        "1. The atomic event is the basic unit describing a specific event, typically presented in the form of a quadruple (S, R, O, T), where S represents the subject, R represents the relation, O represents the object, and T represents the relative time.\n",
        "2. Given a query of (S, R, T) in the future and the list of historical events until T, event forecasting aims to predict the missing object."
        ]
        prompt_rules = "You are an assistant to perform event forecasting with the following rules:\n" + ''.join(rules)
    elif task == 'prune_entity':
        rules = [
        "1. [Subject] represents the event subject in a specific event. [Candidate Set] represents a list of entities.",
        "2. You need to select the entities that may be relevant to [Subject]."
        ]
        prompt_rules = "You are an assistant to find relevant entities with the following rules:\n" + ''.join(rules)

    return prompt_rules


def generate_prune_input(subject, prune_set):
    prune_list = list(prune_set)
    if subject in prune_list:
        prune_list.remove(subject)
    input = (
        "[Subject]: {}\n"
        "[Candidate Set]: {}\n"
        "Your response should be provided in the form of JSON. The example of output format is as following: [selected_entities: [entity0, entity1,...,entityN-1,entityN]]. If none is selected, ouput None."
    ).format(subject,prune_list)
    return input

def generate_reasoning_input(row, search_result):
    def format_events(events):
        if events == []:
            return ""
        return "* " + "\n* ".join(["(" + ", ".join(map(str, event)) + ")" for event in events])
    
    if type(search_result)==list:
        search_result = sorted(search_result, key = lambda x:x[3])
        search_result = format_events(search_result)
    subject = row['Subject']
    relation = row['Relation']
    time = row['timid']
    correct_object = row['Object']
    candidates = eval(row['Candidates'])
    options = candidates + [correct_object]
    random.shuffle(options)
    option_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapped_options = {option: option_letters[i]  for i, option in enumerate(options)}
    input = (
            "[Query]: ({}, {}, {})\n"
            "[Relevant Event]: \n{}\n"
            "[Options]:\n{}\n"
            "You must only generate the letter of the correct option without any explanation."
        ).format(subject, relation, time, search_result,
                "\n".join([f"{letter}: {option}" for option, letter  in mapped_options.items()]))
    return input,[mapped_options[correct_object], correct_object],mapped_options

def run_llm_prune(prompt,prompt_rules, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    # openai.api_key = opeani_api_keys
    client = OpenAI(
        api_key=opeani_api_keys,
        )
    messages = [{"role":"system","content":prompt_rules}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    response = client.chat.completions.create(
                    model=engine,
                    response_format={ "type": "json_object" },
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed = 12345,
                    # top_p=0.5,
                    frequency_penalty=0,
                    presence_penalty=0)
    result = response.choices[0].message.content
    return result


def run_llm(prompt,prompt_rules, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    # openai.api_key = opeani_api_keys
    client = OpenAI(
        api_key=opeani_api_keys,
        )
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
        self.df_all = pd.read_csv(f"{ROOT_PATH}/datasets_forecasting/graph/event/all_dataset.csv",sep="\t",dtype={"Relation_id": str})

    def read_dictionary(self, filename):
        d = {}
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                d[int(line[1])] = line[0]
        return d
    
    def search_subject(self, query_subject, ceid,timid,histlen):
        event_temporal_exploration = self.df_all.loc[(self.df_all['timid'] < timid) & (self.df_all['timid'] >= timid-histlen)]
        event_spatial_exploration_ce = event_temporal_exploration.loc[(event_temporal_exploration['ce_id'] == ceid)]
        entity_list = set(event_spatial_exploration_ce['Subject'].unique().tolist() + event_spatial_exploration_ce['Object'].unique().tolist())
        return entity_list

    def search_events(self, relevant_subject_set, timid, histlen):
        related_events_subject = self.df_all.loc[(self.df_all['Subject'].isin(relevant_subject_set)) & (self.df_all['timid'] < timid) & (self.df_all['timid'] >= timid-histlen)]
        related_events_object = self.df_all.loc[(self.df_all['Object'].isin(relevant_subject_set)) & (self.df_all['timid'] < timid) & (self.df_all['timid'] >= timid-histlen)]
        events = []
        for _, event in related_events_subject.iterrows():
            if [event['Subject'], event['Relation_choice'], event['Object'], event['timid']] not in events:
                events.append([event['Subject'], event['Relation_choice'], event['Object'], event['timid']])
        for _, event in related_events_object.iterrows():
            if [event['Subject'], event['Relation_choice'], event['Object'], event['timid']] not in events:
                events.append([event['Subject'], event['Relation_choice'], event['Object'], event['timid']])
        events = sorted(events, key=lambda x: x[3])
        events = events[:20]
        return events

def prune_entity(subject, relevant_subject_set, args):
    prompt_prune = generate_prune_input(subject, relevant_subject_set)
    prompt_prune_rules = generate_prompt_rules('prune_entity')
    prune_result = run_llm_prune(prompt_prune, prompt_prune_rules, args.temperature, args.max_tokens, args.opeani_api_keys, args.engine)
    try:
        prune_result = eval(prune_result)['selected_entities']
    except:
        prune_result = []

    return prune_result


@torch.inference_mode()
def main(args):
    
    print(f"args.data_path:{args.data_path}")
    print(f"args.max_tokens:{args.max_tokens}")
    print(f"args.temperature:{args.temperature}")
    print(f"args.engine:{args.engine}")
    print(f"args.histlen:{args.histlen}")
    
    df = pd.read_csv(args.data_path, sep="\t",dtype={"Relation_id": str})
    database = LinkDataset()
    total_num = 0
    true_num = 0
    true_num_group = [0]*len(df['groupid'].unique().tolist())
    all_result = []
    wrong_event = []

    with tqdm(total=df.shape[0], desc="Event Forecasting") as pbar:
        for index, row in df.iterrows():
            group_num = row['groupid']
            id = row['ID']
            histlen = args.histlen
            relevant_subject_candidate_set = database.search_subject(row['Subject'], row['ce_id'], row['timid'], histlen)
            if len(relevant_subject_candidate_set) != 0:
                relevant_subject_set = prune_entity(row['Subject'],relevant_subject_candidate_set, args)
                relevant_subject_set.append(row['Subject'])
                relevant_events = database.search_events(list(set(relevant_subject_set)), row['timid'], histlen)
            else:
                relevant_events = None

            prompt_reasoning,correct_option,mapped_options = generate_reasoning_input(row, relevant_events)
            prompt_reasoning_rules = generate_prompt_rules('reasoning')
            result = run_llm(prompt_reasoning, prompt_reasoning_rules, args.temperature, args.max_tokens, args.opeani_api_keys, args.engine)
            sleep(1)

            if result == None:
                pbar.update(1)
                wrong_event.append(index)
                continue

            total_num += 1
            pbar.update(1)
            response = parse_outputs(result)

            if any(item in correct_option for item in response):
                true_num +=1
                true_num_group[group_num] +=1
            elif response[0] == None:
                if correct_option[1] in response[1]:
                    true_num +=1
                    true_num_group[group_num] +=1

            all_result.append(
                {
                    "correct_option": correct_option,
                    # "relevant_subject_set": relevant_subject_set,
                    "predict": result,
                    "mapped_options":mapped_options,
                    "group_num":group_num,
                    "ID": id
                }
            )

            if total_num%200 == 0:
                print(f"current acc:{true_num/total_num}")

        group_dict = df['groupid'].value_counts().to_dict()
        precision = true_num/total_num
        print(f"precision:{precision}")
        for i in range(len(true_num_group)):
            precision_group = true_num_group[i]/group_dict[i]
            print(f"precision{i}:{precision_group}")

        with open("./result_RAG_graph_only.json", "w") as f: 
            json.dump(all_result, f, ensure_ascii=False, indent=2)
        print('save in ./result_RAG_graph_only.json')

        wrong_ev = {}
        wrong_ev['wrong_ev_index'] = wrong_event
        with open("./wrong_event.json", "w") as f: 
            json.dump(wrong_ev, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add_model_args(parser)
    parser.add_argument("--root_path", type=str, default="your root path")
    parser.add_argument("--data_path", type=str, default="datasets_forecasting/graph/event/final_test.csv")
    parser.add_argument("--opeani_api_keys", type=str, default="your api key")
    parser.add_argument("--engine", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--histlen", type=int, default=30)
    args = parser.parse_args()

    ROOT_PATH = args.root_path
    args.data_path = os.path.join(ROOT_PATH, args.data_path)

    main(args)



    


