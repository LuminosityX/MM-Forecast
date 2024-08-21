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

def generate_prompt(row):
    def format_events(events):
        if events == []:
            return ""
        return "* " + "\n* ".join(["(" + ", ".join(map(str, event)) + ")" for event in events])

    rules = [
        "1. The atomic event is the basic unit describing a specific event, typically presented in the form of a quadruple (S, R, O, T), where S represents the subject, R represents the relation, O represents the object, and T represents the relative time.\n",
        "2. Given a query of (S, O, T) in the future and the list of historical events until t, event forecasting aims to predict the missing relation.",
    ]
    prompt_rules = "You are an assistant to perform event forecasting with the following rules:\n" + ''.join(rules) 

    subject = row['Subject']
    correct_relation = row['Relation']
    object = row['Object']
    time = row['timid']

    related_facts_list = absolute2relative(eval(row['Nearest_events_relation'])) + absolute2relative(eval(row['Further_events_relation'])) + absolute2relative(eval(row['Related_Facts_relation']))
    related_facts_list = sorted(related_facts_list, key = lambda x:x[3])

    related_facts = format_events(related_facts_list)

    candidates = eval(row['Candidates_relation'])
    group_num = row['groupid']
    id = row['ID']
    
    options = candidates + [correct_relation]
    random.shuffle(options)

    option_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapped_options = {option: option_letters[i]  for i, option in enumerate(options)}

    prompt = (
        "[Query]: ({}, {}, {})\n"
        "[Related Events]: {}\n"
        "[Options]:\n{}\n"
        "You must only generate the letter of the correct option without any explanation."
    ).format(subject, object, time, related_facts, 
             "\n".join([f"{letter}: {option}" for option, letter  in mapped_options.items()]))
    
    return prompt, prompt_rules, [mapped_options[correct_relation], correct_relation], group_num, id


def run_llm(prompt,prompt_rules, temperature, max_tokens, opeani_api_keys, engine):
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
                    seed = 12345,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)
    result = response.choices[0].message.content

    return result


    



# @torch.inference_mode()
def main(args):
    
    print(f"args.data_path:{args.data_path}")
    print(f"args.max_tokens:{args.max_tokens}")
    print(f"args.temperature:{args.temperature}")
    print(f"args.engine:{args.engine}")
    
    df = pd.read_csv(args.data_path, sep="\t",dtype={"Relation_id": str})
    df = df.loc[df['timid']>365*6]
    total_num = 0
    true_num = 0
    true_num_group = [0]*len(df['groupid'].unique().tolist())
    result = []
    wrong_event = []

    with tqdm(total=df.shape[0], desc="Event Forecasting") as pbar:
        for index, row in df.iterrows():
            prompt,prompt_rules,correct_option,group_num,id = generate_prompt(row)
            response = run_llm(prompt,prompt_rules, args.temperature, args.max_tokens, args.opeani_api_keys, args.engine)
        
            if response == None:
                pbar.update(1)
                wrong_event.append(index)
                continue
            
            total_num += 1
            pbar.update(1)
            response = parse_outputs(response)

            if any(item in correct_option for item in response):
                true_num +=1
                true_num_group[group_num] +=1
            elif response[0] == None:
                if correct_option[1] in response[1]:
                    true_num +=1
                    true_num_group[group_num] +=1

            result.append(
                {
                    "answer": correct_option,
                    "predict": response,
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

        with open("./result_ICL_graph_only_relation.json", "w") as f: 
            json.dump(result, f, ensure_ascii=False, indent=2)
        print('save in ./result_ICL_graph_only_relation.json')

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
    args = parser.parse_args()

    ROOT_PATH = args.root_path
    args.data_path = os.path.join(ROOT_PATH, args.data_path)

    main(args)



    

