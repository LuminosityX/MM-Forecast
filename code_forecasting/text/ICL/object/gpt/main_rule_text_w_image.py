import os
import json
import argparse
import json
import os

import random
from tqdm import tqdm
import pandas as pd
import re

from time import sleep
from openai import OpenAI


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


def generate_prompt(row, summary_list_clean, Md52timid, classification, aligned, complementary):

    def format_events_focus(events):
        sum_all = []
        for event in events:
            event = event[0]
            sum_i = []
            num_list = []
            for image_i in classification[event].keys():
                if classification[event][image_i] == "aligned":
                    num_i = aligned[event][image_i]
                    if num_i not in num_list:
                        num_list.append(num_i)
                        sum_i.append(summary_list_clean[event][num_i].strip('\n'))
            if sum_i != []:
                sum_i = "* " + "\n* ".join(sum_i) 
                sum_all.append("[Date]" + str(Md52timid[event][0]) + ":\n"+sum_i)

        return "\n".join(sum_all)
    
    def format_events_related(events):
        sum_all = []
        for event in events:
            event = event[0]

            num_list = []
            compl_list = []
            for image_i in classification[event].keys():
                if classification[event][image_i] == "aligned":
                    print(event, end=" ")
                    print(image_i, end=" ")
                    
                    num_i = aligned[event][image_i]
                    print(num_i)
                    if num_i not in num_list:
                        num_list.append(num_i)
                elif classification[event][image_i] == "complementary":
                    print(event, end=" ")
                    print(image_i, end=" ")
                    content_compl_i = complementary[event][image_i]
                    print(content_compl_i[0])
                    compl_list.append(content_compl_i[0])

            compl_list_str = '\n'.join(compl_list)

            sum_i = []
            for num_i_sum in range(len(summary_list_clean[event])):
                if num_i_sum == 0 or num_i_sum in num_list:
                    continue
                else:
                    sum_i.append(summary_list_clean[event][num_i_sum].strip('\n'))

            sum_i = "* " + "\n* ".join(sum_i) 

            if compl_list != []:
                sum_i = sum_i + '\n' + compl_list_str

            sum_all.append("[Date]" + str(Md52timid[event][0]) + ":\n"+sum_i)

        return "\n".join(sum_all)

    rules = [
        "1. The atomic event is the basic unit describing a specific event, typically presented in the form of a quadruple (S, R, O, T), where S represents the subject, R represents the relation, O represents the object, and T represents the relative time.\n",
        "2. When formulating the ultimate prediction, the preeminent factor to be meticulously weighed and scrutinized is the [Key Events]. Complementing this paramount consideration is the [Related events], which, though ancillary in nature, serves as a valuable adjunct, furnishing pertinent contextual details and auxiliary insights to fortify the predictive analysis.\n",
        "3. Given a query of (S, R, T) in the future and the list of historical events until t, event forecasting aims to predict the missing object.",
    ]
    prompt_rules = "You are an assistant to perform event forecasting with the following rules:\n" + ''.join(rules)

    subject = row['Subject']
    relation = row['Relation']
    correct_object = row['Object']
    time = row['timid']
    
    related_facts_list = eval(row['Nearest_events_summary']) + eval(row['Further_events_summary']) + eval(row['Related_facts_summary'])
    #print(related_facts_list)
    related_facts_dic = {}
    for re_event in related_facts_list:
        related_facts_dic[re_event] = Md52timid[re_event][0]
    
    related_facts_list_sort = sorted(related_facts_dic.items(), key = lambda kv:(kv[1], kv[0]))
    #print(related_facts_list_sort)

    #print('-------------')
    focus_facts = format_events_focus(related_facts_list_sort)
    related_facts = format_events_related(related_facts_list_sort)
    #print('-------------')

    ###
    candidates = eval(row['Candidates'])
    group_num = row['groupid']
    #id = row['ID']
    
    options = candidates + [correct_object]
    random.shuffle(options)

    option_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapped_options = {option: option_letters[i]  for i, option in enumerate(options)}

    prompt = (
        "[Query]: ({}, {}, {})\n"
        "[Key Events]: \n{}\n"
        "[Related Events]: \n{}\n"
        "[Options]:\n{}\n"
        "You must only generate the letter of the correct option without any explanation."
    ).format(subject, relation, time, focus_facts, related_facts, 
             "\n".join([f"{letter}: {option}" for option, letter  in mapped_options.items()]))
    
    return prompt, prompt_rules, [mapped_options[correct_object], correct_object], group_num


def run_llm(prompt,prompt_rules, temperature, max_tokens, opeani_api_keys, engine):

    client = OpenAI(
        api_key=opeani_api_keys,
        
        )
    messages = [{"role":"system","content":prompt_rules}]
    message_prompt = {"role":"user","content":prompt}
    #print(prompt)
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
    #print(result)

    return result



# @torch.inference_mode()
def main(args):
    # sys.stderr = Logger(stream = sys.stderr)
    print(f"args.data_path:{args.data_path}")
    print(f"args.max_tokens:{args.max_tokens}")
    print(f"args.temperature:{args.temperature}")
    print(f"args.engine:{args.engine}")
    
    df = pd.read_csv(args.data_path, sep="\t",dtype={"Relation_id": str})
    df = df.loc[df['timid']>365*6]
    with open(f'{ROOT_PATH}/datasets_forecasting/function_json/sum_gemini_all.json', 'r') as file:
        summary_list_clean = json.load(file)
    with open(f'{ROOT_PATH}/function_json/Md52timid.json','r') as f:
        Md52timid = json.load(f)

    with open(f'{ROOT_PATH}/function_json/classification_gemini_all.json','r') as f:
        classification = json.load(f)
    with open(f'{ROOT_PATH}/function_json/aligned_gemini_all.json','r') as f:
        aligned = json.load(f)
    with open(f'{ROOT_PATH}/function_json/complementary_gemini_all.json','r') as f:
        complementary = json.load(f)

    total_num = 0
    true_num = 0
    true_num_group = [0]*len(df['groupid'].unique().tolist())
    result = []

    wrong_event = []

    with tqdm(total=df.shape[0], desc="Event Forecasting") as pbar:
        for index, row in df.iterrows():
            
            id = row['ID']
            prompt,prompt_rules,correct_option,group_num = generate_prompt(row,summary_list_clean,Md52timid, classification, aligned, complementary)
            response = run_llm(prompt,prompt_rules, args.temperature, args.max_tokens, args.opeani_api_keys, args.engine)
            sleep(1)

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
        print(f"total_num:{total_num} == {sum(list(group_dict.values()))}")

        precision = true_num/total_num
        print(f"precision:{precision}")
        for i in range(len(true_num_group)):
            precision_group = true_num_group[i]/group_dict[i]
            print(f"precision{i}:{precision_group}")

        with open("./result_ICL_text_w_image.json", "w") as f: 
            json.dump(result, f, ensure_ascii=False, indent=2)
        print('save in ./result_ICL_text_w_image.json')

        wrong_ev = {}
        wrong_ev['wrong_ev_index'] = wrong_event
        with open("./wrong_event_w_image.json", "w") as f: 
            json.dump(wrong_ev, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add_model_args(parser)
    parser.add_argument("--root_path", type=str, default="your root path")
    parser.add_argument("--data_path", type=str, default="datasets_forecasting/text/event/final_test.csv")
    parser.add_argument("--opeani_api_keys", type=str, default="your api keys")
    parser.add_argument("--engine", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    ROOT_PATH = args.root_path
    args.data_path = os.path.join(ROOT_PATH, args.data_path)

    main(args)



    

