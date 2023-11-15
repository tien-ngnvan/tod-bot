from DM import Dialogue_Manager
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration
from accelerate import PartialState
import gradio as gr
import time
import numpy as np
import pandas as pd

dm = Dialogue_Manager()
model_name_or_path = 'google/flan-t5-base'
config_path = r'C:\Users\This PC\PycharmProjects\AIP\src\config.json'
ckpt_dst = r'C:\Users\This PC\PycharmProjects\AIP\checkpoint\model_dst_v1.bin'
ckpt_res = r'C:\Users\This PC\PycharmProjects\AIP\checkpoint\model_res_v1.bin'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
config = AutoConfig.from_pretrained(config_path)

model_dst = T5ForConditionalGeneration.from_pretrained(ckpt_dst, config=config,local_files_only=True)
model_res = T5ForConditionalGeneration.from_pretrained(ckpt_res, config=config,local_files_only=True)

distributed_state_dst = PartialState()
distributed_state_res = PartialState()
model_dst.to(distributed_state_dst.device)
model_res.to(distributed_state_res.device)

embedding_size_dst = model_dst.get_input_embeddings().weight.shape[0]
embedding_size_res = model_res.get_input_embeddings().weight.shape[0]

if len(tokenizer) > embedding_size_dst:
    model_dst.resize_token_embeddings(len(tokenizer))
if len(tokenizer) > embedding_size_res:
    model_res.resize_token_embeddings(len(tokenizer))

sample_dst = {
    "instruction": "The goal of this assignment is to determine the belief state by analyzing the dialogue. Giving the list of personal action [ACTIONS] {list_user_action}. When user query is out of the ontology, respond \"Unsure about answer, you should find with SearchEngine [TERM]\" where TERM is the search term you want to find out if not sure about the answer. Input: <CTX> {context} <QUERY> {current_query} <ONTOLOGY> {ontology}. Output: ",
    "list_user_action": "inform, request, inform_intent, negate_intent, affirm_intent, affirm, negate, select, thank_you, goodbye, greet, general_asking, request_alts",
    "ontology": "HOTELS_1:(slot0=location of the hotel; slot1=number of rooms in the reservation; slot2=start date for the reservation; slot3=number of days in the reservation; slot4=star rating of the hotel; slot5=name of the hotel; slot6=address of the hotel; slot7=phone number of the hotel; slot8=price per night for the reservation; slot9=boolean flag indicating if the hotel has wifi)",
    "current_query": "",
    "context": ""
}
sample_res = {
    "instruction": "Follow the conversation context of the task is taken into consideration <CTX> {context} <EOD> and ensure that use provides in the given <K> {ontology} {system_action} {documents} you must respond to the conversation with <S> {style}. <Q> What should be taken to complete the task effectively?",
    "context": "",
    "ontology": "HOTELS_1:(slot0=location of the hotel; slot1=number of rooms in the reservation; slot2=start date for the reservation; slot3=number of days in the reservation; slot4=star rating of the hotel; slot5=name of the hotel; slot6=address of the hotel; slot7=phone number of the hotel; slot8=price per night for the reservation; slot9=boolean flag indicating if the hotel has wifi)",
    "system_action": "",
    "documents": "",
    "style": "politely"
}

def add_utterance(context, utterance, size):
    context += [(utterance, None)]
    if len(context) > size:
        context.pop(0)
    return context

def add_state(current_query):
    sample_dst_tmp = {
        "instruction": "The goal of this assignment is to determine the belief state by analyzing the dialogue. "
                       "Giving the list of personal action [ACTIONS] {list_user_action}. When user query is out of "
                       "the ontology, respond \"Unsure about answer, you should find with SearchEngine [TERM]\" where "
                       "TERM is the search term you want to find out if not sure about the answer. Input: <CTX> {"
                       "context} <QUERY> {current_query} <ONTOLOGY> {ontology}. Output: ",
        "list_user_action": "inform, request, inform_intent, negate_intent, affirm_intent, affirm, negate, select, "
                            "thank_you, goodbye, greet, general_asking, request_alts",
        "ontology": "HOTELS_1:(slot0=location of the hotel; slot1=number of rooms in the reservation; slot2=start "
                    "date for the reservation; slot3=number of days in the reservation; slot4=star rating of the "
                    "hotel; slot5=name of the hotel; slot6=address of the hotel; slot7=phone number of the hotel; "
                    "slot8=price per night for the reservation; slot9=boolean flag indicating if the hotel has wifi)",
        "current_query": "USER: " + current_query,
        "context": ""}
    item_dst = sample_dst_tmp['instruction'] \
        .replace('{list_user_action}', sample_dst_tmp['list_user_action'].strip()) \
        .replace('{context}', sample_dst_tmp['context'].strip()) \
        .replace('{current_query}', sample_dst_tmp['current_query'].strip()) \
        .replace('{ontology}', sample_dst_tmp['ontology'].strip())
    input_tokens = tokenizer(item_dst, return_tensors="pt")
    output = model_dst.generate(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"],
                                max_new_tokens=100)
    data_dst_1 = tokenizer.decode(output[0], skip_special_tokens=True)
    dm.format_string_input(data_dst_1)
    return dm.input_transformed

def add_action(system_action):
    sample_res_tmp = {
        "instruction": "Follow the conversation context of the task is taken into consideration <CTX> {context} <EOD> "
                       "and ensure that use provides in the given <K> {ontology} {system_action} {documents} you must "
                       "respond to the conversation with <S> {style}. <Q> What should be taken to complete the task "
                       "effectively?",
        "context": "",
        "ontology": "HOTELS_1:(slot0=location of the hotel; slot1=number of rooms in the reservation; slot2=start "
                    "date for the reservation; slot3=number of days in the reservation; slot4=star rating of the "
                    "hotel; slot5=name of the hotel; slot6=address of the hotel; slot7=phone number of the hotel; "
                    "slot8=price per night for the reservation; slot9=boolean flag indicating if the hotel has wifi)",
        "system_action": system_action,
        "documents": "",
        "style": "politely"

    }
    item_res = sample_res_tmp['instruction'] \
        .replace('{ontology}', sample_res_tmp['ontology'].strip()) \
        .replace('{context}', sample_res_tmp['context'].strip()) \
        .replace('{style}', sample_res_tmp['style'].strip()) \
        .replace('{system_action}', sample_res_tmp['system_action'].strip()) \
        .replace('{documents}', sample_res_tmp['documents'].strip())

    input_tokens = tokenizer(item_res, return_tensors="pt")
    output = model_res.generate(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"],
                                max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def add_text(context, current_query):
    sample_dst["current_query"] = "USER: " + current_query
    item_dst = sample_dst['instruction'] \
        .replace('{list_user_action}', sample_dst['list_user_action'].strip()) \
        .replace('{context}', sample_dst['context'].strip()) \
        .replace('{current_query}', sample_dst['current_query'].strip()) \
        .replace('{ontology}', sample_dst['ontology'].strip())
    input_tokens = tokenizer(item_dst, return_tensors="pt")
    output = model_dst.generate(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"],
                                max_new_tokens=100)
    data_dst_1 = tokenizer.decode(output[0], skip_special_tokens=True)

    dm.format_string_input(data_dst_1)
    if dm.classify_dst == "tod":
        dm.transform_action()

        sample_res["system_action"] = dm.format_string_output()
    sample_dst['context'] = ""
    context += [(sample_dst["current_query"], None)]

    for pair in context[-3:]:
        for utt in pair:
            if utt != None:
                sample_dst["context"] += " "+utt

    sample_dst["context"] = sample_dst["context"].strip()
    return context, gr.Textbox(value="", interactive=False)

def bot(context):
    item_res = sample_res['instruction'] \
        .replace('{ontology}', sample_res['ontology'].strip()) \
        .replace('{context}', sample_res['context'].strip()) \
        .replace('{style}', sample_res['style'].strip()) \
        .replace('{system_action}', sample_res['system_action'].strip()) \
        .replace('{documents}', sample_res['documents'].strip())

    input_tokens = tokenizer(item_res, return_tensors="pt")
    output = model_res.generate(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"],
                                max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    context[-1][1] = "AGENT: "
    for character in response:
        context[-1][1] += character
        time.sleep(0.0001)
        yield context


with gr.Blocks() as demo:
    with gr.Tab("Module 1"):
        state = gr.Interface(fn=add_state, inputs="text", outputs="text")

    with gr.Tab("Module 3"):
        response = gr.Interface(fn=add_action, inputs="text", outputs="text")

    with gr.Tab("All modules"):
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
        )
        with gr.Row():
            txt = gr.Textbox(
                scale=6,
                show_label=False,
                placeholder="Enter text and press enter",
                container=False,
            )
        txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False)\
            .then(bot, chatbot, chatbot, api_name="bot_response")

demo.queue()
if __name__ == "__main__":
    demo.launch()
