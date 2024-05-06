import stanza
import re
import torch
from spert import util  
from spert import constant
import sys
# use parse tree as a span selector
# version build: 1.0
# author: Tee @4694

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

def get_synspan(tree, synspan=None):
  # Trích xuất ra các span dựa trên cây cú pháp
    if synspan is None:
        synspan = []

    for child in tree.children:
        if not child.children:
            return
        else:
            synspan.append(child.leaf_labels())
            get_synspan(child, synspan)
    return synspan

def get_synsent(synspan, array):
    # Vì cây cú pháp tách rời ký tự '-', nên ở đây thực hiện một số bước để khôi phục lại từ nối
    synsent = []
    for item in array:
      synsent.append([item])
    for span in synspan:
      if len(span) > 1:
        i = 0
        while i < len(span):        
          if span[i] == "-" or span[i] == "/":
              span[i-1] += span[i] + span[i+1]
              del span[i+1]
              del span[i]
          else:
                i += 1
      if span not in synsent:
        synsent.append(span)
    return synsent

def find_span(sentence, span):
  # Ghi chú: Đang tìm kiếm tuần tự vị trí của span trong sentence
  # Tìm start, end của các span thu được từ cây cú pháp
    span_len = len(span)
    for i in range(len(sentence) - span_len + 1):
        if sentence[i:i+span_len] == span:
            return i, i+span_len-1
    return -1, -1 

def get_index_span(doc):
  array =  []
  for token in doc.tokens:
    array.append(token)
  articles = ["any", "some", "these", "those", "this", "that", "a", "an", "the", 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs']
  array = [str(item) for item in array]
  new_array = []
  for word in array:
    parts = re.split(r'(-)', word)
    new_array.extend(parts)
  array_without_articles = [item for item in new_array if str(item).lower() not in articles]
  sent = ' '.join(array_without_articles)

  docu = nlp(sent)

  tree = docu.sentences[0].constituency # Cây phân tích cú pháp cho câu
  all_span = get_synspan(tree)
  all_token_span = get_synsent(all_span, array)

  span_index = []
  for i in range(len(all_token_span)):
    if i<len(array):
      span_index.append((i, i))
    else:
      start_index, end_index = find_span(array, all_token_span[i])
      if start_index != -1:
        span_index.append((start_index, end_index))
  return span_index

