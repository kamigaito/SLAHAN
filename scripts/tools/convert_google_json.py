import argparse
import sys
import json
import codecs

args = sys.argv 
parser = argparse.ArgumentParser(description='Google sentemce compression dataset converter')
parser.add_argument("-i", "--input-json-file", dest="file_input", type=str, help="path to source file")
parser.add_argument("-s", "--sent-file", dest="file_sent", type=str, help="path to output sent file")
parser.add_argument("-l", "--label-file", dest="file_label", type=str, help="path to output label file")
parser.add_argument("-d", "--dep-file", dest="file_dep", type=str, help="path to output dep file")
parser.add_argument("-r", "--rel-file", dest="file_rel", type=str, help="path to output rel file")
parser.add_argument("-p", "--pos-file", dest="file_pos", type=str, help="path to output pos file")
parser.add_argument("-t", "--conv-large", dest="conv_large", action="store_true", help="conversion flag for large dataset.")
parser.add_argument("-m", "--max-length", dest="max_length", type=int, help="maximum length limitation.", default=200)
args = parser.parse_args()

json_file = codecs.open(args.file_input, encoding="utf8")
s_f = codecs.open(args.file_sent, "w", encoding="utf8")
l_f = codecs.open(args.file_label, "w", encoding="utf8")
d_f = codecs.open(args.file_dep, "w", encoding="utf8")
r_f = codecs.open(args.file_rel, "w", encoding="utf8")
p_f = codecs.open(args.file_pos, "w", encoding="utf8")

each_json = ""
for line in json_file:
    line = line.rstrip("\r\n")
    each_json += line
    if line == "}":
        sent = json.loads(each_json)
        each_json = ""
        word_pos = 0
        word_list = []
        word_map_list = {}
        word_str_list = {}
        source_chunk_list = {}
        selected_word_list = {}
        last_selected_edge_id = 0
        end_char = sent["compression_untransformed"]["text"][-1]
        source_sent = "<s>"
        label_body = "<s>"
        dep_body = "0-0"
        rel_body = "<s>"
        pos_body = "<s>"
        for node in sent["source_tree"]["node"]:
            head_word_id = node["word"][node["head_word_index"]]["id"]
            source_chunk_list[head_word_id] = []
            for word in node["word"]:
                word_map_list[word['id']] = word_pos
                word_str_list[word['id']] = word["form"]
                word_pos += 1
                if word['id'] > -1:
                    word_list.append(word['id'])
                    source_chunk_list[head_word_id].append(word['id'])
                    source_sent += " " + word['form']
                    pos_body += " " + word['tag']
        source_sent += " </s>"
        pos_body += " </s>"
        for node in sent["source_tree"]["edge"]:
            if word_map_list[node['child_id']] > -1:
                for elem in source_chunk_list[node['child_id']]:
                    if elem == node['child_id']:
                        dep_body += " " + str(word_map_list[node['parent_id']]) + "-" + str(word_map_list[node['child_id']])
                        rel_body += " " + node['label']
                    else:
                        dep_body += " " + str(word_map_list[node['parent_id']]) + "-" + str(word_map_list[elem])
                        rel_body += " nummod"
        dep_body += " 0-" + str(word_pos)
        rel_body += " </s>"
        for word in sent["compression_untransformed"]["edge"]:
            word_id = word["child_id"]
            selected_word_list[word_id] = 1
            last_selected_edge_id = word_id
        # for large data
        if args.conv_large == True and (end_char == "." or end_char == ","):
            selected_id = last_selected_edge_id
            for word_id in range(last_selected_edge_id, word_list[-1] + 1):
                if word_id in word_str_list and word_str_list[word_id] == end_char:
                    selected_id = word_id
            selected_word_list[selected_id] = 1
        for word_id in word_list:
            label_body += " "
            if word_id in selected_word_list:
                label_body += "1"
            else:
                label_body += "0"
        label_body += " </s>"
        
        source_sent += "\n"
        label_body += "\n"
        dep_body += "\n"
        rel_body += "\n"
        pos_body += "\n"

        if len(source_sent.split(" ")) > args.max_length:
            continue
        
        s_f.writelines(source_sent)
        l_f.writelines(label_body)
        d_f.writelines(dep_body)
        r_f.writelines(rel_body)
        p_f.writelines(pos_body)

s_f.close()
l_f.close()
d_f.close()
r_f.close()
p_f.close()
