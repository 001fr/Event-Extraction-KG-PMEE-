# -*- encoding:utf -*-
"""
  This script provides an K-BERT example for NER.
"""
import random
import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import  BertAdam
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.model_saver import save_model
import numpy as np
from collections import defaultdict
from brain import KnowledgeGraph
import torch.nn.functional as F
from torchcrf import CRF
import time
'''
残差网络层
'''
###########################################################
# class ResidualBlock(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(ResidualBlock, self).__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, input_size)
#
#     def forward(self, x):
#         out = F.relu(self.linear1(x))
#         out = self.linear2(out)
#         out = F.relu(out + x)  # Adding residual connection
#         return out
##########################################################
'''
自注意力
'''
###########################################################
class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        query = self.query(input)
        key = self.key(input)
        value = self.value(input)

        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = self.softmax(attention_weights)

        output = torch.matmul(attention_weights, value)
        return output
###########################################################
class BertTagger(nn.Module):
    def __init__(self, args, model):
        super(BertTagger, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.labels_num = args.labels_num
        # BiLSTM Layer
        self.bigru = nn.GRU(args.hidden_size, args.hidden_size // 2, num_layers=2, bidirectional=True,
                              batch_first=True)
        # self.attention = SelfAttentionLayer(args.hidden_size)
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        # self.softmax = nn.LogSoftmax(dim=-1)
        self.crf = CRF(self.labels_num, batch_first=True)
    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size x seq_length]
            mask: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        output = self.encoder(emb, mask, vm)
        # BiLSTM
        output, _ = self.bigru(output)
        # output = self.attention(output)
        # Target.
        output = self.output_layer(output)
        # Apply CRF.
        loss = -self.crf(output, label, mask=mask.byte())  # CRF计算负对数似然损失，注意num_tags和targets对齐
        predict = self.crf.decode(output, mask=mask.byte())

        predict = torch.tensor(predict, dtype=torch.long, device=label.device)

        # Compute the number of correct predictions.
        correct = (predict == label).float().sum()

        return loss, correct, predict, label
        # output = output.contiguous().view(-1, self.labels_num)
        # output = self.softmax(output)
        #
        # label = label.contiguous().view(-1, 1)
        # label_mask = (label > 0).float().to(torch.device(label.device))
        # one_hot = torch.zeros(label_mask.size(0), self.labels_num). \
        #     to(torch.device(label.device)). \
        #     scatter_(1, label, 1.0)
        #
        # numerator = -torch.sum(output * one_hot, 1)
        # label_mask = label_mask.contiguous().view(-1)
        # label = label.contiguous().view(-1)
        # numerator = torch.sum(label_mask * numerator)
        # denominator = torch.sum(label_mask) + 1e-6
        # loss = numerator / denominator
        # predict = output.argmax(dim=-1)
        # correct = torch.sum(
        #     label_mask * (predict.eq(label)).float()
        # )
        #
        # return loss, correct, predict, label


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default="D:/pre-trained models/ckbert-base/uer1-pytorch_model.bin", type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/mil_output_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    # parser.add_argument("--train_path", type=str, default="D:/MYstudy/data/ace_2005_td_v7/role/train_role.tsv",
    #                     help="Path of the trainset.")
    # parser.add_argument("--dev_path", type=str, default="D:/MYstudy/data/ace_2005_td_v7/role/dev_role.tsv",
    #                     help="Path of the devset.")
    # parser.add_argument("--test_path", type=str, default="D:/MYstudy/data/ace_2005_td_v7/role/test_role.tsv",
    #                     help="Path of the testset.")
    parser.add_argument("--train_path", type=str, default="C:/Users/FR/Desktop/roleNER/1.txt",
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, default="C:/Users/FR/Desktop/roleNER/2.txt",
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, default="C:/Users/FR/Desktop/roleNER/3.txt",
                        help="Path of the testset.")
    # parser.add_argument("--train_path", type=str, default="C:/Users/FR/Desktop/nan/bio-train.tsv",
    #                     help="Path of the trainset.")
    # parser.add_argument("--dev_path", type=str, default="C:/Users/FR/Desktop/nan/bio-dev.tsv",
    #                     help="Path of the devset.")
    # parser.add_argument("--test_path", type=str, default='C:/Users/FR/Desktop/nan/bio-test.tsv')
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch_size.")
    parser.add_argument("--seq_length", default=28, type=int,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    
    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=50,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # kg
    parser.add_argument("--kg_name", default="./brain/kgs/none.spo", help="KG name or path")
    # parser.add_argument("--no_vm", action="store_true",default=True, help="Disable the visible_matrix")
    args = parser.parse_args()

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    labels_map = {"[PAD]": 0, "[ENT]": 1}
    # labels_map = {}
    begin_ids = []

    # Find tagging labels
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            labels = line.strip().split("\t")[1].split()
            for l in labels:
                if l not in labels_map:
                    if l.startswith("B") or l.startswith("S"):
                        begin_ids.append(len(labels_map))
                    labels_map[l] = len(labels_map)
    
    print("Labels: ", labels_map)
    args.labels_num = len(labels_map)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=False)

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build sequence labeling model.
    model = BertTagger(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)

    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size, :]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vm_ids_batch = vm_ids[i*batch_size: (i+1)*batch_size, :, :]
            tag_ids_batch = tag_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:, :]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vm_ids_batch = vm_ids[instances_num//batch_size*batch_size:, :, :]
            tag_ids_batch = tag_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch

    # Read dataset.
    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            f.readline()
            tokens, labels = [], []
            for line_id, line in enumerate(f):
                tokens, labels = line.strip().split("\t")

                text = ''.join(tokens.split(" "))
                tokens, pos, vm, tag = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)

                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                tag = tag[0]

                tokens = [vocab.get(t) for t in tokens]
                labels = [labels_map[l] for l in labels.split(" ")]
                mask = [1] * len(tokens)

                new_labels = []
                j = 0
                for i in range(len(tokens)):
                    if tag[i] == 0 and tokens[i] != PAD_ID:
                        new_labels.append(labels[j])
                        j += 1
                    elif tag[i] == 1 and tokens[i] != PAD_ID:  # 是添加的实体
                        new_labels.append(labels_map['[ENT]'])
                    else:
                        new_labels.append(labels_map[PAD_TOKEN])
                dataset.append([tokens, new_labels, mask, pos, vm, tag])
        
        return dataset

    def read_text_labels(path):
        text_labels = []
        with open(path, mode="r", encoding="utf-8") as f:
            f.readline()
            tokens, labels = [], []
            for line_id, line in enumerate(f):
                tokens, labels = line.strip().split("\t")

                text = ''.join(tokens.split(" "))
                text_labels.append([tokens, labels])

        return text_labels

    # Evaluation function.
    def evaluate(args, is_test):
        if is_test:
            dataset = read_dataset(args.test_path)
            # text_labels = read_text_labels(args.test_path)
        else:
            dataset = read_dataset(args.dev_path)
            # text_labels = read_text_labels(args.dev_path)
        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([sample[3] for sample in dataset])
        vm_ids = torch.BoolTensor([sample[4] for sample in dataset])
        tag_ids = torch.LongTensor([sample[5] for sample in dataset])

        instances_num = input_ids.size(0)
        batch_size = args.batch_size

        if is_test:
            print("Batch size: ", batch_size)
            print("The number of test instances:", instances_num)
 
        correct = 0
        gold_entities_num = 0
        pred_entities_num = 0
        # Validation_loss_value =[]
        ##########
        # total_eval_loss = 0
        ##########
        confusion = torch.zeros(len(labels_map), len(labels_map), dtype=torch.long)
        model.eval()
        # for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids)):
        #
        #     input_ids_batch = input_ids_batch.to(device)
        #     label_ids_batch = label_ids_batch.to(device)
        #     mask_ids_batch = mask_ids_batch.to(device)
        #     pos_ids_batch = pos_ids_batch.to(device)
        #     tag_ids_batch = tag_ids_batch.to(device)
        #     vm_ids_batch = vm_ids_batch.long().to(device)
        for batch in batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids):
            input_ids_batch = batch[0].to(device)
            label_ids_batch = batch[1].to(device)
            mask_ids_batch = batch[2].to(device)
            pos_ids_batch = batch[3].to(device)
            vm_ids_batch = batch[4].long().to(device)
            tag_ids_batch = batch[5].to(device)

            loss, _, pred, gold = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)
            gold = gold.view(-1)
            pred = pred.view(-1)
            # #####################
            # # 如果是验证集，累加loss
            # if not is_test:
            #     total_eval_loss += loss.item()
            # ######################## gold.size()[0] pred.size()[0]
            for j in range(gold.size(0)):
                if gold[j].item() in begin_ids:
                    gold_entities_num += 1
 
            for j in range(pred.size(0)):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"]:
                    pred_entities_num += 1

            pred_entities_pos = []
            gold_entities_pos = []
            start, end = 0, 0

            for j in range(gold.size(0)):
                if gold[j].item() in begin_ids:
                    start = j
                    for k in range(j+1, gold.size(0)):
                        
                        if gold[k].item() == labels_map['[ENT]']:
                            continue

                        if gold[k].item() == labels_map["[PAD]"] or gold[k].item() == labels_map["O"] or gold[k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = gold.size(0) - 1
                    gold_entities_pos.append((start, end))
            for j in range(pred.size(0)):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"] and gold[j].item() != labels_map["[ENT]"]:
                    start = j
                    for k in range(j+1, pred.size(0)):

                        if gold[k].item() == labels_map['[ENT]']:
                            continue

                        if pred[k].item() == labels_map["[PAD]"] or pred[k].item() == labels_map["O"] or pred[k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = pred.size(0) - 1
                    pred_entities_pos.append((start, end))
            for entity in pred_entities_pos:
                if entity not in gold_entities_pos:
                    continue
                else: 
                    correct += 1

        print("Report precision, recall, and f1:")
        # p = correct/pred_entities_num
        # r = correct/gold_entities_num
        # f1 = 2*p*r/(p+r)
        if pred_entities_num == 0:
            p = 0  # 或者其他不会影响结果的值
        else:
            p = correct / pred_entities_num

        if gold_entities_num == 0:
            r = 0  # 或者其他不会影响结果的值
        else:
            r = correct / gold_entities_num

        # 确保 p 和 r 都不为零，再计算 F1
        if p + r > 0:
            f1 = 2 * (p * r) / (p + r)
        else:
            f1 = 0
        print("{:.4f}, {:.4f}, {:.4f}".format(p,r,f1))
        # if is_test:
        #     entity_level_results = get_result_entity_level(gold_show, pred_show, sort_labels=None, digits=4,
        #                                                return_avg_type='macro')
        #     print(entity_level_results)
        # batch_loader_list = list(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids))
        # print("batch_loader_list-001",len(batch_loader_list))
        # print("total_eval_loss",total_eval_loss)
        # if not is_test:
        #     Validation_loss=total_eval_loss / len(batch_loader_list)
        #     print("Validation loss: {:.3f}".format(Validation_loss))
        #     Validation_loss_value.append(round(Validation_loss, 3))
        #     print("Validation_loss_value", Validation_loss_value)
        return f1


    # Training phase.
    print("Start training.")
    instances = read_dataset(args.train_path)
    input_ids = torch.LongTensor([ins[0] for ins in instances])
    label_ids = torch.LongTensor([ins[1] for ins in instances])
    mask_ids = torch.LongTensor([ins[2] for ins in instances])
    pos_ids = torch.LongTensor([ins[3] for ins in instances])
    vm_ids = torch.BoolTensor([ins[4] for ins in instances])
    tag_ids = torch.LongTensor([ins[5] for ins in instances])

    instances_num = input_ids.size(0)
    batch_size = args.batch_size
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    f1 = 0.0
    best_f1 = 0.0
    # Training_loss_value=[]
    for epoch in range(1, args.epochs_num+1):
        start_time = time.time()  # 记录epoch开始时间
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)

            loss, _, _, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.

            loss.backward()
            optimizer.step()

        end_time = time.time()  # 记录epoch结束时间
        epoch_duration = end_time - start_time  # 计算epoch持续时间
        print("Epoch [{}] finished, Duration: {:.2f} seconds.".format(epoch, epoch_duration))
        # batch_loader_list = list(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids))
        # print("batch_loader_list-002", len(batch_loader_list))
        # print("total_loss", total_loss)
        # Training_loss=total_loss / len(batch_loader_list)
        # print("Training loss: {:.3f}".format(Training_loss))
        # Training_loss_value.append(round(Training_loss, 3))
        # print("Training_loss_value",Training_loss_value)
        # Evaluation phase.
        print("Start evaluate on dev dataset.")
        f1 = evaluate(args, False)
        print("Start evaluation on test dataset.")
        evaluate(args, True)

        if f1 > best_f1:
            best_f1 = f1
            save_model(model, args.output_model_path)
        else:
            continue

    # Evaluation phase.
    print("Final evaluation on test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))

    evaluate(args, True)

if __name__ == "__main__":
    main()
