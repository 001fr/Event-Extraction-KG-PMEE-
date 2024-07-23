# import csv
# import chardet
# def extract_triples(csv_file):
#     triples = []
#     # with open(csv_file, 'rb') as f:
#     #     result = chardet.detect(f.read())
#     # encoding = result['encoding']
#     with open(csv_file, mode='r', encoding="gbk", newline='') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             for key, value in row.items():
#                 if key == 'ID':  # ID不是谓词，我们稍后会用它作为主体
#                     subject = value
#                 elif value:  # 如果值存在（非空、非None、非空字符串等）
#                     predicate = key
#                     object_ = value
#                     triples.append((subject, predicate, object_))
#     return triples
#
#
# # 使用函数并打印结果
# csv_file = 'E:/Neo4j/neo4j-community-4.4.24-windows/neo4j-community-4.4.24/import/kg.csv'  # 替换为你的CSV文件名
# triples = extract_triples(csv_file)
# for triple in triples:
#     print(triple)

import pandas as pd


def extract_triples(xlsx_file):
    triples = []
    # 读取Excel文件
    df = pd.read_excel(xlsx_file)

    # 定义我们关心的列名
    columns_of_interest = ['名称', '国家', '类型']

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        subject = row['名称']
        if subject:  # 确保主体存在
            # 遍历我们关心的列
            for predicate in ['国家', '类型']:
                if predicate in columns_of_interest and pd.notnull(row[predicate]):
                    object_ = row[predicate]
                    triples.append((subject, predicate, object_))

    return triples


# 使用函数并打印结果
xlsx_file = 'E:/Neo4j/neo4j-community-4.4.24-windows/neo4j-community-4.4.24/import/kgdb.xlsx'  # 替换为你的Excel文件名
triples = extract_triples(xlsx_file)
# 将三元组保存为txt文件
output_txt_file = 'E:/Neo4j/neo4j-community-4.4.24-windows/neo4j-community-4.4.24/import/military.txt'
with open(output_txt_file, 'w', encoding='utf-8') as txtfile:
    for triple in triples:
        # 使用制表符分隔三元组
        txtfile.write('\t'.join(map(str, triple)) + '\n')  # 或者使用逗号 ','.join(map(str, triple)) + '\n'

print(f"三元组已保存到文件: {output_txt_file}")