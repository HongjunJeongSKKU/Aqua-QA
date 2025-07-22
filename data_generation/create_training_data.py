import pandas as pd
import numpy as np
from ast import literal_eval
import json
import click
import os


def process_single_head(df_column):
    return df_column.dropna().apply(lambda x: [x[0]]).values.tolist()


def process_double_head(df_column, indices=((0, 0), (1, 0))):
    data = df_column.dropna()
    heads = [data.apply(lambda x: [x[i][j]]).values.tolist() for i, j in indices]
    return [[*a, *b] for a, b in zip(*heads)]


def process_triple_head(df_column):
    data = df_column.dropna()
    heads = [data.apply(lambda x: [x[i][0]]).values.tolist() for i in range(3)]
    return [[*a, *b, *c] for a, b, c in zip(*heads)]


def process_inp_head(df_column):
    data = df_column.dropna()
    head1 = data.apply(lambda x: [x[0][0][0]]).values.tolist()
    head2 = data.apply(lambda x: [x[0][1][0]]).values.tolist()
    return [[*a, *b] for a, b in zip(head1, head2)]


def process_heads(d_head, mode):
    heads = []

    patterns = {
        "single": ["1p_origin", "2p_origin", "3p_origin"],
        "double": ["2i_origin", "2in_origin", "pin_origin", "pni_origin", "2u_origin"],
        "triple": ["3i_origin", "3in_origin"],
        "empty": ["ip_origin", "pi_origin", "up_origin"],
    }

    for pattern in d_head.columns:
        if pattern in patterns["single"]:
            heads.extend(process_single_head(d_head[pattern]))
        elif pattern in patterns["double"]:
            heads.extend(process_double_head(d_head[pattern]))
        elif pattern in patterns["triple"]:
            heads.extend(process_triple_head(d_head[pattern]))
        elif pattern == "inp_origin":
            heads.extend(process_inp_head(d_head["inp_origin"]))
        elif mode in ["valid", "test"] and pattern in patterns["empty"]:
            heads.extend([[-1] for _ in range(len(d_head[pattern].dropna()))])

    return heads


def process_single_relation(df_column):
    return df_column.dropna().apply(lambda x: [*x[1]]).values.tolist()


def process_double_relation_type1(df_column):
    # 2i, 2u 패턴용 (x[i][1])
    data = df_column.dropna()
    rel1 = data.apply(lambda x: [*x[0][1]]).values.tolist()
    rel2 = data.apply(lambda x: [*x[1][1]]).values.tolist()
    return [[*a, *b] for a, b in zip(rel1, rel2)]


def process_double_relation_type2(df_column):
    # 2in 패턴용 (x[i][1][0])
    data = df_column.dropna()
    rel1 = data.apply(lambda x: [x[0][1][0]]).values.tolist()
    rel2 = data.apply(lambda x: [x[1][1][0]]).values.tolist()
    return [[*a, *b] for a, b in zip(rel1, rel2)]


def process_triple_relation_type1(df_column):
    # 3i 패턴용
    data = df_column.dropna()
    rels = [data.apply(lambda x: [*x[i][1]]).values.tolist() for i in range(3)]
    return [[*a, *b, *c] for a, b, c in zip(*rels)]


def process_triple_relation_type2(df_column):
    # 3in 패턴용
    data = df_column.dropna()
    rels = [data.apply(lambda x: [x[i][1][0]]).values.tolist() for i in range(3)]
    return [[*a, *b, *c] for a, b, c in zip(*rels)]


def process_inp_relation(df_column):
    data = df_column.dropna()
    rel1 = data.apply(lambda x: [x[0][0][1][0]]).values.tolist()
    rel2 = data.apply(lambda x: [x[0][1][1][0], x[1][0]]).values.tolist()
    return [[*a, *b] for a, b in zip(rel1, rel2)]


def process_pin_relation(df_column):
    data = df_column.dropna()
    rel1 = data.apply(lambda x: [*x[0][1]]).values.tolist()
    rel2 = data.apply(lambda x: [x[1][1][0]]).values.tolist()
    return [[*a, *b] for a, b in zip(rel1, rel2)]


def process_pni_relation(df_column):
    data = df_column.dropna()
    rel1 = data.apply(lambda x: [*x[0][1][:-1]]).values.tolist()
    rel2 = data.apply(lambda x: [x[1][1][0]]).values.tolist()
    return [[*a, *b] for a, b in zip(rel1, rel2)]


def process_relations(d_head, mode):
    relations = []

    patterns = {
        "single": ["1p_origin", "2p_origin", "3p_origin"],
        "double_type1": ["2i_origin", "2u_origin"],
        "double_type2": ["2in_origin"],
        "triple_type1": ["3i_origin"],
        "triple_type2": ["3in_origin"],
        "empty": ["ip_origin", "pi_origin", "up_origin"],
    }

    for pattern in d_head.columns:
        if pattern in patterns["single"]:
            relations.extend(process_single_relation(d_head[pattern]))
        elif pattern in patterns["double_type1"]:
            relations.extend(process_double_relation_type1(d_head[pattern]))
        elif pattern in patterns["double_type2"]:
            relations.extend(process_double_relation_type2(d_head[pattern]))
        elif pattern in patterns["triple_type1"]:
            relations.extend(process_triple_relation_type1(d_head[pattern]))
        elif pattern in patterns["triple_type2"]:
            relations.extend(process_triple_relation_type2(d_head[pattern]))
        elif pattern == "inp_origin":
            relations.extend(process_inp_relation(d_head["inp_origin"]))
        elif pattern == "pin_origin":
            relations.extend(process_pin_relation(d_head["pin_origin"]))
        elif pattern == "pni_origin":
            relations.extend(process_pni_relation(d_head["pni_origin"]))
        elif mode in ["valid", "test"] and pattern in patterns["empty"]:
            relations.extend([-1] * len(d_head[pattern].dropna()))

    return relations


def map_relations(relations, id2rel):
    return [["-1"] if data == -1 else [id2rel[x] for x in data] for data in relations]


def split_relations(data):
    positive_relation = []
    negative_relation = []

    for q_type, path in zip(data["query_type"], data["paths"]):
        if q_type in ["2in", "3in", "pin"]:
            negative_relation.append([path[-1]])
            positive_relation.append(path[:-1])
        elif q_type in ["inp", "pni"]:
            negative_relation.append([path[1]])
            positive_relation.append([path[0], path[2]])
        else:
            negative_relation.append([])
            positive_relation.append(path)

    return positive_relation, negative_relation


def process_dataframe(data_nl, data, heads, mapped_relations, id2ent, mode):

    if mode == "train":
        df = pd.concat(
            [
                data_nl.iloc[:, 1::2].melt(var_name="query_type", value_name="query"),
                data.iloc[:, 23::2].melt(var_name="answer_dummy", value_name="answer"),
            ],
            axis=1,
        ).dropna(subset=["query"])
        df.drop(["answer_dummy"], axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = df.applymap(lambda x: "{}" if isinstance(x, float) else x)
        df["answer"] = df["answer"].apply(literal_eval)
        df["answer"] = df["answer"].apply(lambda x: [id2ent[z] for z in x])

    elif mode in ["valid", "test"]:
        df = pd.concat(
            [
                data_nl.iloc[:, 1::2].melt(var_name="query_type", value_name="query"),
                data.iloc[:, 29::3].melt(
                    var_name="easy_ans_dummy", value_name="easy_ans"
                ),
                data.iloc[:, 30::3].melt(
                    var_name="hard_ans_dummy", value_name="hard_ans"
                ),
            ],
            axis=1,
        ).dropna(subset=["query"])
        df.drop(["easy_ans_dummy", "hard_ans_dummy"], axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = df.applymap(lambda x: "{}" if isinstance(x, float) else x)
        for col in ["easy_ans", "hard_ans"]:
            df[col] = df[col].apply(literal_eval)
            df[col] = df[col].apply(lambda x: [id2ent[z] for z in x])

    df["q_entity"] = [[id2ent[z] for z in x if z != -1] for x in heads]

    df["query_type"] = df["query_type"].str.split("_").str[0]
    df["paths"] = mapped_relations

    return df


@click.command()
@click.option("--dataset", default="CODEX")
@click.option("--gen_train", is_flag=True, default=False)
@click.option("--gen_valid", is_flag=True, default=False)
@click.option("--gen_test", is_flag=True, default=False)
@click.option("--llm_model", default="llama")
def main(dataset, gen_train, gen_valid, gen_test, llm_model):
    if gen_train:
        mode = "train"
    elif gen_valid:
        mode = "valid"
    elif gen_test:
        mode = "test"
    else:
        print("Set the mode (--gen_train, --gen_valid, --gen_test)")
        exit(1)
    data = pd.read_csv(f"data/{dataset}/{mode}.csv")
    data_nl = pd.read_csv(f"data/{dataset}/{llm_model}_{mode}_nl.csv")
    # with open(f"data/{dataset}/ent2txt.json") as f:
    #     ent2txt = json.loads(f.read())
    # with open(f"data/{dataset}/rel2txt.json") as f:
    #     rel2txt = json.loads(f.read())
    id2ent = pd.read_pickle(f"data/{dataset}/id2ent.pkl")
    id2rel = pd.read_pickle(f"data/{dataset}/id2rel.pkl")

    if mode == "train":
        d_head = data.iloc[:, 22::2]
    elif mode in ["valid", "test"]:
        d_head = data.iloc[:, 28::3]
    d_head = d_head.applymap(lambda x: literal_eval(x) if pd.notna(x) else x)
    heads = process_heads(d_head, mode)
    relations = process_relations(d_head, mode)

    mapped_relations = map_relations(relations, id2rel)
    data = process_dataframe(data_nl, data, heads, mapped_relations, id2ent, mode)
    positive_relation, negative_relation = split_relations(data)
    data["negative_path"] = negative_relation
    data["positive_path"] = positive_relation

    train = pd.read_parquet(f"../data/{dataset}/train-drop_dup.parquet")
    valid = pd.read_parquet(f"../data/{dataset}/valid-drop_dup.parquet")
    test = pd.read_parquet(f"../data/{dataset}/test-drop_dup.parquet")

    reasoning_data = pd.concat([train, valid, test])
    data_filtered = data[~data["query"].isin(reasoning_data["query"])]

    file_path = f"data/output/{dataset}/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    data_filtered.to_parquet(f"data/output/{dataset}/synthetic_drop_dup.parquet")


if __name__ == "__main__":
    main()
