import pandas as pd
import numpy as np
import json
import click

QUERY_STRUCTURE = {
    ("e", ("r",)): "1p",
    ("e", ("r", "r")): "2p",
    ("e", ("r", "r", "r")): "3p",
    (("e", ("r",)), ("e", ("r",))): "2i",
    (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
    ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
    (("e", ("r", "r")), ("e", ("r",))): "pi",
    (("e", ("r",)), ("e", ("r", "n"))): "2in",
    (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
    ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
    (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
    (("e", ("r", "r", "n")), ("e", ("r",))): "pni",
    (("e", ("r",)), ("e", ("r",)), ("u",)): "2u-DNF",
    ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up-DNF",
    # ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
    # ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
}


def process_train_data(train_queries, train_answers):
    QUERY_TYPES = [
        "p1",
        "p2",
        "p3",
        "i2",
        "i3",
        "in2",
        "in3",
        "pin",
        "pni",
        "inp",
        "u2",
    ]

    query_keys = list(train_queries.keys())
    idx_order = range(len(QUERY_TYPES))

    train_queries = {query_keys[i]: train_queries[query_keys[i]] for i in idx_order}
    query_dfs = {}
    for i, query_type in enumerate(QUERY_TYPES):
        current_queries = train_queries[query_keys[i]]
        query_dfs[query_type] = pd.DataFrame(
            {
                "q": list(current_queries),
                "answers": [
                    (
                        list(train_answers[q])
                        if isinstance(train_answers[q], set)
                        else train_answers[q]
                    )
                    for q in current_queries
                ],
            }
        )

    return query_dfs, train_queries


def process_eval_data(eval_queries, eval_easy_ans, eval_hard_ans):
    QUERY_TYPES = [
        "p1",
        "p2",
        "p3",
        "i2",
        "i3",
        "pi",
        "ip",
        "in2",
        "in3",
        "pin",
        "pni",
        "inp",
        "u2",
        "up",
    ]

    query_keys = list(eval_queries.keys())
    idx_order = range(len(QUERY_TYPES))

    processed_queries = {
        query_keys[idx]: eval_queries[query_keys[idx]] for idx in idx_order
    }
    query_dfs = {}
    query_keys = list(processed_queries.keys())

    for i, query_type in enumerate(QUERY_TYPES):
        current_queries = processed_queries[query_keys[i]]
        data = {"q": [], "easy_ans": [], "hard_ans": []}

        for q in current_queries:
            data["q"].append(q)
            if q in eval_easy_ans:
                data["easy_ans"].append(
                    list(eval_easy_ans[q]) if len(eval_hard_ans[q]) > 0 else np.nan
                )
                data["hard_ans"].append(
                    list(eval_hard_ans[q]) if len(eval_hard_ans[q]) > 0 else np.nan
                )
        query_dfs[query_type] = pd.DataFrame(data)

    return query_dfs, eval_queries


def tuple2fol(mode, dataframe, queries, ent2txt, id2ent, id2rel):

    if mode == "train":
        p1, p2, p3, i2, i3, in2, in3, pin, pni, inp, u2 = [
            dataframe[name] for name in dataframe.keys()
        ]
        queries = [x for x in queries if x not in ["pi", "ip", "up"]]
    else:
        p1, p2, p3, i2, i3, pi, ip, in2, in3, pin, pni, inp, u2, up = [
            dataframe[name] for name in dataframe.keys()
        ]

    df = pd.DataFrame()
    for q_type in queries:
        if "1p" == q_type:
            er = pd.DataFrame(tuple(p1["q"].tolist()), columns=["e", "r"])
            er["e"] = er["e"].apply(lambda x: ent2txt[id2ent[x]])
            er["r"] = er["r"].apply(lambda x: (id2rel[x[0]],))
            er_f = [(er["e"][i], (er["r"][i])) for i in range(len(er))]
            er_df = pd.DataFrame({"1p": [item for item in er_f]})
            er_FOL = pd.DataFrame(
                {
                    "1p_FOL": [
                        "𝑞 = X? . ∃X: ("
                        + er.loc[i]["r"][0]
                        + "('"
                        + er.loc[i]["e"]
                        + "', X?))"
                        for i in range(len(er))
                    ]
                }
            )
            df = pd.concat([df, er_df, er_FOL], axis=1)
        elif "2p" == q_type:
            err = pd.DataFrame(tuple(p2["q"].tolist()), columns=["e", "rr"])
            err["e"] = err["e"].apply(lambda x: ent2txt[id2ent[x]])
            err["rr"] = err["rr"].apply(lambda x: (id2rel[x[0]], id2rel[x[1]]))
            err_f = [(err["e"][i], (err["rr"][i])) for i in range(len(err))]
            err_df = pd.DataFrame({"2p": [item for item in err_f]})
            err_FOL = pd.DataFrame(
                {
                    "2p_FOL": [
                        "𝑞 = Y? . ∃X ∃Y: ("
                        + err.loc[i]["rr"][0]
                        + "('"
                        + err.loc[i]["e"]
                        + "', X) ∧ "
                        + err.loc[i]["rr"][1]
                        + "(X, Y?))"
                        for i in range(len(err))
                    ]
                }
            )
            df = pd.concat([df, err_df, err_FOL], axis=1)
        elif "3p" == q_type:
            errr = pd.DataFrame(tuple(p3["q"].tolist()), columns=["e", "rrr"])
            errr["e"] = errr["e"].apply(lambda x: ent2txt[id2ent[x]])
            errr["rrr"] = errr["rrr"].apply(
                lambda x: (id2rel[x[0]], id2rel[x[1]], id2rel[x[2]])
            )
            errr_f = [(errr["e"][i], (errr["rrr"][i])) for i in range(len(errr))]
            errr_df = pd.DataFrame({"3p": [item for item in errr_f]})
            errr_FOL = pd.DataFrame(
                {
                    "3p_FOL": [
                        "𝑞 = Z? . ∃X ∃Y ∃Z: ("
                        + errr.loc[i]["rrr"][0]
                        + "('"
                        + errr.loc[i]["e"]
                        + "', X) ∧ "
                        + errr.loc[i]["rrr"][1]
                        + "(X, Y)) ∧ "
                        + errr.loc[i]["rrr"][2]
                        + "(Y, Z?))"
                        for i in range(len(errr))
                    ]
                }
            )
            df = pd.concat([df, errr_df, errr_FOL], axis=1)
        elif "2i" == q_type:
            erer = pd.DataFrame(tuple(i2["q"].tolist()), columns=["er1", "er2"])
            erer["er1"] = erer["er1"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            erer["er2"] = erer["er2"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            erer_f = [(erer["er1"][i], erer["er2"][i]) for i in range(len(erer))]
            erer_df = pd.DataFrame({"2i": [item for item in erer_f]})
            erer_FOL = pd.DataFrame(
                {
                    "2i_FOL": [
                        "𝑞 = X? . ∃X : ("
                        + erer.loc[i]["er1"][1][0]
                        + "('"
                        + erer.loc[i]["er1"][0]
                        + "', X?) ∧ "
                        + erer.loc[i]["er2"][1][0]
                        + "('"
                        + erer.loc[i]["er2"][0]
                        + "', X?))"
                        for i in range(len(erer))
                    ]
                }
            )
            df = pd.concat([df, erer_df, erer_FOL], axis=1)
        elif "3i" == q_type:
            ererer = pd.DataFrame(
                tuple(i3["q"].tolist()), columns=["er1", "er2", "er3"]
            )
            ererer["er1"] = ererer["er1"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            ererer["er2"] = ererer["er2"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            ererer["er3"] = ererer["er3"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            ererer_f = [
                (ererer["er1"][i], (ererer["er2"][i]), (ererer["er3"][i]))
                for i in range(len(ererer))
            ]
            ererer_df = pd.DataFrame({"3i": [item for item in ererer_f]})
            ererer_FOL = pd.DataFrame(
                {
                    "3i_FOL": [
                        "𝑞 = X? . ∃X: ("
                        + ererer.loc[i]["er1"][1][0]
                        + "('"
                        + ererer.loc[i]["er1"][0]
                        + "', X?) ∧ "
                        + ererer.loc[i]["er2"][1][0]
                        + "('"
                        + ererer.loc[i]["er2"][0]
                        + "', X?) ∧ "
                        + ererer.loc[i]["er3"][1][0]
                        + "('"
                        + ererer.loc[i]["er3"][0]
                        + "', X?))"
                        for i in range(len(ererer))
                    ]
                }
            )
            df = pd.concat([df, ererer_df, ererer_FOL], axis=1)
        elif "2in" == q_type:
            erern = pd.DataFrame(tuple(in2["q"].tolist()), columns=["er", "ern"])
            erern["er"] = erern["er"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            erern["ern"] = erern["ern"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]], x[1][1]))
            )
            erern_f = [(erern["er"][i], (erern["ern"][i])) for i in range(len(erern))]
            erern_df = pd.DataFrame({"2in": [item for item in erern_f]})
            erern_FOL = pd.DataFrame(
                {
                    "2in_FOL": [
                        "𝑞 = X? . ∃X: ("
                        + erern.loc[i]["er"][1][0]
                        + "('"
                        + erern.loc[i]["er"][0]
                        + "', X?) ∧ ¬"
                        + erern.loc[i]["ern"][1][0]
                        + "('"
                        + erern.loc[i]["ern"][0]
                        + "', X?))"
                        for i in range(len(erern))
                    ]
                }
            )
            df = pd.concat([df, erern_df, erern_FOL], axis=1)
        elif "3in" == q_type:
            ererern = pd.DataFrame(
                tuple(in3["q"].tolist()), columns=["er1", "er2", "ern"]
            )
            ererern["er1"] = ererern["er1"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            ererern["er2"] = ererern["er2"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            ererern["ern"] = ererern["ern"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            ererern_f = [
                (ererern["er1"][i], (ererern["er2"][i]), (ererern["ern"][i]))
                for i in range(len(ererern))
            ]
            ererern_df = pd.DataFrame({"3in": [item for item in ererern_f]})
            ererern_FOL = pd.DataFrame(
                {
                    "3in_FOL": [
                        "𝑞 = X? . ∃X: ("
                        + ererern.loc[i]["er1"][1][0]
                        + "('"
                        + ererern.loc[i]["er1"][0]
                        + "', X?) ∧ "
                        + ererern.loc[i]["er2"][1][0]
                        + "('"
                        + ererern.loc[i]["er2"][0]
                        + "', X?) ∧ ¬"
                        + ererern.loc[i]["ern"][1][0]
                        + "('"
                        + ererern.loc[i]["ern"][0]
                        + "', X?))"
                        for i in range(len(ererern))
                    ]
                }
            )
            df = pd.concat([df, ererern_df, ererern_FOL], axis=1)
        elif "inp" == q_type:
            erernr = pd.DataFrame(tuple(inp["q"].tolist()), columns=["erern", "r"])
            erernr["erern"] = erernr["erern"].apply(
                lambda x: (
                    (ent2txt[id2ent[x[0][0]]], (id2rel[x[0][1][0]],)),
                    (ent2txt[id2ent[x[1][0]]], (id2rel[x[1][1][0]], -2)),
                )
            )
            erernr["r"] = erernr["r"].apply(lambda x: ((id2rel[x[0]],)))
            erernr_f = [
                (erernr["erern"][i], (erernr["r"][i])) for i in range(len(erernr))
            ]
            erernr_df = pd.DataFrame({"inp": [item for item in erernr_f]})
            erernr_FOL = pd.DataFrame(
                {
                    "inp_FOL": [
                        "𝑞 = Y? . ∃X ∃Y : (("
                        + erernr.loc[i]["erern"][0][1][0]
                        + "('"
                        + erernr.loc[i]["erern"][0][0]
                        + "', X) ∧ ¬"
                        + erernr.loc[i]["erern"][1][1][0]
                        + "('"
                        + erernr.loc[i]["erern"][1][0]
                        + "', X)) ∧ "
                        + erernr.loc[i]["r"][0]
                        + "(X, Y?))"
                        for i in range(len(erernr))
                    ]
                }
            )
            df = pd.concat([df, erernr_df, erernr_FOL], axis=1)
        elif "pin" == q_type:
            errern = pd.DataFrame(tuple(pin["q"].tolist()), columns=["err", "ern"])
            errern["err"] = errern["err"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]], id2rel[x[1][1]]))
            )
            errern["ern"] = errern["ern"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]], -2))
            )
            errern_f = [
                (errern["err"][i], (errern["ern"][i])) for i in range(len(errern))
            ]
            errern_df = pd.DataFrame({"pin": [item for item in errern_f]})
            errern_FOL = pd.DataFrame(
                {
                    "pin_FOL": [
                        "𝑞 = Y? . ∃X ∃Y : (("
                        + errern.loc[i]["err"][1][0]
                        + "('"
                        + errern.loc[i]["err"][0]
                        + "', X) ∧ "
                        + errern.loc[i]["err"][1][1]
                        + "(X, Y?)) ∧ ¬"
                        + errern.loc[i]["ern"][1][0]
                        + "('"
                        + errern.loc[i]["ern"][0]
                        + "', Y?))"
                        for i in range(len(errern))
                    ]
                }
            )
            df = pd.concat([df, errern_df, errern_FOL], axis=1)
        elif "pni" == q_type:
            errner = pd.DataFrame(tuple(pni["q"].tolist()), columns=["errn", "er"])
            errner["errn"] = errner["errn"].apply(
                lambda x: (
                    ent2txt[id2ent[x[0]]],
                    (id2rel[x[1][0]], id2rel[x[1][1]], -2),
                )
            )
            errner["er"] = errner["er"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]], -2))
            )
            errner_f = [
                (errner["errn"][i], (errner["er"][i])) for i in range(len(errner))
            ]
            errner_df = pd.DataFrame({"pni": [item for item in errner_f]})
            errner_FOL = pd.DataFrame(
                {
                    "pni_FOL": [
                        "𝑞 = Y? . ∃X ∃Y : (("
                        + errner.loc[i]["errn"][1][0]
                        + "('"
                        + errner.loc[i]["errn"][0]
                        + "', X) ∧ ¬"
                        + errner.loc[i]["errn"][1][1]
                        + "(X, Y?)) ∧ "
                        + errner.loc[i]["er"][1][0]
                        + "('"
                        + errner.loc[i]["er"][0]
                        + "', Y?))"
                        for i in range(len(errner))
                    ]
                }
            )
            df = pd.concat([df, errner_df, errner_FOL], axis=1)
        elif "ip" == q_type:
            ererr = pd.DataFrame(tuple(ip["q"].tolist()), columns=["ererr", "r"])
            ererr["ererr"] = ererr["ererr"].apply(
                lambda x: (
                    (ent2txt[id2ent[x[0][0]]], (id2rel[x[0][1][0]],)),
                    (ent2txt[id2ent[x[1][0]]], (id2rel[x[1][1][0]],)),
                )
            )
            ererr["r"] = ererr["r"].apply(lambda x: ((id2rel[x[0]],)))
            ererr_f = [(ererr["ererr"][i], (ererr["r"][i])) for i in range(len(ererr))]
            ererr_df = pd.DataFrame({"ip": [item for item in ererr_f]})
            ererr_FOL = pd.DataFrame(
                {
                    "ip_FOL": [
                        "𝑞 = Y? . ∃X ∃Y : (("
                        + ererr.loc[i]["ererr"][0][1][0]
                        + "('"
                        + ererr.loc[i]["ererr"][0][0]
                        + "', X) ∧ "
                        + ererr.loc[i]["ererr"][1][1][0]
                        + "('"
                        + ererr.loc[i]["ererr"][1][0]
                        + "', X)) ∧ "
                        + ererr.loc[i]["r"][0]
                        + "(X, Y?))"
                        for i in range(len(ererr))
                    ]
                }
            )
            df = pd.concat([df, ererr_df, ererr_FOL], axis=1)
        elif "pi" == q_type:
            errer = pd.DataFrame(tuple(pi["q"].tolist()), columns=["err", "er"])
            errer["err"] = errer["err"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]], id2rel[x[1][1]]))
            )
            errer["er"] = errer["er"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            errer_f = [(errer["err"][i], (errer["er"][i])) for i in range(len(errer))]
            errer_df = pd.DataFrame({"pi": [item for item in errer_f]})
            errer_FOL = pd.DataFrame(
                {
                    "pi_FOL": [
                        "𝑞 = Y? . ∃X ∃Y : (("
                        + errer.loc[i]["err"][1][0]
                        + "('"
                        + errer.loc[i]["err"][0]
                        + "', X) ∧ "
                        + errer.loc[i]["err"][1][1]
                        + "(X, Y?)) ∧ "
                        + errer.loc[i]["er"][1][0]
                        + "('"
                        + errer.loc[i]["er"][0]
                        + "', Y?))"
                        for i in range(len(errer))
                    ]
                }
            )
            df = pd.concat([df, errer_df, errer_FOL], axis=1)
        elif "2u" == q_type:
            ereru = pd.DataFrame(tuple(u2["q"].tolist()), columns=["er1", "er2", "u"])
            ereru["er1"] = ereru["er1"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            ereru["er2"] = ereru["er2"].apply(
                lambda x: (ent2txt[id2ent[x[0]]], (id2rel[x[1][0]],))
            )
            ereru_f = [
                (ereru["er1"][i], ereru["er2"][i], (-1,)) for i in range(len(ereru))
            ]
            ereru_df = pd.DataFrame({"2u": [item for item in ereru_f]})
            ereru_FOL = pd.DataFrame(
                {
                    "2u_FOL": [
                        "𝑞 = X? . ∃X : ("
                        + ereru.loc[i]["er1"][1][0]
                        + "('"
                        + ereru.loc[i]["er1"][0]
                        + "', X?) ∨ "
                        + ereru.loc[i]["er2"][1][0]
                        + "('"
                        + ereru.loc[i]["er2"][0]
                        + "', X?))"
                        for i in range(len(ereru))
                    ]
                }
            )
            df = pd.concat([df, ereru_df, ereru_FOL], axis=1)
        elif "up" == q_type:
            ererur = pd.DataFrame(tuple(up["q"].tolist()), columns=["ereru", "r"])
            ererur["ereru"] = ererur["ereru"].apply(
                lambda x: (
                    (ent2txt[id2ent[x[0][0]]], (id2rel[x[0][1][0]],)),
                    (ent2txt[id2ent[x[1][0]]], (id2rel[x[1][1][0]],)),
                    (-1,),
                )
            )
            ererur["r"] = ererur["r"].apply(lambda x: ((id2rel[x[0]],)))
            ererur_f = [
                (ererur["ereru"][i], (ererur["r"][i])) for i in range(len(ererur))
            ]
            ererur_df = pd.DataFrame({"up": [item for item in ererur_f]})
            ererur_FOL = pd.DataFrame(
                [
                    "𝑞 = Y? . ∃X ∃Y : (("
                    + ererur.loc[i]["ereru"][0][1][0]
                    + "('"
                    + ererur.loc[i]["ereru"][0][0]
                    + "', X) ∨ "
                    + ererur.loc[i]["ereru"][1][1][0]
                    + "('"
                    + ererur.loc[i]["ereru"][1][0]
                    + "', X)) ∧ "
                    + ererur.loc[i]["r"][0]
                    + "(X, Y?))"
                    for i in range(len(ererur))
                ]
            )
            ererur_FOL.columns = ["up_FOL"]
            df = pd.concat([df, ererur_df, ererur_FOL], axis=1)

    if mode == "train":
        df_names = [p1, p2, p3, i2, i3, in2, in3, inp, pin, pni, u2]
        base_cols = df.columns[::2]
        processed_data = {}
        for df_name, base_col in zip(df_names, base_cols):
            processed_data.update(
                {
                    f"{base_col}_origin": df_name["q"],
                    f"{base_col}_ans": df_name["answers"],
                }
            )
        df = pd.concat([df, pd.DataFrame(processed_data)], axis=1)
    else:
        df_names = [p1, p2, p3, i2, i3, in2, in3, inp, pin, pni, ip, pi, u2, up]
        base_cols = df.columns[::2]

        processed_data = {}
        for df_name, base_col in zip(df_names, base_cols):
            processed_data.update(
                {
                    f"{base_col}_origin": df_name["q"],
                    f"{base_col}_easy_ans": df_name["easy_ans"],
                    f"{base_col}_hard_ans": df_name["hard_ans"],
                }
            )

        df = pd.concat([df, pd.DataFrame(processed_data)], axis=1)
    return df


@click.command()
@click.option("--dataset", default="CODEX")
@click.option("--gen_train", is_flag=True, default=False)
@click.option("--gen_valid", is_flag=True, default=False)
@click.option("--gen_test", is_flag=True, default=False)
@click.option("--query_type", default="1p.2p.3p.2i.3i.2in.3in.inp.pin.pni.ip.pi.2u.up")
def main(dataset, gen_train, gen_valid, gen_test, query_type):

    id2ent = pd.read_pickle(f"data/{dataset}/id2ent.pkl")
    id2rel = pd.read_pickle(f"data/{dataset}/id2rel.pkl")

    with open(f"data/{dataset}/ent2txt.json") as f:
        ent2txt = json.loads(f.read())

    if dataset == "CODEX":
        with open(f"data/{dataset}/rel2txt.json") as f:
            rel2txt = json.loads(f.read())
        id2rel = {k: rel2txt[v] for k, v in id2rel.items()}

    if gen_train:
        mode = "train"

        train_queries = pd.read_pickle(f"data/{dataset}/train-queries.pkl")
        train_answers = pd.read_pickle(f"data/{dataset}/train-answers.pkl")

        # 훈련 데이터 처리
        query_dfs, processed_queries = process_train_data(train_queries, train_answers)

        print(f"Query types count: {len(processed_queries)}")
        print(
            f"Query structures: {[QUERY_STRUCTURE[k] for k in processed_queries.keys()]}"
        )

    elif gen_valid or gen_test:
        mode = "valid" if gen_valid else "test"
        eval_queries = pd.read_pickle(f"data/{dataset}/{mode}-queries.pkl")
        eval_easy_answers = pd.read_pickle(f"data/{dataset}/{mode}-easy-answers.pkl")
        eval_hard_answers = pd.read_pickle(f"data/{dataset}/{mode}-hard-answers.pkl")

        query_dfs, processed_queries = process_eval_data(
            eval_queries, eval_easy_answers, eval_hard_answers
        )

        print(f"Query types count: {len(processed_queries)}")
        print(
            f"Query structures: {[QUERY_STRUCTURE[k] for k in processed_queries.keys()]}"
        )

    queries = query_type.split(".")
    fol_queries = tuple2fol(mode, query_dfs, queries, ent2txt, id2ent, id2rel)
    fol_queries.to_csv(f"data/{dataset}/{mode}.csv", index=False)


if __name__ == "__main__":
    main()
