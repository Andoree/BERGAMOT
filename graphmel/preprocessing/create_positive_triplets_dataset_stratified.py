import itertools
import logging
import os
import os.path
import random
from argparse import ArgumentParser
from typing import Dict
from typing import List, Set, Tuple

import pandas as pd
from tqdm import tqdm

from graphmel.utils.io import save_tuples, save_dict, save_node_id2terms_list
from graphmel.utils.io import write_strings, read_mrconso, read_mrrel
from graphmel.utils.umls2graph import get_concept_list_groupby_cui, extract_umls_oriented_edges_with_relations


def create_graph_files(mrconso_df: pd.DataFrame, mrrel_df: pd.DataFrame, rel2id: Dict[str, int],
                       cui2node_id: Dict[str, int], rela2id: Dict[str, int], output_node_id2terms_list_path: str,
                       output_node_id2cui_path: str, output_edges_path: str, output_rel2rel_id_path: str,
                       output_rela2rela_id_path, ignore_not_mapped_edges: bool):
    node_id2cui: Dict[int, str] = {node_id: cui for cui, node_id in cui2node_id.items()}
    node_id2terms_list = get_concept_list_groupby_cui(mrconso_df=mrconso_df, cui2node_id=cui2node_id)
    logging.info("Generating edges....")

    edges = extract_umls_oriented_edges_with_relations(mrrel_df=mrrel_df, cui2node_id=cui2node_id,
                                                       rel2rel_id=rel2id, rela2rela_id=rela2id,
                                                       ignore_not_mapped_edges=ignore_not_mapped_edges)

    logging.info("Saving the result....")
    save_node_id2terms_list(save_path=output_node_id2terms_list_path, mapping=node_id2terms_list, )
    save_dict(save_path=output_node_id2cui_path, dictionary=node_id2cui)
    save_dict(save_path=output_rel2rel_id_path, dictionary=rel2id)
    save_dict(save_path=output_rela2rela_id_path, dictionary=rela2id)
    save_tuples(save_path=output_edges_path, tuples=edges)


def create_cui2node_id_mapping(mrconso_df: pd.DataFrame) -> Dict[str, int]:
    unique_cuis_set = set(mrconso_df["CUI"].unique())
    cui2node_id: Dict[str, int] = {cui: node_id for node_id, cui in enumerate(unique_cuis_set)}

    return cui2node_id


def create_relations2id_dicts(mrrel_df: pd.DataFrame):
    mrrel_df.REL.fillna("NAN", inplace=True)
    mrrel_df.RELA.fillna("NAN", inplace=True)
    rel2id = {rel: rel_id for rel_id, rel in enumerate(mrrel_df.REL.unique())}
    rela2id = {rela: rela_id for rela_id, rela in enumerate(mrrel_df.RELA.unique())}
    rel2id["LOOP"] = max(rel2id.values()) + 1
    rela2id["LOOP"] = max(rela2id.values()) + 1
    logging.info(f"There are {len(rel2id.keys())} unique RELs and {len(rela2id.keys())} unique RELAs")
    print("REL2REL_ID", rel2id)
    print("RELA2RELA_ID", rela2id)
    return rel2id, rela2id


def create_cui2lang_synonym_list_mapping(cui_lang_synonym_set: Set[Tuple[str, str, str]]) \
        -> Dict[str, List[Tuple[str, str]]]:
    cui2synonyms_list = {}
    for (cui, lang, synonym) in tqdm(cui_lang_synonym_set):
        if cui2synonyms_list.get(cui) is None:
            cui2synonyms_list[cui] = []
        cui2synonyms_list[cui].append((lang, synonym))
    return cui2synonyms_list


def create_lang_aware_tradename_mapping(mrrel_df: pd.DataFrame, cui2synonyms_list: Dict[str, List[Tuple[str, str]]]) \
        -> Dict[str, List[Tuple[str, str]]]:
    tradename_mapping = {}
    for idx, row in tqdm(mrrel_df.iterrows(), total=mrrel_df.shape[0]):
        if row["RELA"] == "has_tradename" or row["RELA"] == "tradename_of":
            cui_1, cui_2 = row["CUI1"], row["CUI2"]
            try:
                sfs = cui2synonyms_list[cui_2]
                tradename_mapping[cui_1] = sfs
            except:
                continue
    return tradename_mapping


def gen_pairs_groupby_language_pair(synonyms_list: List[Tuple[str, str]], eng_crosslingual_only) \
        -> Dict[str, List[Tuple[str, str]]]:
    lang_synonym_tuple_pairs = list(itertools.combinations(synonyms_list, r=2))
    lang_pair_label2synonym_pair: Dict[str, List[Tuple[str, str]]] = {}
    for ((lang_1, synonym_1), (lang_2, synonym_2)) in lang_synonym_tuple_pairs:
        if lang_1 == lang_2:
            label = lang_1
        else:
            if eng_crosslingual_only:
                if lang_1 == "ENG" or lang_2 == "ENG":
                    (lang_1, lang_2) = (lang_2, lang_1) if lang_2 == "ENG" else (lang_1, lang_2)
                    label = f"{lang_1}|{lang_2}"
                else:
                    continue
            else:
                label = "CROSS"
        if lang_pair_label2synonym_pair.get(label) is None:
            lang_pair_label2synonym_pair[label] = []
        lang_pair_label2synonym_pair[label].append((synonym_1, synonym_2))

    return lang_pair_label2synonym_pair


def generate_language_aware_positive_pairs_from_synonyms(concept_id2synonyms_list: Dict[str, List[Tuple[str, str]]],
                                                         max_pairs_eng,
                                                         max_pairs_non_eng,
                                                         max_pairs_crosslingual,
                                                         eng_crosslingual_only) -> List[str]:
    pos_pairs = []
    for concept_id, synonyms_list in tqdm(concept_id2synonyms_list.items()):
        concept_pos_pairs = set()
        # synonym_pairs = gen_pairs(synonyms_list)
        lang_pair_label2synonym_pair = gen_pairs_groupby_language_pair(synonyms_list=synonyms_list,
                                                                       eng_crosslingual_only=eng_crosslingual_only)
        for lang_pair_label, synonym_pair_tuples in lang_pair_label2synonym_pair.items():
            if len(lang_pair_label) == 3:
                assert len(lang_pair_label) == 3
                if lang_pair_label == "ENG":
                    label_limit = max_pairs_eng
                else:
                    label_limit = max_pairs_non_eng
            else:
                label_limit = max_pairs_crosslingual
            if len(synonym_pair_tuples) > label_limit:
                synonym_pair_tuples = random.sample(synonym_pair_tuples, label_limit)
            for (syn_1, syn_2) in synonym_pair_tuples:
                concept_pos_pairs.add(f"{concept_id}||{syn_1}||{syn_2}")
        pos_pairs.extend(concept_pos_pairs)

    return pos_pairs


def generate_positive_pairs(mrconso_df: pd.DataFrame, mrrel_df: pd.DataFrame,
                            cui2node_id: Dict[str, int],
                            max_pairs_eng: int,
                            max_pairs_non_eng: int,
                            max_pairs_crosslingual: int,
                            eng_crosslingual_only: bool) -> List[str]:
    cui_lang_synonym_set: Set[Tuple[str, str, str]] = set()
    for idx, row in tqdm(mrconso_df.iterrows()):
        cui, lang, synonym = row["CUI"], row["LAT"], row["STR"]
        cui_lang_synonym_set.add((cui, lang, synonym.lower()))

    logging.info(f"{len(cui_lang_synonym_set)} <CUI, language, synonym> concepts remaining after duplicates drop.")
    i = 0
    logging.info(f"<CUI, language, synonym> examples:")
    for t in cui_lang_synonym_set:
        logging.info(t)
        if i >= 3:
            break
        i += 1
    cui2lang_synonym_list = create_cui2lang_synonym_list_mapping(cui_lang_synonym_set=cui_lang_synonym_set)
    logging.info(f"Created CUI to synonyms mapping, there are {len(cui2lang_synonym_list.keys())} entries")

    tradename_mapping: Dict[str, List[Tuple[str, str]]] = create_lang_aware_tradename_mapping(mrrel_df=mrrel_df,
                                                                                              cui2synonyms_list=cui2lang_synonym_list)
    logging.info(f"Created tradename mapping, there are {len(tradename_mapping.keys())} entries")
    # adding tradenames
    for cui, synonyms_list in tradename_mapping.items():
        for (lang, synonym) in synonyms_list:
            cui_lang_synonym_set.add((cui, lang, synonym))

    logging.info(f"There are {len(cui_lang_synonym_set)} <CUI, synonym> concepts after tradenames addition")

    cui2lang_synonym_list = create_cui2lang_synonym_list_mapping(cui_lang_synonym_set=cui_lang_synonym_set)

    node_id2synonyms_list = {cui2node_id[cui]: synonyms_list for cui, synonyms_list in cui2lang_synonym_list.items()}
    pos_pairs = generate_language_aware_positive_pairs_from_synonyms(concept_id2synonyms_list=node_id2synonyms_list,
                                                                     max_pairs_eng=max_pairs_eng,
                                                                     max_pairs_non_eng=max_pairs_non_eng,
                                                                     max_pairs_crosslingual=max_pairs_crosslingual,
                                                                     eng_crosslingual_only=eng_crosslingual_only)

    return pos_pairs


def main(args):
    random.seed(42)
    output_dir = args.output_dir
    last_subdir = os.path.basename(output_dir.rstrip('/'))
    output_dir = output_dir.rstrip('/').rstrip(last_subdir)
    logging.info("Loading MRCONSO....")
    mrconso_df = read_mrconso(args.mrconso)
    logging.info(f"MRCONSO is loaded. There are {mrconso_df.shape[0]} rows")

    if args.ontology is not None:
        mrconso_df = mrconso_df[mrconso_df.SAB.isin(args.ontology)]
        last_subdir = '_'.join(args.ontology) + f"_{last_subdir}"
    logging.info(f"There are {mrconso_df.shape[0]} MRCONSO rows after ontology filtering")
    if args.langs is not None:
        mrconso_df = mrconso_df[mrconso_df.LAT.isin(args.langs)]
        last_subdir = '_'.join(args.langs) + f"_{last_subdir}"
    output_dir = os.path.join(output_dir, last_subdir)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    logging.info(f"There are {mrconso_df.shape[0]} MRCONSO rows after language filtering")
    mrconso_df["STR"].fillna('', inplace=True)
    filtered_mrconso_present_cuis_set = set(mrconso_df["CUI"].unique())

    logging.info("Loading MRREL....")
    mrrel_df = read_mrrel(args.mrrel)[["CUI1", "REL", "RELA", "CUI2"]]
    logging.info(f"Removing MRREL duplicated rows. There are {mrrel_df.shape[0]} rows with duplicates")
    mrrel_df.drop_duplicates(inplace=True)
    logging.info(f"Filtering MRREL by CUI1 and CUI2 fields. There are {mrrel_df.shape[0]} rows before filtering")
    mrrel_df = mrrel_df[(mrrel_df['CUI1'].isin(filtered_mrconso_present_cuis_set)) & (
        mrrel_df['CUI2'].isin(filtered_mrconso_present_cuis_set))]
    logging.info(f"Finished filtering MRREL by CUI1 and CUI2 fields. "
                 f"There are {mrrel_df.shape[0]} rows after filtering")
    rel2id, rela2id = create_relations2id_dicts(mrrel_df)
    cui2node_id = create_cui2node_id_mapping(mrconso_df=mrconso_df)

    logging.info("Creating graph files")
    output_node_id2terms_list_path = os.path.join(output_dir, "node_id2terms_list")
    output_node_id2cui_path = os.path.join(output_dir, "id2cui")
    output_edges_path = os.path.join(output_dir, "edges")
    output_rel2rel_id_path = os.path.join(output_dir, f"rel2id")
    output_rela2rela_id_path = os.path.join(output_dir, f"rela2id")

    create_graph_files(mrconso_df=mrconso_df, mrrel_df=mrrel_df, rel2id=rel2id, rela2id=rela2id,
                       cui2node_id=cui2node_id,
                       output_node_id2terms_list_path=output_node_id2terms_list_path,
                       output_node_id2cui_path=output_node_id2cui_path,
                       output_edges_path=output_edges_path, output_rel2rel_id_path=output_rel2rel_id_path,
                       output_rela2rela_id_path=output_rela2rela_id_path, ignore_not_mapped_edges=True, )
    pos_pairs = generate_positive_pairs(mrconso_df=mrconso_df, mrrel_df=mrrel_df, cui2node_id=cui2node_id,
                                        max_pairs_eng=args.max_pairs_eng,
                                        max_pairs_non_eng=args.max_pairs_non_eng,
                                        max_pairs_crosslingual=args.max_pairs_crosslingual,
                                        eng_crosslingual_only=args.eng_crosslingual_only)
    if args.split_val:
        output_train_pos_pairs_path = os.path.join(output_dir, f"train_pos_pairs")
        output_val_pos_pairs_path = os.path.join(output_dir, f"val_pos_pairs")

        train_proportion = args.train_proportion
        num_pos_pairs = len(pos_pairs)
        # Shuffling positive pairs
        random.shuffle(pos_pairs)
        num_train_pos_pairs = int(num_pos_pairs * train_proportion)
        train_pos_pairs = pos_pairs[:num_train_pos_pairs]
        val_pos_pairs = pos_pairs[num_train_pos_pairs:]
        logging.info(f"Positive pairs are split: {len(train_pos_pairs)} in train and {len(val_pos_pairs)} in val")
        logging.info(f"Train positive pair examples:\n" + '\n'.join(train_pos_pairs[:3]))
        logging.info(f"Val positive pair examples:\n" + '\n'.join(val_pos_pairs[:3]))
        write_strings(fpath=output_train_pos_pairs_path, strings_list=train_pos_pairs)
        write_strings(fpath=output_val_pos_pairs_path, strings_list=val_pos_pairs)
    else:
        random.shuffle(pos_pairs)
        output_pos_pairs_path = os.path.join(output_dir, f"train_pos_pairs")
        logging.info(f"Positive pair examples:\n" + '\n'.join(pos_pairs[:3]))
        write_strings(fpath=output_pos_pairs_path, strings_list=pos_pairs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--mrconso', type=str,
                        help="Path to MRCONSO.RRF file of the UMLS metathesaurus")
    parser.add_argument('--mrrel', type=str,
                        help="Path to MRREL.RRF file of the UMLS metathesaurus")
    parser.add_argument('--langs', nargs='+', default=None,
                        help="UMLS language codes to keep. All languages are kept if not specified")
    parser.add_argument('--split_val', action="store_true",
                        help="Whether perform train/validation split of synonymous (positive) concept name pairs")
    parser.add_argument('--eng_crosslingual_only', action="store_true",
                        help="Whether reduce all cross-lingual interactions to pairs with an English concept name")
    parser.add_argument('--max_pairs_eng', type=int,
                        help="Maximum number of English synonymous (positive) concept name pairs for a given "
                             "concept (for a fixed UMLS CUI)")
    parser.add_argument('--max_pairs_non_eng', type=int,
                        help="Maximum number of non-English synonymous (positive) concept name pairs for a given "
                             "concept and language (for a fixed <UMLS CUI, language> pair)")
    parser.add_argument('--max_pairs_crosslingual', type=int,
                        help="Maximum number of cross-lingual synonymous (positive) concept name pairs for a given "
                             "concept (for a fixed UMLS CUI)")
    parser.add_argument('--train_proportion', type=float,
                        help="Train ratio if split_val argument is True")
    parser.add_argument('--ontology', default=None, nargs='+',
                        help="List of UMLS SABs (Source ABbreviations) to keep for concept name pairs creation")
    parser.add_argument('--output_dir', type=str)
    arguments = parser.parse_args()

    main(arguments)
