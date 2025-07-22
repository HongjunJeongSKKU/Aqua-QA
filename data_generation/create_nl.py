import os

import transformers
import torch
import pandas as pd
import math
from tqdm import tqdm
import numpy as np
import argparse
import click

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import set_seed


@click.command()
@click.option("--dataset", default="CODEX")
@click.option("--gen_train", is_flag=True, default=False)
@click.option("--gen_valid", is_flag=True, default=False)
@click.option("--gen_test", is_flag=True, default=False)
@click.option("--query_type", default="1p.2p.3p.2i.3i.2in.3in.inp.pin.pni.ip.pi.2u.up")
@click.option("--llm_model", default="llama")
def main(dataset, gen_train, gen_valid, gen_test, query_type, llm_model):

    print(f"Received query type: {query_type}")
    print(f"LLM model: {llm_model}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    queries = [x for x in query_type.split(".")]

    if llm_model == "llama":

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            bnb_4bit_compute_dtype=torch.bfloat16,
            padding_side="left",
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            device_map="auto",
            quantization_config=quantization_config,
        )

    set_seed(0)

    if gen_train:
        mode = "train"
        queries = [x for x in queries if x not in ["pi", "ip", "up"]]
    elif gen_valid:
        mode = "valid"
    elif gen_test:
        mode = "test"
    else:
        print("Set the mode (--gen_train, --gen_valid, --gen_test)")
        exit(1)

    fol_df = pd.read_csv(f"data/{dataset}/{mode}.csv")

    FOL = {
        "1p": {
            "query": "𝑞 = X? . ∃X: r(e, X?)",
            "example": """𝑞 = X? . ∃X: (breakup_is_participated_by_celebrity('Howard Hughes', X?)) => Who did Howard Hughes participate in a breakup with?
𝑞 = X? . ∃X: (actor_in_regular_tv_appearance_is_a_regular_cast_of_tv_program('Will Arnett', X?)) => What TV programs is Will Arnett a regular cast member of?
𝑞 = X? . ∃X: (prequel_is_a_preceding_version_of_film('National Treasure', X?)) => What are the prequels to the film "National Treasure"?""",
        },
        "2p": {
            "query": "𝑞 = Y? . ∃X ∃Y: r(e, X) ∧ r(X, Y?)",
            "example": """𝑞 = Y? . ∃X ∃Y: (place_lived_is_a_location_of_person('Franklin', X) ∧ award_nominee_is_an_award_nomination_of_an_award_nominee(X, Y?)) => Who are the award nominees nominated by Franklin?
𝑞 = Y? . ∃X ∃Y: (performance_role_has_track_performances_who_are_track_contributions('String', X) ∧ dataworld_has_gardening_hint_split_to(X, Y?)) => What are the gardening hints related to the track performances by 'String'?
𝑞 = Y? . ∃X ∃Y: (award_honor_is_awarded_to_award_winning_work('Daytime Emmy Award for Outstanding Talk Show Host', X) ∧ tv_personality_in_tv_regular_personal_appearance_is_a_program(X, Y?)) => What TV programs feature personal appearances by winners of the Daytime Emmy Award for Outstanding Talk Show Host?""",
        },
        "3p": {
            "query": "𝑞 = Z? . ∃X ∃Y ∃Z: (r(e, X) ∧ r(X, Y)) ∧ r(Y, Z?)",
            "example": """𝑞 = Z? . ∃X ∃Y ∃Z: (award_winner_is_winner_of_award_honor_in_award_category('Don DeLillo', X) ∧ award_category_has_category_of(X, Y)) ∧ award_is_an_award_nomination_for_an_award(Y, Z?)) => What are the award nominations for the categories in which Don DeLillo won awards?
𝑞 = Z? . ∃X ∃Y ∃Z: (film_has_performance_by_actor_in_film('Terry Crews', X) ∧ titles_are_in_netflix_genre(X, Y)) ∧ people_with_this_degree_are_in_major_field_of_study_in_education(Y, Z?)) => What are the major fields of study for people with degrees related to films on Netflix in which Terry Crews has performed?
𝑞 = Z? . ∃X ∃Y ∃Z: (team_is_in_sports_team_roster_of_current_team_of_football_player('De Graafschap', X) ∧ person_has_profession(X, Y)) ∧ profession_has_specialization(Y, Z?)) => What are the specializations within the professions of football players currently on the roster of De Graafschap?""",
        },
        "2i": {
            "query": "𝑞 = X? . ∃X : (r(e, X?) ∧ r(e, X?))",
            "example": """𝑞 = X? . ∃X : (disciplines_or_subjects_are_a_discipline_or_subject_of_award_category('Literature', X?) ∧ award_winner_is_winner_of_award_honor_in_award_category('James A. Michener', X?)) => What awards in the category of Literature has James A. Michener won?
𝑞 = X? . ∃X : (artists_are_in_genre('John Fogerty', X?) ∧ genre_is_a_parent_of_parent_genre('Soft rock', X?)) => What genres related to Soft rock is John Fogerty associated with?
𝑞 = X? . ∃X : (instrument_has_instrumentalists('Guitar', X?) ∧ profession_is_had_by_person('Composer', X?)) => What composers are instrumentalists for the guitar?""",
        },
        "3i": {
            "query": "𝑞 = X? . ∃X: (r(e, X?) ∧ r(e, X?) ∧ r(e, X?))",
            "example": """𝑞 = X? . ∃X: (profession_is_had_by_person('Television Director', X?) ∧ award_winner_is_winner_of_award_honor('Jennifer Celotta', X?) ∧ award_nominee_is_an_award_nomination_of_an_award_nominee('Angela Kinsey', X?)) => What awards have Jennifer Celotta won that Angela Kinsey was nominated for, in their shared profession of Television Director?
𝑞 = X? . ∃X: (sport_is_played_by_sports_team('Football', X?) ∧ position_is_held_by_football_roster_of_football_team('Goalkeeper', X?) ∧ football_player_is_in_current_team_of_team('Chris Waddle', X?)) => Which football teams has Chris Waddle played for as a goalkeeper?
𝑞 = X? . ∃X: (adjoining_relationship_is_adjoin_s_of_location('Gelderland', X?) ∧ location adjoins adjoining relationship.('Friesland', X?) ∧ location contains location('Netherlands', X?)) => What locations in the Netherlands adjoin both Gelderland and Friesland?""",
        },
        "2in": {
            "query": "𝑞 = X? . ∃X: (r(e, X?) ∧ ¬r(e, X?))",
            "example": """𝑞 = X? . ∃X: (actor_is_in_performance_in_film('Rango', X?) ∧ ¬celebrity_dated_participant('Rachael Leigh Cook', X?)) => Who are the actors in "Rango" who have not dated Rachael Leigh Cook?
𝑞 = X? . ∃X: (actor_is_in_performance_in_film('Wild at Heart', X?) ∧ ¬participant_was_dated_by_celebrity('Ben Harper', X?)) => Who are the actors in "Wild at Heart" who have not been dated by Ben Harper?
𝑞 = X? . ∃X: (people_cause_of_death('Pneumonia', X?) ∧ ¬profession_is_had_by_person('Television producer-GB', X?)) => Who are the people whose cause of death was pneumonia and are not television producers in Great Britain?""",
        },
        "3in": {
            "query": "𝑞 = X? . ∃X: (r(e, X?) ∧ r(e, X?) ∧ ¬r(e, X?))",
            "example": """𝑞 = X? . ∃X: (film_has_performance_by_actor_in_film('Helen Mirren', X?) ∧ country_has_film('Germany', X?) ∧ ¬netflix_genre_has_titles('Psychological thriller', X?)) => What films featuring Helen Mirren are available in Germany but are not Psychological thrillers on Netflix?
𝑞 = X? . ∃X: (award_honor_ceremony_has_winners_of_award_category('41st Annual Grammy Awards', X?) ∧ category_of_is_category_of_award('Grammy Awards', X?) ∧ ¬award_honor_ceremony_has_winners_of_award_category('54th Annual Grammy Awards-US', X?)) => Which awards were won at the 41st Annual Grammy Awards and are part of the Grammy Awards category, but not won at the 54th Annual Grammy Awards-US?
𝑞 = X? . ∃X: (genre_is_a_genre_of_film('Biographical film', X?) ∧ country_has_film('United States of America', X?) ∧ ¬costume_design_by_is_performed_by_film('Irene Sharaff', X?)) => What Biographical films are available in the United States that do not involve costume design by Irene Sharaff?""",
        },
        "inp": {
            "query": "𝑞 = Y? . ∃X ∃Y : ((r(e, X) ∧ ¬r(e, X)) ∧ r(X, Y?))",
            "example": """𝑞 = Y? . ∃X ∃Y : ((medicine_has_notable_people_with_disease('Dyslexia', X) ∧ ¬place_of_birth_is_place_of_birth_of_person('Manhattan', X)) ∧ award_winner_is_awarded_with_award_honor(X, Y?)) => What awards have been won by notable people with dyslexia who were not born in Manhattan?
𝑞 = Y? . ∃X ∃Y : ((award_honor_ceremony_has_winners_of_award_category('55th Primetime Emmy Awards', X) ∧ ¬nominated_for_is_a_nominee_of_award_category('The Sopranos', X)) ∧ award_category_has_winners_who_win_award_honor_at_ceremony(X, Y?)) => Who are the winners of the 55th Primetime Emmy Awards who were not nominated for 'The Sopranos'?
𝑞 = Y? . ∃X ∃Y : ((tv_producer_has_programs_produced_which_are_program('Manny Coto', X) ∧ ¬netflix_genre_has_titles('Showtime', X)) ∧ award_winning_work_has_awards_won_which_are_award_honors_of_award(X, Y?)) => What awards have programs produced by Manny Coto won, excluding those in the Showtime genre?""",
        },
        "pin": {
            "query": "𝑞 = Y? . ∃X ∃Y : ((r(e, X) ∧ r(X, Y?)) ∧ ¬r(e, Y?))",
            "example": """𝑞 = Y? . ∃X ∃Y : ((netflix_genre_has_titles('Hong Kong', X) ∧ film_film_distributor_relationship_is_a_relationship_between_film_and_films_distributed_by_film_distributor(X, Y?)) ∧ ¬citytown_is_mailing_address_of_headquarters_of_organization('New York City', Y?)) => What films distributed from Hong Kong genres are not headquartered in New York City?
𝑞 = Y? . ∃X ∃Y : ((nationality_is_held_by_person('United States of America', X) ∧ pro_athlete_is_a_member_of_sports_team_roster(X, Y?)) ∧ ¬pro_athlete_is_a_member_of_sports_team_roster('Stephen Jackson', Y?)) => Which athletes from the United States are not members of the sports teams associated with Stephen Jackson?
𝑞 = Y? . ∃X ∃Y : ((artist_has_origin('Everything but the Girl', X) ∧ location_is_contained_by_location(X, Y?)) ∧ ¬country_has_second_level_divisions('England', Y?)) => What larger locations contain places related to Everything but the Girl and are not divisions of England?""",
        },
        "pni": {
            "query": "𝑞 = Y? . ∃X ∃Y : ((r(e, X) ∧ ¬r(X, Y?)) ∧ r(e, Y?))",
            "example": """𝑞 = Y? . ∃X ∃Y : ((user_has_default_domain_for_olympic_games_sports('1948 Winter Olympics', X) ∧ ¬olympic_sport_has_athletes_who_are_olympic_athlete_affiliation_of_olympics(X, Y?)) ∧ sports_are_part_of_olympic_games('Biathlon', Y?)) => Which biathlon events were part of the 1948 Winter Olympics but did not include athletes associated with any sports from the default Olympic domains?
𝑞 = Y? . ∃X ∃Y : ((category_is_a_category_of_webpage('Official Website', X) ∧ ¬currency_is_used_for_net_worth_of_person_extra_in_dated_money_value(X, Y?)) ∧ award_nominee_is_an_award_nomination_of_an_award_nominee('Ann Curry', Y?)) => Which awards has Ann Curry been nominated for that are not associated with currencies used on her official website?
𝑞 = Y? . ∃X ∃Y : ((olympics_olympic_athlete_affiliation_is_affiliated_with_olympics('2006 Winter Olympics', X) ∧ ¬sport_is_played_by_sports_team(X, Y?)) ∧ sports_team_location_has_teams('Dallas', Y?)) => Which sports teams based in Dallas did not participate in the 2006 Winter Olympics?""",
        },
        "ip": {
            "query": "𝑞 = Y? . ∃X ∃Y : ((r(e, X) ∧ r(e, X)) ∧ r(X, Y?))",
            "example": """𝑞 = Y? . ∃X ∃Y : ((place_of_birth_is_place_of_birth_of_person('Neath', X) ∧ profession_is_had_by_person('Television producer-GB', X)) ∧ tv_producer_has_programs_produced_which_are_program(X, Y?)) => What programs have been produced by television producers who were born in Neath and have the profession of "Television producer" in Great Britain?
𝑞 = Y? . ∃X ∃Y : ((person_has_nationality('Jagdish Raj', X) ∧ film_has_release_date_s_which_is_a_film_regional_release_date_in_film_release_region('Abraham Lincoln: Vampire Hunter', X)) ∧ release_date_s_is_release_date_of_film_release_region(X, Y?)) => What are the release dates in different regions for the film "Abraham Lincoln: Vampire Hunter" that match the nationality of a person named Jagdish Raj?
𝑞 = Y? . ∃X ∃Y : ((influence_node_is_influenced_by('Dylan McDermott', X) ∧ award_winning_work_has_awards_won_which_are_award_honors_won_by_award_winner('The Godfather', X)) ∧ person_has_profession(X, Y?)) => What professions are held by individuals who influenced Dylan McDermott and have won awards for "The Godfather"?""",
        },
        "pi": {
            "query": "𝑞 = Y? . ∃X ∃Y : ((r(e, X) ∧ r(X, Y?)) ∧ r(e, Y?))",
            "example": """𝑞 = Y? . ∃X ∃Y : ((school_type_is_had_by_educational_institution('University-preparatory school', X) ∧ location_is_contained_by_location(X, Y?)) ∧ school_type_is_had_by_educational_institution('German Shepherd Dog', Y?)) => What locations contain an educational institution that is a university-preparatory school, and also have an educational institution with the type "German Shepherd Dog"?
𝑞 = Y? . ∃X ∃Y : ((organization_has_headquarters_in_citytown('Pan American World Airways', X) ∧ sports_team_location_has_teams(X, Y?)) ∧ company_was_Employed_by_person('Forward-center', Y?)) => Which sports teams are located in a city or town where Pan American World Airways has its headquarters, and have employed a person with the position "Forward-center"?
𝑞 = Y? . ∃X ∃Y : ((student_is_a_graduate_of_educational_institution('Swoosie Kurtz', X) ∧ school_has_draft_picks_in_sports_league_draft_pick(X, Y?)) ∧ film_has_performance_by_actor_in_film('The Bronx', Y?)) => What sports league draft picks have been made from a school that Swoosie Kurtz graduated from, and have performances in films set in "The Bronx"?""",
        },
        "2u": {
            "query": "𝑞 = X? . ∃X : (r(e, X?) ∨ r(e, X?))",
            "example": """𝑞 = X? . ∃X : (location_is_contained_by_location('Gaston County', X?) ∨ adjoining_relationship_is_adjoin_s_of_location('Virginia', X?)) => What are the locations that are either contained within Gaston County or are adjoining locations of Virginia?
𝑞 = X? . ∃X : (celebrities_impersonated_are_impressed_by_celebrity_impressionist('Bugs Bunny', X?) ∨ profession_is_had_by_person('Actor-GB', X?)) => Who are the people that either impersonate Bugs Bunny or have the profession of an actor
𝑞 = X? . ∃X : (tv_program_has_program_creator('Mighty Morphin Power Rangers', X?) ∨ place_of_birth_is_place_of_birth_of_person('Alexandria', X?)) => Who are the individuals that either created the TV program "Mighty Morphin Power Rangers" or were born in Alexandria?""",
        },
        "up": {
            "query": "𝑞 = Y? . ∃X ∃Y : ((r(e, X) ∨ r(e, X)) ∧ r(X, Y?))",
            "example": """𝑞 = Y? . ∃X ∃Y : ((person_has_profession('Friedrich Schiller', X) ∨ person_has_profession('Bill Cosby', X)) ∧ profession_is_had_by_person(X, Y?)) => What professions are associated with people who either have the same profession as Friedrich Schiller or Bill Cosby?
𝑞 = Y? . ∃X ∃Y : ((location_is_contained_by_location('Irvine', X) ∨ person_has_nationality('T. S. Eliot', X)) ∧ country_has_film(X, Y?)) => What films are associated with countries that either contain the location Irvine or have a person with the nationality of T. S. Eliot?
𝑞 = Y? . ∃X ∃Y : ((team_has_players_who_play_sports_position('New Jersey Devils', X) ∨ hockey_team_has_current_roster_with_position('Syracuse Crunch', X)) ∧ sports_position_has_players_who_are_in_sports_team_roster(X, Y?)) => What players hold sports positions that are either played by members of the New Jersey Devils or are part of the current roster of the Syracuse Crunch?""",
        },
    }

    batch_size = 40
    final_data = pd.DataFrame()
    for q_type in tqdm(queries):
        print(q_type)

        result = []  # for generated text
        for j in tqdm(
            range(math.ceil(len(fol_df[q_type + "_FOL"].dropna()) / batch_size))
        ):

            batch_messages = []  # for batch sample
            for i in range(batch_size):  # batch size
                index = i + j * batch_size
                if index >= len(fol_df[q_type + "_FOL"]) or type(
                    fol_df[q_type + "_FOL"][index]
                ) == type(np.nan):
                    break

                messages1 = [
                    {
                        "role": "system",
                        "content": "You are a natural language generator. Your task is to understand a First-Order Logic (FOL) query and convert it into a natural language question.",
                    },
                    {
                        "role": "user",
                        "content": f"""Please convert the following First-Order Logic (FOL) query into a natural language question. Provide only one natural language query as output, without any additional explanation.

FOL Query Structure:
{FOL[q_type]["query"]}

Examples of Conversion:
{FOL[q_type]["example"]}

FOL Query to Convert:
{fol_df[q_type + '_FOL'][index]} =>""",
                    },
                ]

                batch_messages.append(messages1)

            model_inputs = tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            input_length = model_inputs.shape[1]
            generated_ids = model.generate(
                model_inputs, max_new_tokens=300, min_length=10
            )

            output_texts = tokenizer.batch_decode(
                generated_ids[:, input_length:], skip_special_tokens=True
            )
            result.extend(output_texts)

        result = pd.DataFrame(result)
        result.columns = [q_type + "_FOL_NL"]
        results = pd.concat([fol_df[q_type + "_FOL"], result], axis=1)
        final_data = pd.concat([final_data, results], axis=1)

        # results.to_csv(f"data/{dataset}/{model}_{q_type}.csv", index=False)
    final_data.to_csv(f"data/{dataset}/{llm_model}_{mode}_nl.csv", index=False)


if __name__ == "__main__":
    main()
