import requests
import pandas as pd

TK_DATA_PATH = "./data/tk.csv"


def href_to_title(href):
    try:
        href = href.replace('https://', '') \
            .replace('http://', '') \
            .replace('www.', '') \
            .split('/')[0].upper()
        return href
    except Exception as e:
        print(f"[ERROR {__name__}] {e}")
        return href


def check_sources(list_of_sources):
    list_of_texts = []
    for s in list_of_sources:
        print(f"Checking {s}...")
        try:
            r = requests.get(s)
            if r.status_code == 200:
                text_from_source = r.text
                list_of_texts.append(text_from_source)
                print(f"Source {s}: {len(text_from_source)} symbols")
                print("First 1000 symbols:", text_from_source[:1000])
            else:
                print(f"Error {r.status_code}")
        except Exception as e:
            print(f"Error {e}")


def add_tk_sources(list_of_sources, tk_data_path=TK_DATA_PATH, verbose=False):
    try:
        df = pd.read_csv(tk_data_path, index_col=2)
        list_of_sources_updated = {}
        for s in list_of_sources:
            if s == 'OpenAI':
                list_of_sources_updated[s] = {'href': 'https://chat.openai.com/',
                                              'name': 'OpenAI GPT-3.5 Turbo w/o context'}
                print(f"Skipping {s}...")
                continue
            s = 'ст.' + s.split('ст.')[1]
            if s in df.index:
                s_data = df.loc[s, ['href', 'name']]
                if s_data.empty:
                    continue
                else:
                    list_of_sources_updated[s] = s_data.to_dict()
            else:
                s = href_to_title(s)
                list_of_sources_updated[s] = {'href': s, 'name': s}

        # If verbose, print first 10 sources
        if verbose:
            for idx, s in enumerate(list_of_sources_updated):
                print(f"\033[091m{s}\033[0m: {list_of_sources_updated[s]}")
                if idx > 10:
                    break
        print(f"Added {len(list_of_sources_updated.items())} sources {list_of_sources_updated}")
        return list_of_sources_updated

    except Exception as e:
        print(f"[ERROR {__name__}] {e}")
        return list_of_sources


def add_yt_sources(list_of_sources, verbose=False):
    youtube_url = 'https://www.youtube.com/watch?v='
    list_of_sources_updated = {}
    try:
        for s in list_of_sources:
            source_name, youtube_id = s.split('[')
            youtube_id = youtube_id.replace(']', '')
            source_name = source_name.strip()
            list_of_sources_updated[source_name] = {'href': youtube_url + youtube_id, 'name': source_name}
        # If verbose, print first 10 sources
        if verbose:
            for idx, s in enumerate(list_of_sources_updated):
                print(f"\033[091m{s}\033[0m: {list_of_sources_updated[s]}")
                if idx > 10:
                    break
        return list_of_sources_updated

    except Exception as e:
        print(f"[ERROR {__name__}] {e}")
        return list_of_sources


def enrich_sources(list_of_sources, topic="tk", verbose=False):
    print(f"Enriching sources for topic {topic}...") if verbose else None
    if topic not in ['tk', 'yt']:
        print(f"Unknown topic {topic}. Returning original list of sources.") if verbose else None
        return list_of_sources
    elif topic == 'tk':
        tk_enriched = add_tk_sources(list_of_sources, verbose=verbose)
        print(f"Enriched sources: {tk_enriched}") if verbose else None
        return tk_enriched
    elif topic == 'yt':
        yt_enriched = add_yt_sources(list_of_sources, verbose=verbose)
        print(f"Enriched sources: {yt_enriched}") if verbose else None
        return yt_enriched


if __name__ == "__main__":
    TEST_QUESTION = "Какая стоимость доставки в Москве и Московской области?"
    TEST_SOURCES1 = [
        'ст. 71 ТК РФ',
        'ч. 2 ст. 221 ТК РФ',
        'ч. 4 ст. 256 ТК РФ',
        'ч. 5 ст. 80 ТК РФ',
        'ч. 8 ст. 178 ТК РФ',
        'ст. 111 ТК РФ',
        'ч. 2 ст. 127 ТК РФ',
        'ст. 352 ТК РФ',
        'ст. 123 ТК РФ',
        'ст. 199 ТК РФ']
    TEST_SOURCES2 = [
        "Europe's Energy Nightmare Has Only Just Begun [AySC0LbRP7M]",
        'Our Electrifying Future ｜ FT Rethink [wIpbK2Lg1sY]',
        'Война экономик, удар по среднему классу и разрыв с Европой： Михаил Задорнов [Q685TJHOiaM]'
    ]

    # answers, sources = answer_with_openai(TEST_QUESTION, verbose=True, api_key=OPENAI_API_KEY)
    # print(answers)
    # print(sources)

    tk_enriched = add_tk_sources(TEST_SOURCES1, tk_data_path="./../data/tk.csv", verbose=True)
    print(tk_enriched)
    # print(tk_enriched['ст. 71 ТК РФ'])

    # yt_enriched = add_yt_sources(TEST_SOURCES2, verbose=True)
    # print(yt_enriched.get('Our Electrifying Future ｜ FT Rethink [wIpbK2Lg1sY]'))
