import requests
from bs4 import BeautifulSoup
from gensim.models import LdaMulticore, CoherenceModel
from collections import defaultdict


def Amazon_search(path, headers, cookies, proxies):
    url = "https://www.amazon.com" + path
    try:
        r = requests.get(url, headers=headers, cookies=cookies, proxies=proxies)
        return r
    except requests.exceptions.RequestException as e:  # If response is not 200, debug
        raise SystemExit(e)


def Amazon_crawler(max_reviews_pages, max_asin_pages, headers, cookies, proxies):
    # Search and store the asin of each product
    data_asin = []
    for asin_page in range(max_asin_pages):
        response = Amazon_search("/s?k=electronics" + '&page=' + str(asin_page), headers, cookies, proxies)
        soup = BeautifulSoup(response.content, features="html.parser")
        # By using the class_="s-widget-spacing-small" we are scrapping each of the products
        for i in soup.find_all(name="div", class_="s-widget-spacing-small"):
            strclass = " ".join(i["class"])
            # If a product has an AdHolder class, it is ignored, as there are duplicates in each page
            if "AdHolder" not in strclass:
                data_asin.append(i["data-asin"])
    print("Number of unique products: {} \nasin context: {}\n".format((len(data_asin)), data_asin))

    # Search and store the url of each product
    href = []
    for i in range(len(data_asin)):
        response = Amazon_search("/dp/" + data_asin[i], headers, cookies, proxies)
        soup = BeautifulSoup(response.content, features="html.parser")
        for i in soup.findAll(name="a", attrs={'data-hook': "see-all-reviews-link-foot"}):
            href.append(i['href'])
    print("Number of unique urls: {} \nurl context: {}\n".format((len(href)), href))

    # Search and store the amazon reviews of each product
    total_reviews = defaultdict(list)
    name = "reviewsText"
    for review in range(len(href)):
        for review_page in range(max_reviews_pages):
            response = Amazon_search(href[review] + '&pageNumber=' + str(review_page), headers, cookies, proxies)
            soup = BeautifulSoup(response.content, features="html.parser")
            for i in soup.findAll(name="span", attrs={'data-hook': "review-body"}):
                total_reviews[name].append(i.text)
    print("Number of reviews: {}".format(len(total_reviews["reviewsText"])))

    return total_reviews


# compute coherence value at various values of decay and num_topics
def grid_search_coherence(corpus, dictionary, texts, offset_range, learning_decay_range, num_topics_range):
    coherence_values = []
    model_list = []
    for num_topics in num_topics_range:
        for learning_decay in learning_decay_range:
            for offset in offset_range:
                lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, chunksize=len(corpus),
                                         offset=offset, random_state=42, decay=learning_decay, passes=15,
                                         per_word_topics=True)
                model_list.append(lda_model)

                coherencemodel = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append((offset, learning_decay, num_topics, coherencemodel.get_coherence(),
                                         lda_model.log_perplexity(corpus)))  # a measure of how good the model is. The lower the better.

    return model_list, coherence_values
