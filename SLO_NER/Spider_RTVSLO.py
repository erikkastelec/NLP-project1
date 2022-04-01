import errno
import logging
import os
import random
from os import path
from time import sleep

import networkx as nx
import w3lib.html
from inline_requests import inline_requests
from scrapy import Spider, Request
from scrapyscript import Job, Processor

from helper_functions import slo_month_to_num, write_pickle


class Spider_RTVSLO(Spider):
    name = 'Spider_RTVSLO'

    # enable fake-useragent python package
    custom_settings = {
        "DOWNLOADER_MIDDLEWARES": {
            'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
            'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,
            'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware': 400,
            'scrapy_fake_useragent.middleware.RetryUserAgentMiddleware': 401,
        },
        'FAKEUSERAGENT_PROVIDERS': [
            'scrapy_fake_useragent.providers.FakeUserAgentProvider',  # this is the first provider we'll try
            'scrapy_fake_useragent.providers.FakerProvider',
            # if FakeUserAgentProvider fails, we'll use faker to generate a user-agent string for us
            'scrapy_fake_useragent.providers.FixedUserAgentProvider',  # fall back to USER_AGENT value
        ]
    }

    def __init__(self, category, page, url, **kw):
        self.category = category
        self.page = page
        self.log = logging.getLogger('scrapy')
        self.log.setLevel(logging.WARNING)
        self.start_urls = [url]
        super().__init__(**kw)

    @inline_requests
    def parse(self, response):
        text = ""
        yield_out = {
            "continue": True
        }
        # GET URLS FOR ALL THE ARTICLES ON THE PAGE
        articles = response.xpath('//div[@class="article-archive-item"]//div[@class="md-news"]/h3/a')
        if len(articles) != 20:
            yield_out["continue"] = False
        for article in articles:
            try:
                article_resp = yield Request("https://www.rtvslo.si" + article.attrib["href"])
                article_text = ([w3lib.html.remove_tags(x).strip('\xa0').strip("\'") for x in
                                 article_resp.xpath('//article//p').extract() if
                                 x is not None])
                text = ' '.join(article_text)
                # example [12. maj 2021]
                article_date = (article_resp.xpath('//div[@class="publish-meta"]/text()').get()).strip('\n').replace(
                    '.', '').split(" ")[0:3]
                # convert string month to int
                article_date[1] = str(slo_month_to_num(article_date[1]))
                filename = './data/RTVSLO/' + article_date[2] + '/' + ".".join(article_date[:2]) + \
                           '_'.join((article.attrib["href"].split('/'))) + ".txt"
                # Create dir structure to write into
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                # Write extracted text to file
                try:
                    if not path.exists(filename):
                        with open(filename, 'w') as f:
                            f.write(text)
                    else:
                        break
                except:
                    self.log.error("Error saving file: ", filename)
                # Simple rate limiting
                sleep(random.uniform(0.5, 1))
            except:
                self.log.info("Failed request %s", article.attrib["href"], exc_info=True)

        # Save progress
        write_pickle((self.category, self.page), './progress/RTVSLO.pickle')
        yield yield_out


if __name__ == '__main__':
    graph = nx.MultiGraph()
    job = Job(Spider_RTVSLO, 'slovenija', 0, "https://www.rtvslo.si/slovenija/arhiv")
    processor = Processor(settings=None)
    data = processor.run([job])
    # try:
    #     data = processor.run([wemportalJob])[0]
    #     classla.download('sl')
    #     nlp = classla.Pipeline("sl", processors='tokenize,ner')
    #     doc = nlp(data["res"][0])
    #     print(doc.to_conll())
    # except IndexError:
    #     print("err")
